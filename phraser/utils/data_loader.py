import glob
import json
from functools import reduce
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phraser.utils.utils import STEPS_PER_SEC, get_index, SECTION_LABELS, MUQ_HZ, BEATS_HZ
from phraser.paths import CORPUS, is_excluded


#
# def collate_fn_with_preprocess(batch):
#     """
#     Collate function for wdsdataloader.
#     batch: a list of dict, each dict is a sample
#     """
#     waveforms = []
#     section_encodings = []
#     section_type_encodings = []
#     urls = []
#     segments = []
#     stem_splits = []
#     stem_silences = []
#     on_set_encoding = []
#     beats_encoding = []
#
#     for sample in batch:
#         waveforms.append(sample['waves'].unsqueeze(0))
#         section_encodings.append(sample['segments_split_frames_encoded'])
#         section_type_encodings.append(sample['segments_encoded'])
#         urls.append(sample['path'])
#         segments.append(sample['segments'])
#         stem_splits.append(sample['stem_splits'].unsqueeze(0))
#         stem_silences.append(sample['stem_silences'].unsqueeze(0))
#         on_set_encoding.append(sample['on_set_encoding'])
#         beats_encoding.append(sample['beats_encoding'])
#
#     return torch.concatenate(waveforms), torch.concatenate(section_encodings), torch.concatenate(section_type_encodings), torch.concatenate(stem_splits), torch.concatenate(stem_silences), torch.concatenate(on_set_encoding), torch.concatenate(beats_encoding), urls, segments



class CollateEmbeddingsWithPreprocess:
    def __init__(self, length_pattern=None):
        self.length_pattern = length_pattern
        self.batch_idx = 0

    def set_epoch(self, epoch):
        self.batch_idx = 0

    def __call__(self, batch):
        """
        Collate function for wdsdataloader.
        batch: a list of dict, each dict is a sample
        """
        # Determine target length for the current batch
        if self.length_pattern:
            target_length = self.length_pattern[self.batch_idx % len(self.length_pattern)]
            self.batch_idx += 1
            if target_length == -1:
                # Random length between 30 and 300s as per instructions (-1 means random)
                target_length = np.random.randint(30, 301)
        else:
            target_length = 300  # Default 5 min

        # Since all samples in batch are prepared as 300s (segment_size),
        # we pick a common random start point for the batch if target_length < 300
        max_start = max(0, 300 - target_length)
        start_sec = int(np.random.uniform(0, max_start)) if max_start > 0 else 0

        embeddings = []
        segments = []
        segments_type = []
        segments_splits = []
        segments_splits_long = []
        beats = []
        on_set = []
        stem_splits = []
        stem_splits_long = []
        stem_silences = []
        path = []

        for sample in batch:
            # Slicing indices for different frequencies
            # embeddings: MUQ_HZ (25 Hz)
            e_start = int(start_sec * MUQ_HZ)
            e_end = e_start + int(target_length * MUQ_HZ)
            embeddings.append(sample['embeddings'][:, :, e_start:e_end, :])

            # segments/stems: STEPS_PER_SEC (1 Hz)
            s_start = int(start_sec * STEPS_PER_SEC)
            s_end = s_start + int(target_length * STEPS_PER_SEC)
            
            # segments_type is [1, L, C]
            segments_type.append(sample['segments_type'][:, s_start:s_end, :])
            # splits are [1, L]
            segments_splits.append(sample['segments_splits'][:, s_start:s_end])
            
            # long versions: STEPS_PER_SEC * BEATS_HZ (100 Hz)
            l_start = int(start_sec * STEPS_PER_SEC * BEATS_HZ)
            l_end = l_start + int(target_length * STEPS_PER_SEC * BEATS_HZ)
            
            segments_splits_long.append(sample['segments_splits_long'][:, l_start:l_end])
            
            # beats, on_set: BEATS_HZ (100 Hz)
            b_start = int(start_sec * BEATS_HZ)
            b_end = b_start + int(target_length * BEATS_HZ)
            beats.append(sample['beats'][:, b_start:b_end])
            on_set.append(sample['on_set'][:, b_start:b_end])

            # stem info: [1, 4, L]
            stem_splits.append(sample['stem_splits'][:, :, s_start:s_end])
            stem_splits_long.append(sample['stem_splits_long'][:, :, l_start:l_end])
            stem_silences.append(sample['stem_silences'][:, :, s_start:s_end])

            path.append(sample['path'])

            # Adjust 'segments' (list of tuples) for the new start and length
            # Each tuple is (label, start_time, end_time) relative to the 300s segment
            sample_segments = sample['segments']
            adjusted_segments = []
            for label, s_start, s_end in sample_segments:
                # Calculate new relative times
                new_start = max(s_start - start_sec, 0)
                new_end = min(s_end - start_sec, target_length)
                
                # Only include if the segment overlaps with the new range
                if new_start < new_end:
                    adjusted_segments.append((label, new_start, new_end))
            segments.append(adjusted_segments)

        segments_info = {
            "segments": segments,
            "segments_type": torch.concatenate(segments_type),
            "segments_splits": torch.concatenate(segments_splits),
            "segments_splits_long": torch.concatenate(segments_splits_long),
        }

        elements_info = {
            "stem_splits": torch.cat(stem_splits, dim=0),
            "stem_silences": torch.cat(stem_silences, dim=0),
            "stem_splits_long": torch.cat(stem_splits_long, dim=0),
        }

        beats_info = {
            "beats": torch.cat(beats, dim=0),
            "on_set": torch.cat(on_set, dim=0),
        }

        return torch.concatenate(embeddings), segments_info, elements_info, beats_info, path


def collate_embeddings_fn_with_preprocess(batch, length_pattern=None):
    # This is kept for backward compatibility if needed, but we prefer the class
    # Actually, the task says use partial function, so we might just use the class or a function that holds state.
    # If we use a class instance as collate_fn, it works fine with DataLoader.
    pass


def find_robust_onsets(elements, length, tolerance=0.1):
    candidates = elements[elements > 0]

    best_onsets = np.array([])
    max_matches = -1

    for start_candidate in candidates:
        end_point = elements[-1]
        current_grid = np.arange(start_candidate, end_point + tolerance, length)

        matches = 0
        for point in current_grid:
            if np.any(np.isclose(candidates, point, atol=tolerance)):
                matches += 1

        if matches > max_matches:
            max_matches = matches
            best_onsets = current_grid

    return best_onsets

#
# class PhaserDataset(Dataset):
#
#     def __init__(self, files, config=None, start=-1, transforms=None):
#         super(PhaserDataset, self).__init__()
#
#         assert all('parsed' in x and x.endswith('.json') for x in files), "Use parsed json files"
#
#         self.files = files
#         self.config = config
#         self.segment_size = int(config.segment_size)
#         self.start = start
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         json_file_path = self.files[idx]
#
#         with open(json_file_path, "r") as f:
#             json_dict_raw = json.load(f)
#
#         json_dict_raw['segments'] = json_dict_raw['segmnets']
#         # related_audios = glob.glob(json_file_path.replace('.json', '*.mp3'))
#         #
#         # audio_file_path = np.random.choice(related_audios)
#
#         audios = dict()
#         for stem_name in ['vocals', 'other', 'bass', 'drums']:
#             audio_path = json_file_path.replace('parsed', 'separated').replace('.json', f'_{stem_name}.wav')
#
#             try:
#                 y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
#                 audios[stem_name] = y
#
#             except:
#                 pass
#
#         sr = self.config.sample_rate
#         duration = librosa.get_duration(y=audios['other'], sr=sr)
#
#     #    duration = librosa.get_duration(y=y, sr=sr)
#         possible_start = int(duration - self.segment_size)
#
#         if self.start == -1:
#             start_sec = np.random.randint(low=0, high=possible_start+1)
#         else:
#             start_sec = self.start
#
#         end_sec = start_sec + self.segment_size
#
#         start = librosa.time_to_samples(start_sec, sr=sr)
#         end = librosa.time_to_samples(end_sec, sr=sr)
#
#         bar_length = 240/json_dict_raw['bpm']
#         beat_length = 60/json_dict_raw['bpm']
#
#         # get only segments from start to end
#         supported_segments = [(row['name'], row['start']*bar_length, row['end']*bar_length) for row in json_dict_raw['segmnets'] if (row['start']*bar_length <= start_sec <= row['end']*bar_length) or (start_sec <= row['start']*bar_length <= end_sec)]
#
#         # move segments to 0
#         moved_segments = [(label, max(s_start - start_sec, 0), min(s_end - start_sec, self.segment_size)) for label, s_start, s_end in supported_segments]
#         moved_segments = [x for x in moved_segments if x[1] != x[2]]
#
#         segments_split = torch.Tensor([int(s_end * STEPS_PER_SEC + 0.5) for label, s_start, s_end in moved_segments][:-1])
#
#         if len(segments_split) and segments_split[-1] == int(self.segment_size * STEPS_PER_SEC):
#             segments_split[-1] -= 1
#
#         segments_split_frames_encoded = F.one_hot(segments_split.long(), int(self.segment_size * STEPS_PER_SEC)).sum(0)
#
#         segments_encoded = []
#
#         for label, s_start, s_end in moved_segments:
#             start_segment, end_segment = int(s_start * STEPS_PER_SEC + 0.5), int(s_end * STEPS_PER_SEC + 0.5)
#             segment_encoding = F.one_hot(torch.Tensor([get_index(label)]).long(), len(SECTION_LABELS))
#             segment_encoding = segment_encoding.repeat(end_segment - start_segment, 1)
#
#             if len(segment_encoding) != 0:
#                 segments_encoded.append(segment_encoding)
#
#         segments_encoded_r = torch.concatenate(segments_encoded, dim=0).float()
#         end_segment = segments_encoded_r.shape[0]
#
#         try:
#             if end_segment != self.segment_size * STEPS_PER_SEC:
#                 missing_part_encoded = self.segment_size * STEPS_PER_SEC - end_segment
#                 segments_encoded.append(segments_encoded[-1][-1].repeat(missing_part_encoded, 1))
#
#             segments_encoded_r = torch.concatenate(segments_encoded, dim=0).float()
#
#         except Exception as e:
#             pass
#
#         elements_end = np.array([x['end'] for x in reduce(lambda x,y: x+y, [x for x in json_dict_raw['stem_elements'].values()])])
#         on_sets_bars = find_robust_onsets(elements_end, length=1, tolerance=0.1)
#         correct_bar_offset = on_sets_bars[0] % 1
#         correct_beat_offset = correct_bar_offset % 0.25
#
#         # get only segments from start to end
#
#         on_sets = [(x+correct_bar_offset)*bar_length for x in range(int(duration//bar_length+1))]
#         beats = [(x+correct_beat_offset)*beat_length for x in range(int(duration//beat_length+1))]
#
#         moved_on_sets = [(start - start_sec) for start in on_sets if start_sec <= start <= end_sec]
#         moved_beats = [(start - start_sec) for start in beats if start_sec <= start <= end_sec]
#
#         on_sets_split = torch.Tensor([int(start * MUQ_HZ + 0.5) for start in moved_on_sets][:-1])
#         beats_split = torch.Tensor([int(start * MUQ_HZ + 0.5) for start in moved_beats][:-1])
#
#         on_set_encoding = torch.zeros(MUQ_HZ*self.segment_size)
#         beats_encoding = torch.zeros(MUQ_HZ*self.segment_size)
#
#         for start_onset in on_sets_split:
#             if start_onset >= MUQ_HZ*self.segment_size:
#                 continue
#             on_set_encoding[int(start_onset)] = 1
#
#         for start_beat in beats_split:
#             if start_beat >= MUQ_HZ*self.segment_size:
#                 continue
#             beats_encoding[int(start_beat)] = 1
#
#         stems_splits = dict()
#         stems_silent = dict()
#
#         for stem_name, stem_elements in json_dict_raw['stem_elements'].items():
#
#             supported_elements = [(row['title'], row['start']*bar_length, row['end']*bar_length) for row in stem_elements if (row['start']*bar_length <= start_sec <= row['end']*bar_length) or (start_sec <= row['start']*bar_length <= end_sec)]
#             moved_elements = [(title, max(s_start - start_sec, 0), min(s_end - start_sec,self.segment_size*STEPS_PER_SEC)) for title, s_start, s_end in supported_elements]
#
#             stem_silent = torch.ones(STEPS_PER_SEC*self.segment_size+1)
#             stem_splits = torch.zeros(STEPS_PER_SEC*self.segment_size+1)
#
#             for element_title, start_element, end_element in moved_elements:
#                 stem_silent[int(start_element*STEPS_PER_SEC):int(end_element*STEPS_PER_SEC)+1] = 0
#                 stem_splits[int(end_element*STEPS_PER_SEC)] = 1
#                 stem_splits[int(start_element*STEPS_PER_SEC)] = 1
#
#             stems_splits[stem_name] = stem_splits.unsqueeze(0)
#             stems_silent[stem_name] = stem_silent.unsqueeze(0)
#
#         for stem_name, audio in audios.items():
#             audio = audio[start:end]
#
#             if self.transforms is not None:
#                 audio = self.transforms(audio)
#
#             audios[stem_name] = audio
#
#         sample = dict()
#
#         sample["waves"] = torch.concatenate([torch.Tensor(x).unsqueeze(0) for x in audios.values()])
#         sample["segments_encoded"] = segments_encoded_r.unsqueeze(0)
#         sample["segments_split_frames_encoded"] = segments_split_frames_encoded.unsqueeze(0)
#         sample["segments"] = moved_segments
#         sample["path"] = json_file_path
#         sample["on_set_encoding"] = on_set_encoding.unsqueeze(0)
#         sample["beats_encoding"] = beats_encoding.unsqueeze(0)
#
#         correct_frames = sample["segments_split_frames_encoded"].shape[-1]
#
#         sample['stem_splits'] = torch.concatenate([stems_splits['vocal'], stems_splits['other'], stems_splits['bass'], stems_splits['drums']])[:, :correct_frames]
#         sample['stem_silences'] = torch.concatenate([stems_silent['vocal'], stems_silent['other'], stems_silent['bass'], stems_silent['drums']])[:, :correct_frames]
#
#         for stem_name, stem_split  in stems_splits.items():
#             sample[f"split_{stem_name}"] = stem_split
#
#         for stem_name, stem_split  in stems_silent.items():
#             sample[f"silent_{stem_name}"] = stem_split
#
#         return sample


class PhaserEncodedDataset(Dataset):

    def __init__(self, files, config=None, start=-1, transforms: bool = False):
        super(PhaserEncodedDataset, self).__init__()

        assert all('parsed' in x and x.endswith('.json') for x in files), "Use parsed json files"

        self.files = files
        self.config = config
        self.segment_size = int(config.segment_size)
        self.start = start
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        json_file_path = self.files[idx]

        with open(json_file_path, "r") as f:
            json_dict_raw = json.load(f)

        json_dict_raw['segments'] = json_dict_raw['segmnets']
        # related_audios = glob.glob(json_file_path.replace('.json', '*.mp3'))
        #
        # audio_file_path = np.random.choice(related_audios)

        embeddings = dict()
        for stem_name in ['vocals', 'other', 'bass', 'drums']:
            emb_path = json_file_path.replace('parsed', 'encoded_short').replace('.json', f'_{stem_name}.npy')

            # if self.transforms:
            #     max_random = 3
            #     if stem_name == 'vocals':
            #         max_random = 1
            #
            # else:
            #     max_random = 1
            #
            # random_stem = np.random.randint(low=0, high=max_random)
            # emb_path = emb_path.replace(f'_{stem_name}.npy', f'_{stem_name}_{random_stem}.npy')

            try:
                embeddings[stem_name] = np.load(emb_path)

            except Exception as e:
                pass

        duration = embeddings['other'].shape[1] / MUQ_HZ

    #    duration = librosa.get_duration(y=y, sr=sr)
        possible_start = int(duration - self.segment_size)

        if self.start == -1:
            start_sec = np.random.randint(low=0, high=possible_start+1)
        else:
            start_sec = self.start

        end_sec = start_sec + self.segment_size

        start = librosa.time_to_samples(start_sec, sr=MUQ_HZ)
        end = librosa.time_to_samples(end_sec, sr=MUQ_HZ)

        bar_length = 240/json_dict_raw['bpm']
        beat_length = 60/json_dict_raw['bpm']

        supported_segments = [(row['name'], row['start']*bar_length, row['end']*bar_length) for row in json_dict_raw['segmnets'] if (row['start']*bar_length <= start_sec <= row['end']*bar_length) or (start_sec <= row['start']*bar_length <= end_sec)]

        # move segments to 0
        moved_segments = [(label, max(s_start - start_sec, 0), min(s_end - start_sec, self.segment_size)) for label, s_start, s_end in supported_segments]
        moved_segments = [x for x in moved_segments if x[1] != x[2]]


        segments_split = torch.Tensor([int(s_end * STEPS_PER_SEC + 0.5) for label, s_start, s_end in moved_segments][:-1])
        # Poprawka: użycie BEATS_HZ do long splits, zaokrąglenie +0.5
        segments_split_long = torch.Tensor([int(s_end * STEPS_PER_SEC * BEATS_HZ + 0.5) for label, s_start, s_end in moved_segments][:-1])

        # Poprawka: sprawdzenie czy indeks nie wykracza poza zakres (zaokrąglenia)
        if len(segments_split) and segments_split[-1] >= int(self.segment_size * STEPS_PER_SEC):
            segments_split[-1] = int(self.segment_size * STEPS_PER_SEC) - 1

        if len(segments_split_long) and segments_split_long[-1] >= int(self.segment_size * STEPS_PER_SEC * BEATS_HZ):
            segments_split_long[-1] = int(self.segment_size * STEPS_PER_SEC * BEATS_HZ) - 1

        segments_splits_ecoded = F.one_hot(segments_split.long(), int(self.segment_size * STEPS_PER_SEC)).sum(0)
        segments_splits_long_ecoded = F.one_hot(segments_split_long.long(), int(self.segment_size * STEPS_PER_SEC* BEATS_HZ)).sum(0)

        segments_encoded = []

        for label, s_start, s_end in moved_segments:
            start_segment, end_segment = int(s_start * STEPS_PER_SEC + 0.5), int(s_end * STEPS_PER_SEC + 0.5)
            segment_encoding = F.one_hot(torch.Tensor([get_index(label)]).long(), len(SECTION_LABELS))
            segment_encoding = segment_encoding.repeat(end_segment - start_segment, 1)

            if len(segment_encoding) != 0:
                segments_encoded.append(segment_encoding)

        segments_types_encoded = torch.concatenate(segments_encoded, dim=0).float()
        end_segment = segments_types_encoded.shape[0]

        try:
            if end_segment != self.segment_size * STEPS_PER_SEC:
                missing_part_encoded = self.segment_size * STEPS_PER_SEC - end_segment
                segments_encoded.append(segments_encoded[-1][-1].repeat(missing_part_encoded, 1))

            segments_types_encoded = torch.concatenate(segments_encoded, dim=0).float()

        except Exception as e:
            pass

        beats_encoding, on_set_encoding = self.get_beats_onsets(bar_length, beat_length, duration, end_sec, json_dict_raw, start_sec)
        stems_silent, stems_splits = self.get_stems_splits_silents(bar_length, end_sec, json_dict_raw, start_sec)
        stems_splits_long = self.get_stems_splits_long(bar_length, end_sec, json_dict_raw, start_sec)

        for stem_name, audio in embeddings.items():
            audio = audio[:, start:end, :]
            #
            # if self.transforms is not None:
            #     audio = self.transforms(audio)

            embeddings[stem_name] = audio

        sample = dict()


        sample["embeddings"] = torch.concatenate([torch.Tensor(x) for x in embeddings.values()]).unsqueeze(0)

        sample["segments"] = moved_segments
        sample["segments_type"] = segments_types_encoded.unsqueeze(0)
        sample["segments_splits"] = segments_splits_ecoded.unsqueeze(0)
        sample["segments_splits_long"] = segments_splits_long_ecoded.unsqueeze(0)

        sample["beats"] = beats_encoding.unsqueeze(0)
        sample["on_set"] = on_set_encoding.unsqueeze(0)

        correct_frames = sample["segments_splits"].shape[-1]
        correct_long_frames = sample["segments_splits_long"].shape[-1]

        sample['stem_splits'] = torch.concatenate([stems_splits['vocal'], stems_splits['other'], stems_splits['bass'], stems_splits['drums']])[:, :correct_frames].unsqueeze(0)
        sample['stem_splits_long'] = torch.concatenate([stems_splits_long['vocal'], stems_splits_long['other'], stems_splits_long['bass'], stems_splits_long['drums']])[:, :correct_long_frames].unsqueeze(0)
        sample['stem_silences'] = torch.concatenate([stems_silent['vocal'], stems_silent['other'], stems_silent['bass'], stems_silent['drums']])[:, :correct_frames].unsqueeze(0)

        # for stem_name, stem_split  in stems_splits.items():
        #     sample[f"split_{stem_name}"] = stem_split
        #
        # for stem_name, stem_split  in stems_silent.items():
        #     sample[f"silent_{stem_name}"] = stem_split

        sample["path"] = json_file_path

        return sample

    def get_beats_onsets(self, bar_length, beat_length, duration, end_sec, json_dict_raw, start_sec):
        elements_end = np.array([x['end'] for x in reduce(lambda x, y: x + y, [x for x in json_dict_raw['stem_elements'].values()])])
        on_sets_bars = find_robust_onsets(elements_end, length=1, tolerance=0.1)
        correct_bar_offset = on_sets_bars[0] % 1
        correct_beat_offset = correct_bar_offset % 0.25

        # get only segments from start to end

        on_sets = [(x + correct_bar_offset) * bar_length for x in range(int(duration // bar_length + 1))]
        beats = [(x + correct_beat_offset) * beat_length for x in range(int(duration // beat_length + 1))]

        moved_on_sets = [(start - start_sec) for start in on_sets if start_sec <= start <= end_sec]
        moved_beats = [(start - start_sec) for start in beats if start_sec <= start <= end_sec]

        on_sets_split = torch.Tensor([int(start * BEATS_HZ + 0.5) for start in moved_on_sets][:-1])
        beats_split = torch.Tensor([int(start * BEATS_HZ + 0.5) for start in moved_beats][:-1])

        on_set_encoding = torch.zeros(BEATS_HZ * self.segment_size)
        beats_encoding = torch.zeros(BEATS_HZ * self.segment_size)

        for start_onset in on_sets_split:
            if start_onset >= BEATS_HZ * self.segment_size:
                continue
            on_set_encoding[int(start_onset)] = 1

        for start_beat in beats_split:
            if start_beat >= BEATS_HZ * self.segment_size:
                continue
            beats_encoding[int(start_beat)] = 1
        return beats_encoding, on_set_encoding

    def get_stems_splits_silents(self, bar_length, end_sec, json_dict_raw, start_sec):
        stems_splits = dict()
        stems_silent = dict()
    
        for stem_name, stem_elements in json_dict_raw['stem_elements'].items():
    
            supported_elements = [(row['title'], row['start'] * bar_length, row['end'] * bar_length) for row in stem_elements if
                                  (row['start'] * bar_length <= start_sec <= row['end'] * bar_length) or (start_sec <= row['start'] * bar_length <= end_sec)]
            moved_elements = [(title, max(s_start - start_sec, 0), min(s_end - start_sec, self.segment_size * STEPS_PER_SEC)) for title, s_start, s_end in supported_elements]
    
            stem_silent = torch.ones(STEPS_PER_SEC * self.segment_size + 1)
            stem_splits = torch.zeros(STEPS_PER_SEC * self.segment_size + 1)
    
            for element_title, start_element, end_element in moved_elements:
                stem_silent[int(start_element * STEPS_PER_SEC):int(end_element * STEPS_PER_SEC) + 1] = 0
                stem_splits[int(end_element * STEPS_PER_SEC)] = 1
                stem_splits[int(start_element * STEPS_PER_SEC)] = 1
    
            stems_splits[stem_name] = stem_splits.unsqueeze(0)
            stems_silent[stem_name] = stem_silent.unsqueeze(0)
        return stems_silent, stems_splits


    def get_stems_splits_long(self, bar_length, end_sec, json_dict_raw, start_sec):
        stems_splits_long = dict()
        for stem_name, stem_elements in json_dict_raw['stem_elements'].items():

            supported_elements = [(row['title'], row['start'] * bar_length, row['end'] * bar_length) for row in stem_elements if
                                  (row['start'] * bar_length <= start_sec <= row['end'] * bar_length) or (start_sec <= row['start'] * bar_length <= end_sec)]
            moved_elements = [(title, max(s_start - start_sec, 0), min(s_end - start_sec, BEATS_HZ * self.segment_size * STEPS_PER_SEC)) for title, s_start, s_end in supported_elements]


            stem_splits = torch.zeros(BEATS_HZ * STEPS_PER_SEC * self.segment_size + 1)

            for element_title, start_element, end_element in moved_elements:
                end_segment = int(end_element * STEPS_PER_SEC * BEATS_HZ)

                if BEATS_HZ * STEPS_PER_SEC * self.segment_size >= end_segment:
                    stem_splits[int(end_element * STEPS_PER_SEC * BEATS_HZ)] = 1

                stem_splits[int(start_element * STEPS_PER_SEC * BEATS_HZ)] = 1

            stems_splits_long[stem_name] = stem_splits.unsqueeze(0)
        return stems_splits_long


def get_loaders(config):

    train_files = []
    test_files = []

    import os as _os
    available_datasets = glob.glob(_os.environ.get("PHRASER_DATA_GLOB", _os.path.join(CORPUS, "*", "splits.json")))

    for dataset_path in available_datasets:
        if is_excluded(dataset_path):
            continue

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        train_files += [dataset_path.replace('splits.json', f"parsed/{x}.json") for x in dataset['train']]
        test_files += [dataset_path.replace('splits.json', f"parsed/{x}.json") for x in dataset['test']]

    train_dataset = PhaserEncodedDataset(files=train_files, config=config, start=-1, transforms=True)
    test_dataset = PhaserEncodedDataset(files=test_files, config=config, start=0)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                  collate_fn=CollateEmbeddingsWithPreprocess(length_pattern=[300, -1, -1, -1, -1, 60, -1, 30]),
                                  num_workers=8, prefetch_factor=2, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 collate_fn=CollateEmbeddingsWithPreprocess(length_pattern=[300, 120, 60, 30]),
                                 num_workers=8, prefetch_factor=2, drop_last=True)

    return train_dataloader, test_dataloader

#
