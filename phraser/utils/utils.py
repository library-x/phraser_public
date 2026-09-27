import glob
import io
import json
from collections import Counter

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pydub import AudioSegment

STEPS_PER_SEC = 1

MUQ_HZ = 25
BEATS_HZ = 100

SECTION_LABELS = [
    'silence',
  'intro',
  'outro',
  'bridge',
  'verse',
  'chorus',
]

SECTION_COLORS = [
    'black',
    'purple',
    'red',
    'cyan',
    'green',
    'orange',
    'blue',
    'yellow',
]

MUQ_SR = 24000


def get_index(label):
    if label in SECTION_LABELS:
        return SECTION_LABELS.index(label)

    return SECTION_LABELS.index('verse')




def convert_to_audio_segment(data, samplerate):
    y_l = np.array(data[:, 0] * (1 << 15), dtype=np.int16)
    y_r = np.array(data[:, 1] * (1 << 15), dtype=np.int16)

    result = np.empty(len(y_l) * 2, dtype=np.int16)
    result[0::2] = y_l
    result[1::2] = y_r

    result = AudioSegment(result.tobytes(), frame_rate=samplerate, sample_width=result.dtype.itemsize, channels=2)
    return result


def convert_to_librosa(audiosegment):
    data = np.array(audiosegment.get_array_of_samples())
    y_l = data[0::2].astype(np.float32) / 32768
    y_r = data[1::2].astype(np.float32) / 32768

    return np.stack([y_l, y_r]).T, audiosegment.frame_rate



def collate_fn_with_preprocess(batch):
    """
    Collate function for wdsdataloader.
    batch: a list of dict, each dict is a sample
    """
    waveforms = []
    section_encodings = []
    section_type_encodings = []
    urls = []
    segments = []
    for sample in batch:
        waveforms.append(sample['waveform'])
        section_encodings.append(sample['segments_split_frames_encoded'])
        section_type_encodings.append(sample['segments_encoded'])
        urls.append(sample['path'])
        segments.append(sample['segments'])
    return torch.concatenate(waveforms), torch.concatenate(section_encodings), torch.concatenate(section_type_encodings), urls, segments

def get_index(label):
    if label in SECTION_LABELS:
        return SECTION_LABELS.index(label)

    return SECTION_LABELS.index('verse')


def print_params(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model size: {:.3f}MB'.format((param_size + buffer_size) / 1024 ** 2))
    print(f'model params: {params}')



def most_common_spacing(binary_array):
    binary_array = np.asarray(binary_array).astype(bool).astype(int).squeeze()

    ones_idx = np.where(binary_array == 1)[0]
    if len(ones_idx) < 2:
        return None, None, binary_array.copy()

    diffs = []
    for i in range(len(ones_idx)):
        for j in range(i + 1, len(ones_idx)):
            diffs.append(ones_idx[j] - ones_idx[i])

    counter = Counter(diffs)
    spacing, count = counter.most_common(1)[0]

    confidence = count / len(ones_idx)

    best_start = None
    best_matches = -1

    for start in ones_idx:
        grid = np.arange(start, len(binary_array), spacing)
        matches = np.sum(np.isin(grid, ones_idx))

        if matches > best_matches:
            best_matches = matches
            best_start = start

    result = np.zeros_like(binary_array)
    for i in range(best_start, len(binary_array), spacing):
        result[i] = 1

    return spacing, confidence, result


def robust_periodic_reconstruction(on_set_predict_one):
    try:
        data = np.where((on_set_predict_one > 0.5))[0]
        target_end = len(on_set_predict_one)

        # 1. Automatic period detection (Precise median)
        # We look at the differences between points; median is robust against gaps (missing data)
        diffs = np.diff(data)
        base_period = np.median(diffs)

        # Fine-tuning the period
        # We only consider intervals close to the median (+/- 20%)
        # This excludes large gaps or noise spikes, providing a precise fractional period
        refined_diffs = diffs[(diffs > base_period * 0.8) & (diffs < base_period * 1.2)]
        precise_period = np.mean(refined_diffs)

        # 2. Phase (Start Offset) Detection
        # We calculate the remainder of each point divided by the period.
        # We look for the shift that minimizes the error for the majority of the data.
        potential_offsets = data % precise_period

        # We use the median of offsets to find the "most popular" phase shift.
        # This prevents a noisy first element (data[0]) from shifting the entire grid.
        best_offset = np.median(potential_offsets)

        # 3. Determining the True Start
        # The first value is the nearest multiple of the period + offset to the start of our data.
        first_theoretical = np.round((data[0] - best_offset) / precise_period) * precise_period + best_offset

        # 4. Building the Grid
        end_val = target_end if target_end else data[-1]
        num_steps = int(np.round((end_val - first_theoretical) / precise_period)) + 1

        reconstructed_indices = []
        for i in range(num_steps):
            # Theoretical point on the ideal fractional grid
            expected_val = first_theoretical + i * precise_period

            # Tolerance window (half of the period)
            # Any point within this range is considered to belong to this "slot"
            tolerance = precise_period / 2
            matches = data[(data >= expected_val - tolerance) & (data <= expected_val + tolerance)]

            if len(matches) > 0:
                # If a match exists in the input, pick the one closest to the theoretical ideal.
                # This prioritizes original data (e.g., keeping 267 instead of rounding to 266).
                best_match = matches[np.argmin(np.abs(matches - expected_val))]
                reconstructed_indices.append(int(best_match))
            else:
                # If there is a gap in the data, fill it with the rounded theoretical value.
                reconstructed_indices.append(int(np.round(expected_val)))

        # Parse indexes into array of 0 and 1
        reconstructed_binary = np.zeros_like(on_set_predict_one)
        for idx in reconstructed_indices:
            if idx < len(reconstructed_binary):
                reconstructed_binary[idx] = 1

        return reconstructed_binary
        
    except Exception as e:
       # print(f"Error in robust_periodic_reconstruction: {str(e)}")
        return np.zeros_like(on_set_predict_one)