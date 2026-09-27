import copy
import glob
import os
import torch
import librosa
import numpy as np
from muq import MuQ
import json

from phraser.modules.rope import Wav2Vec2ConformerRotaryPositionalEmbeddingScale
from phraser.paths import CORPUS, is_excluded

# Initialize MuQ model
if torch.cuda.device_count():
    device = torch.device('cuda')
else:
    print("WARNING, using CPU")
    device = torch.device('cpu')

muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
#
# config = copy.deepcopy(muq.model.conformer.config)
# config['rope_scaling_factor'] = 10 #scale from 30 s to 5min
# muq.model.conformer.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbeddingScale(config)

muq = muq.to(device).eval()

DATASET_PATH = CORPUS
MUQ_SR = 24000  # MuQ's required sample rate
SEGMENT_LENGTH = 30  # seconds

def process_audio_segment(wav_segment):
    """Process a single audio segment and return its embedding"""
    try:
        # Convert to tensor and add batch dimension
        wavs = torch.tensor(wav_segment).unsqueeze(0).to(device)
        
        # Calculate embedding
        with torch.no_grad():
            output = muq(wavs, output_hidden_states=True)
            
        # Get the embedding
        embedding = output.last_hidden_state.cpu().numpy()
        return embedding
        
    except Exception as e:
        print(f"Error processing segment: {e}")
        return None

def process_audio_file(audio_path):
    """Process audio file by splitting it into segments and combining embeddings"""
    try:
        # Load and resample audio
        wav, sr = librosa.load(audio_path, sr=MUQ_SR)
        
        # Calculate segment length in samples
        segment_samples = SEGMENT_LENGTH * MUQ_SR
        
        # Split audio into segments
        segments = []
        for start in range(0, len(wav), segment_samples):
            end = start + segment_samples
            segment = wav[start:end]
            
            # # Pad last segment if needed
            # if len(segment) < segment_samples:
            #     segment = np.pad(segment, (0, segment_samples - len(segment)))
                
            segments.append(segment)
        
        # Process each segment
        embeddings = []
        for segment in segments:
            embedding = process_audio_segment(segment)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            return None
            
        # Concatenate all embeddings along the time dimension
        combined_embedding = np.concatenate(embeddings, axis=1)
        return combined_embedding
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    # Iterate through all datasets
    for folder_path in glob.glob(f"{DATASET_PATH}*"):
        separated_path = os.path.join(folder_path, "separated")
        encoded_path = os.path.join(folder_path, "encoded_short")

        if is_excluded(folder_path):
            continue
         # Create encoded directory if it doesn't exist
        os.makedirs(encoded_path, exist_ok=True)
        
        # Process all separated files
        for audio_file in glob.glob(f"{separated_path}/*.wav"):
            try:
                # Get the base filename and instrument type
                basename = os.path.basename(audio_file)
                cue_id, instrument = basename.replace(".wav", "").split("_")
                
                # Define output path for the embedding
                output_path = os.path.join(encoded_path, f"{cue_id}_{instrument}.npy")
                
                # Skip if already processed
                try:

                    if os.path.exists(output_path) and np.load(output_path).shape[0] > 0:
                        print(f"Skipping {basename} - already encoded")
                        continue
                except:
                    pass

                
                # Process the audio file
                embedding = process_audio_file(audio_file)
                
                if embedding is not None:
                    # Save the embedding
                    np.save(output_path, embedding)
                    print(f"Processed {basename}")
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"ERROR processing {audio_file}: {e}")

if __name__ == "__main__":
    main()