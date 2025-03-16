import os
import random
import numpy as np
import soundfile as sf
import IPython.display as ipd
from dataset import MUSANDataset
from utils import mel_to_audio, visualize_spectrograms

def save_and_play_audio(waveform, sr=16000, filename="output.wav"):
    sf.write(filename, waveform, sr)
    print(f"Saved audio to {filename}")
    return ipd.Audio(filename, rate=sr)

# Load dataset
musan_dir = "D:\Backup\musan"
dataset = MUSANDataset(musan_dir, max_duration=5, noise_ratio=0.3)

# Get one sample with info
idx = random.randint(0, len(dataset) - 1)
speech_path, noise_path = dataset.get_random_path(idx)
data = dataset.process_noise_mask_mel(speech_path, noise_path)
noisy_segments = data["noisy_segments"]
noisy_mel = data["noisy_mel"]
segment_length = data["segment_length"]
mask_mel = data["mask_mel"]
segment_times = dataset.get_segment_times(noisy_segments, segment_length)

# Print noise information
print("\nAudio mixing info:")
print(f"Speech file: {os.path.basename(speech_path)}")
print(f"Noise file: {os.path.basename(noise_path)}")
print(f"Noisy segments (idx): {noisy_segments}")
print("\nNoise time ranges:")
for start, end in segment_times:
    print(f"  {start:.1f}s - {end:.1f}s")

# Convert to audio and save
audio = mel_to_audio(noisy_mel, hop_length=dataset.hop_length)
print(f"\nAudio length: {len(audio)/16000:.2f}s")
save_and_play_audio(audio)

# Save spectrograms
spec_filename = f"spectrograms_{os.path.basename(speech_path).split('.')[0]}.png"
visualize_spectrograms(noisy_mel, mask_mel, hop_length=dataset.hop_length, output_path=spec_filename)
