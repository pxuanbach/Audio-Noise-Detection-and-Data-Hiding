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
dataset = MUSANDataset(musan_dir)

# Get one sample with info
noisy_mel, mask, info = dataset[0]

# Print noise information
print("\nAudio mixing info:")
print(f"Speech file: {info['speech_file']}")
print(f"Noise file: {info['noise_file']}")
print(f"Noisy segments (idx): {info['noisy_segments']}")
print("\nNoise time ranges:")
for start, end in info['segment_times']:
    print(f"  {start:.1f}s - {end:.1f}s")

# Convert to audio and save
audio = mel_to_audio(noisy_mel)
print(f"\nAudio length: {len(audio)/16000:.2f}s")
save_and_play_audio(audio)

# Save spectrograms
spec_filename = f"spectrograms_{info['speech_file'].split('.')[0]}.png"
visualize_spectrograms(noisy_mel, mask, output_path=spec_filename)
