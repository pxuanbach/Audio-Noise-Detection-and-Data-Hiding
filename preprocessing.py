import os
import numpy as np
from tqdm import tqdm

from utils.audio_to_stft import audio_to_stft
import json

# Project and data paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(CURRENT_DIR, "datasets", "processed")
MUSAN_ROOT = "D:\Backup\musan"  # Keep this absolute since it's external data source

# Parameters for STFT (360, 360)
SAMPLE_RATE = 22000  # Standard audio rate
N_FFT = 719         # Will give ~360 frequency bins
MAX_DURATION = 10.0  # Duration in seconds
TRAIN_RATIO = 0.8
TARGET_FRAMES = 360
TARGET_FREQ_BINS = 360
TRAIN_MAX_SEGMENTS = 5
TEST_MAX_SEGMENTS = 5

# Data augmentation parameters
AUGMENTATION = {
    'time_stretch': [0.9, 1.1],  # Range for time stretching
    'pitch_shift': [-2, 2],      # Semitones for pitch shifting
    'noise_level': [0.001, 0.002]  # Background noise level range
}

def preprocess_musan_dataset_stft(root_dir, output_dir=PROCESSED_DIR):
    """Process MUSAN dataset with augmentation"""
    # Create main directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create train and test subdirectories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    for split_dir in [train_dir, test_dir]:
        for category in ["speech", "music"]:
            os.makedirs(os.path.join(split_dir, category), exist_ok=True)

    # Process each category
    for category in ["speech", "music"]:
        category_path = os.path.join(root_dir, category)
        if not os.path.exists(category_path):
            print(f"Directory {category_path} does not exist!")
            continue

        # Get all wav files
        wav_files = []
        for root, _, files in os.walk(category_path):
            wav_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])

        # Shuffle files
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(wav_files)

        # Split into train and test
        n_train = int(len(wav_files) * TRAIN_RATIO)
        train_files = wav_files[:n_train]
        test_files = wav_files[n_train:]

        # Process train files with augmentation
        with tqdm(total=len(train_files), desc=f"Processing {category} train files") as pbar:
            for file_path in train_files:
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                # Process audio
                stft_segments, sr, n_fft, hop_length = audio_to_stft(
                    file_path,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    segment_len_sec=MAX_DURATION,
                    target_frames=TARGET_FRAMES,
                    target_freq_bins=TARGET_FREQ_BINS,
                    max_segments=TRAIN_MAX_SEGMENTS
                )

                # Save each segment
                for i, stft_spec in enumerate(stft_segments):
                    output_file = os.path.join(train_dir, category, f"{file_name}_seg{i}.npy")
                    np.save(output_file, stft_spec)
                pbar.update(1)

            # Save STFT parameters to JSON file
            params = {
                'sample_rate': sr,
                'n_fft': n_fft,
                'max_duration': MAX_DURATION,
                'target_frames': TARGET_FRAMES,
                'target_freq_bins': TARGET_FREQ_BINS,
                'max_segments': TRAIN_MAX_SEGMENTS,
                'hop_length': hop_length,
            }

            # Save parameters
            params_file = os.path.join(train_dir, 'stft_params.json')
            if not os.path.exists(params_file):
                with open(params_file, 'w') as f:
                    json.dump(params, f, indent=4)


        # Process test files
        with tqdm(total=len(test_files), desc=f"Processing {category} test files") as pbar:
            for file_path in test_files:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                stft_segments, sr, n_fft, hop_length = audio_to_stft(
                    file_path,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    segment_len_sec=MAX_DURATION,
                    target_frames=TARGET_FRAMES,
                    target_freq_bins=TARGET_FREQ_BINS,
                    max_segments=TEST_MAX_SEGMENTS
                )
                # Save first segment only for test files
                if len(stft_segments) > 0:
                    output_file = os.path.join(test_dir, category, f"{file_name}.npy")
                    np.save(output_file, stft_segments[0])
                pbar.update(1)

            # Save STFT parameters to JSON file
            params = {
                'sample_rate': sr,
                'n_fft': n_fft,
                'max_duration': MAX_DURATION,
                'target_frames': TARGET_FRAMES,
                'target_freq_bins': TARGET_FREQ_BINS,
                'max_segments': TEST_MAX_SEGMENTS,
                'hop_length': hop_length,
            }

            # Save parameters
            params_file = os.path.join(test_dir, 'stft_params.json')
            if not os.path.exists(params_file):
                with open(params_file, 'w') as f:
                    json.dump(params, f, indent=4)


    print(f"\nData split completed:")
    print(f"Train data saved to: {train_dir}")
    print(f"Test data saved to: {test_dir}")

if __name__ == "__main__":
    # Process and split dataset
    preprocess_musan_dataset_stft(MUSAN_ROOT)
