import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project and data paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")
MUSAN_ROOT = "D:\Backup\musan"  # Keep this absolute since it's external data source

# Parameters based on the report
SAMPLE_RATE = 22000
N_MELS = 360
N_FFT = 719
HOP_LENGTH = None
TARGET_FRAMES = 360 # Target size 360x360
MAX_DURATION = 10.0 # Maximum audio length limit
TRAIN_RATIO = 0.7   # Ratio for train/test split
REDUCE_SIZE = True  # Flag to control data size reduction

# Function to process STFT instead of mel-spectrogram
def audio_to_stft(file_path, sr=SAMPLE_RATE, n_fft=N_FFT, target_frames=TARGET_FRAMES, max_duration=MAX_DURATION):
    # Load full audio without duration limit
    audio, sr = librosa.load(file_path, sr=sr)

    # Calculate number of complete segments
    segment_samples = int(max_duration * sr)
    total_samples = len(audio)
    segments = []

    # Calculate maximum possible segments
    possible_segments = total_samples // segment_samples

    # Randomly select up to 5 segments
    n_segments = min(5, possible_segments)
    if n_segments > 0:
        # Get random starting points for segments
        segment_starts = np.random.choice(
            range(0, total_samples - segment_samples + 1),
            size=n_segments,
            replace=False
        )

        # Process selected segments
        for i in segment_starts:
            segment = audio[i:i + segment_samples]
            # Only keep segments that are full length (30s)
            if len(segment) == segment_samples:
                # Process STFT for this segment
                hop_length = len(segment) // (target_frames - 1)
                stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)

                # Convert to float32 if REDUCE_SIZE is True
                if REDUCE_SIZE:
                    real_part = np.real(stft).astype(np.float32)
                    imag_part = np.imag(stft).astype(np.float32)
                else:
                    real_part = np.real(stft)
                    imag_part = np.imag(stft)

                # Process padding/cropping
                target_size = (N_MELS, N_MELS)
                for part in [real_part, imag_part]:
                    if part.shape[1] > target_size[1]:
                        part = part[:, :target_size[1]]
                    elif part.shape[1] < target_size[1]:
                        pad_width = target_size[1] - part.shape[1]
                        part = np.pad(part, ((0, 0), (0, pad_width)), mode='constant')
                    if part.shape[0] > target_size[0]:
                        part = part[:target_size[0], :]
                    elif part.shape[0] < target_size[0]:
                        pad_width = target_size[0] - part.shape[0]
                        part = np.pad(part, ((0, pad_width), (0, 0)), mode='constant')

                # Stack without scaling factor
                stft_2ch = np.stack((real_part, imag_part), axis=0)
                segments.append(stft_2ch)

    return segments

# Use in preprocess_musan_dataset
def preprocess_musan_dataset_stft(root_dir, output_dir=PROCESSED_DIR):
    """Process MUSAN dataset and split into train/test sets"""
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

        # Process train files
        with tqdm(total=len(train_files), desc=f"Processing {category} train files") as pbar:
            for file_path in train_files:
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                # Get all segments for this file
                stft_segments = audio_to_stft(file_path)

                # Save each segment
                for i, stft_spec in enumerate(stft_segments):
                    output_file = os.path.join(train_dir, category, f"{file_name}_seg{i}.npy")
                    if REDUCE_SIZE:
                        np.save(output_file, stft_spec, allow_pickle=False, fix_imports=False)
                    else:
                        np.save(output_file, stft_spec)
                pbar.update(1)

        # Process test files
        with tqdm(total=len(test_files), desc=f"Processing {category} test files") as pbar:
            for file_path in test_files:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                stft_segments = audio_to_stft(file_path)
                # Save first segment only for test files
                if len(stft_segments) > 0:
                    output_file = os.path.join(test_dir, category, f"{file_name}.npy")
                    if REDUCE_SIZE:
                        np.save(output_file, stft_segments[0], allow_pickle=False, fix_imports=False)
                    else:
                        np.save(output_file, stft_segments[0])
                pbar.update(1)

    print(f"\nData split completed:")
    print(f"Train data saved to: {train_dir}")
    print(f"Test data saved to: {test_dir}")

if __name__ == "__main__":
    # Process and split dataset
    preprocess_musan_dataset_stft(MUSAN_ROOT)
