import torch
import torchaudio
import librosa
import numpy as np
import os
import random

class MUSANDataset(torch.utils.data.Dataset):
    def __init__(self, musan_dir, sr=16000, n_mels=128, max_duration=3, noise_ratio=0.5):
        """
        self.noise_ratio = noise_ratio
        MUSAN dataset loader with automatic noise/music mixing.

        Args:
            musan_dir (str): Path to MUSAN dataset directory.
            sr (int): Sample rate of audio files.
            n_mels (int): Number of Mel frequency bands.
            max_duration (int): Maximum duration of each sample (in seconds).
            noise_ratio (float): Ratio of segments to add noise to.
        """
        self.sr = sr
        self.n_mels = n_mels
        self.max_samples = sr * max_duration  # Maximum number of samples per file
        self.noise_ratio = noise_ratio
        self.n_fft = 1024
        self.hop_length = 256

        # Load file lists
        self.speech_files = self.get_all_audio_files(os.path.join(musan_dir, "speech"))
        self.noise_files = self.get_all_audio_files(os.path.join(musan_dir, "noise"))
        self.music_files = self.get_all_audio_files(os.path.join(musan_dir, "music"))

    def __len__(self):
        return len(self.speech_files)

    def get_all_audio_files(self, root_dir):
        """ Recursively collect all audio files from subdirectories. """
        audio_files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(subdir, file))
        return sorted(audio_files)

    def get_speech_files(self):
        return self.speech_files

    def get_music_files(self):
        return self.music_files

    def get_noise_files(self):
        return self.noise_files

    def __getitem__(self, idx):
        # Set random seed based on idx for reproducibility
        np.random.seed(idx)
        random.seed(idx)

        # Load speech
        speech_path = random.choice(self.speech_files + self.music_files)
        speech, _ = librosa.load(speech_path, sr=self.sr)

        # Trim or pad speech to max_duration
        if len(speech) > self.max_samples:
            start_idx = random.randint(0, len(speech) - self.max_samples)
            speech = speech[start_idx:start_idx + self.max_samples]
        else:
            speech = np.pad(speech, (0, self.max_samples - len(speech)), mode='constant')

        # Select a noise/music file with fixed seed
        noise_path = random.choice(self.noise_files)
        noise, _ = librosa.load(noise_path, sr=self.sr)

        # Trim or repeat noise to match speech length
        if len(noise) > len(speech):
            noise = noise[:len(speech)]
        else:
            noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))[:len(speech)]

        # Create a binary mask for noise
        mask = np.zeros_like(speech)  # Initially all clean (0)

        # Divide speech into segments (e.g., 500ms = 8000 samples)
        segment_length = self.sr // 2  # 500ms
        num_segments = len(speech) // segment_length

        # Store which segments have noise
        noisy_segments = []

        # Randomly add noise to some segments
        for i in range(num_segments):
            if random.random() < self.noise_ratio:  # Only add noise to `noise_ratio` % of segments
                start = i * segment_length
                end = start + segment_length
                mask[start:end] = 1  # Mark as noise
                speech[start:end] += noise[start:end]  # Add noise to this segment
                noisy_segments.append(i)  # Store segment index
                print(f"Added noise to segment {i} start {start} end {end}")

        # Convert segment indices to time ranges
        segment_times = []
        for seg_idx in noisy_segments:
            start_time = seg_idx * (segment_length / self.sr)  # to seconds
            end_time = (seg_idx + 1) * (segment_length / self.sr)
            segment_times.append((start_time, end_time))

        # Convert to Mel-Spectrogram
        noisy_mel = librosa.feature.melspectrogram(
            y=speech,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mask = librosa.feature.melspectrogram(
            y=mask,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Convert to PyTorch tensors
        noisy_mel = torch.tensor(noisy_mel).unsqueeze(0)  # (1, F, T)
        mask = torch.tensor(mask).unsqueeze(0)  # (1, F, T)

        # Return additional info
        info = {
            'speech_file': os.path.basename(speech_path),
            'noise_file': os.path.basename(noise_path),
            'noisy_segments': noisy_segments,
            'segment_times': segment_times
        }

        return noisy_mel, mask, info
