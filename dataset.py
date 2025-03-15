import torch
import torchaudio
import librosa
import numpy as np
import os
import random

class MUSANDataset(torch.utils.data.Dataset):
    def __init__(self, musan_dir, sr=16000, n_mels=128, max_duration=3, noise_ratio=0.5, max_frames=186):
        """
        MUSAN dataset loader with automatic noise/music mixing.

        Args:
            musan_dir (str): Path to MUSAN dataset directory.
            sr (int): Sample rate of audio files.
            n_mels (int): Number of Mel frequency bands.
            max_duration (int): Maximum duration of each sample (in seconds).
            noise_ratio (float): Ratio of segments to add noise to.
            max_frames (int): Số frame thời gian cố định cho Mel-spectrogram.
        """
        self.sr = sr
        self.n_mels = n_mels
        self.max_samples = sr * max_duration  # Maximum number of samples per file
        self.noise_ratio = noise_ratio
        self.n_fft = 1024
        self.hop_length = 256
        self.max_frames = max_frames  # Số frame cố định cho Mel-spectrogram

        # Load file lists
        self.speech_files = self.get_all_audio_files(os.path.join(musan_dir, "speech"))
        self.noise_files = self.get_all_audio_files(os.path.join(musan_dir, "noise"))
        self.music_files = self.get_all_audio_files(os.path.join(musan_dir, "music"))

    def __len__(self):
        return len(self.speech_files + self.music_files)

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

    def preprocess_speech_and_noise(self, speech_path: str, noise_path: str):
        # Load speech
        speech, _ = librosa.load(speech_path, sr=self.sr)

        # Trim or pad speech to max_duration
        if len(speech) > self.max_samples:
            start_idx = random.randint(0, len(speech) - self.max_samples)
            speech = speech[start_idx:start_idx + self.max_samples]
        else:
            speech = np.pad(speech, (0, self.max_samples - len(speech)), mode='constant')

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

        return speech, noise, mask, num_segments, segment_length

    def get_random_path(self, idx):
        np.random.seed(idx)
        random.seed(idx)

        speech_path = random.choice(self.speech_files + self.music_files)
        noise_path = random.choice(self.noise_files)
        return speech_path, noise_path

    def get_segment_times(self, noisy_segments, segment_length):
        segment_times = []
        for seg_idx in noisy_segments:
            start_time = seg_idx * (segment_length / self.sr)  # to seconds
            end_time = (seg_idx + 1) * (segment_length / self.sr)
            segment_times.append((start_time, end_time))

        return segment_times

    def process_noise_mask_mel(self, speech_path: str, noise_path: str):
        # Preprocess speech and noise
        (
            speech, noise, mask, num_segments, segment_length
        ) = self.preprocess_speech_and_noise(speech_path, noise_path)

        # Store which segments have noise
        noisy_segments = []

        # Randomly add noise to some segments
        for i in range(num_segments):
            if random.random() < self.noise_ratio:  # Only add noise to `noise_ratio` % of segments
                start = i * segment_length
                end = start + segment_length
                mask[start:end] = 1  # Mark as noise
                speech[start:end] += noise[start:end]  # Add noise to this segment
                noisy_segments.append(i)

        noisy_mel = librosa.feature.melspectrogram(
            y=speech,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mask_mel = librosa.feature.melspectrogram(
            y=mask,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        return {
            "speech": speech,
            "noise": noise,
            "num_segments": num_segments,
            "segment_length": segment_length,
            "noisy_segments": noisy_segments,
            "noisy_mel": noisy_mel,
            "mask_mel": mask_mel
        }

    def __getitem__(self, idx):
        # Set random seed based on idx for reproducibility
        speech_path, noise_path = self.get_random_path(idx)

        speech_path = random.choice(self.speech_files + self.music_files)
        noise_path = random.choice(self.noise_files)

        # Get noisy Mel-spectrogram and noisy segments
        pre_data = self.process_noise_mask_mel(speech_path, noise_path)
        noisy_mel = pre_data["noisy_mel"]
        mask_mel = pre_data["mask_mel"]
        noisy_segments = pre_data["noisy_segments"]
        segment_length = pre_data["segment_length"]

        noisy_mel = librosa.power_to_db(noisy_mel, ref=np.max)  # Convert to log scale
        mask_mel = librosa.power_to_db(mask_mel, ref=np.max)  # Convert to log scale

        # Normalize to [0, 1] range with safety checks
        noisy_mel = (noisy_mel - noisy_mel.min()) / (noisy_mel.max() - noisy_mel.min() + 1e-8)

        # For mask_mel, we want binary values (0 or 1)
        mask_mel = (mask_mel > mask_mel.mean()).astype(np.float32)

        # Cắt hoặc đệm Mel-spectrogram về max_frames
        if noisy_mel.shape[1] > self.max_frames:
            noisy_mel = noisy_mel[:, :self.max_frames]
            mask_mel = mask_mel[:, :self.max_frames]
        else:
            noisy_mel = np.pad(noisy_mel, ((0, 0), (0, self.max_frames - noisy_mel.shape[1])), mode='constant')
            mask_mel = np.pad(mask_mel, ((0, 0), (0, self.max_frames - mask_mel.shape[1])), mode='constant')

        # Convert to PyTorch tensors
        noisy_mel = torch.tensor(noisy_mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, max_frames)
        mask_mel = torch.tensor(mask_mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, max_frames)

        # Return additional info
        info = {
            'speech_file': os.path.basename(speech_path),
            'noise_file': os.path.basename(noise_path),
            'noisy_segments': noisy_segments,
            'segment_times': self.get_segment_times(noisy_segments, segment_length)
        }

        return noisy_mel, mask_mel, info
