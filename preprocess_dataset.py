from math import floor
import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path

class DatasetPreprocessor:
    def __init__(self, musan_dir, output_dir, sr=16000, n_mels=128, max_duration=3,
                 noise_ratio=0.5, augment_factor=3):
        self.musan_dir = musan_dir
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.noise_ratio = noise_ratio
        self.n_fft = 1024
        self.hop_length = 256
        # Calculate max_frames from audio parameters
        self.max_frames = floor((sr * max_duration) / self.hop_length)
        self.augment_factor = augment_factor
        self.max_samples = sr * max_duration

        # Create output directories
        self.specs_dir = self.output_dir / "spectrograms"
        self.masks_dir = self.output_dir / "masks"
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # Store metadata
        self.metadata = {
            'params': {
                'sr': sr,
                'n_mels': n_mels,
                'max_duration': max_duration,
                'noise_ratio': noise_ratio,
                'max_frames': self.max_frames,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length
            },
            'combinations': []
        }

    def process_audio_pair(self, source_file, noise_file, start_time, save_idx):
        """Process a single pair of source and noise files"""
        # Load source audio segment
        speech, _ = librosa.load(source_file, sr=self.sr, offset=start_time,
                               duration=self.max_duration)

        # Load and prepare noise
        noise, _ = librosa.load(noise_file, sr=self.sr)

        # Process audio similar to original dataset
        speech = self._prepare_audio(speech)
        noise = self._prepare_audio(noise)

        # Create segments and add noise
        segment_length = self.sr // 2
        num_segments = len(speech) // segment_length
        mask = np.zeros_like(speech)
        noisy_segments = []

        # Add noise to random segments
        for i in range(num_segments):
            if random.random() < self.noise_ratio:
                start = i * segment_length
                end = start + segment_length
                noise_segment = noise[start:end] * 0.3
                speech[start:end] += noise_segment
                mask[start:end] = 1
                noisy_segments.append(i)

        # Normalize
        speech = speech / (np.max(np.abs(speech)) + 1e-8)

        # Convert to mel spectrograms
        noisy_mel = librosa.feature.melspectrogram(
            y=speech, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mask_mel = librosa.feature.melspectrogram(
            y=mask, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Process spectrograms
        noisy_mel = librosa.power_to_db(noisy_mel, ref=np.max)
        noisy_mel = (noisy_mel - noisy_mel.min()) / (noisy_mel.max() - noisy_mel.min() + 1e-8)
        mask_mel = (mask_mel > 0).astype(np.float32)

        # Adjust spectrogram size to max_frames
        if noisy_mel.shape[1] > self.max_frames:
            # Cut if longer
            noisy_mel = noisy_mel[:, :self.max_frames]
            mask_mel = mask_mel[:, :self.max_frames]
        elif noisy_mel.shape[1] < self.max_frames:
            # Pad if shorter
            pad_width = self.max_frames - noisy_mel.shape[1]
            noisy_mel = np.pad(noisy_mel, ((0, 0), (0, pad_width)), mode='constant')
            mask_mel = np.pad(mask_mel, ((0, 0), (0, pad_width)), mode='constant')

        # Save spectrograms
        spec_path = self.specs_dir / f"spec_{save_idx}.npy"
        mask_path = self.masks_dir / f"mask_{save_idx}.npy"
        np.save(spec_path, noisy_mel)
        np.save(mask_path, mask_mel)

        # Return metadata
        return {
            'id': save_idx,
            'spec_path': str(spec_path.relative_to(self.output_dir)),
            'mask_path': str(mask_path.relative_to(self.output_dir)),
            'source_file': source_file,
            'noise_file': noise_file,
            'start_time': start_time,
            'noisy_segments': noisy_segments,
            'segment_times': [(i * segment_length / self.sr,
                             (i + 1) * segment_length / self.sr)
                            for i in noisy_segments]
        }

    def _prepare_audio(self, audio):
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        return audio

    def preprocess_dataset(self):
        """Main preprocessing function"""
        # Get file lists
        speech_files = self._get_audio_files("speech")
        noise_files = self._get_audio_files("noise")
        music_files = self._get_audio_files("music")

        # Generate combinations
        combinations = []
        save_idx = 0

        for source_file in tqdm(speech_files + music_files, desc="Processing files"):
            duration = librosa.get_duration(path=source_file)
            if duration < 1.0:
                continue

            num_segments = int(duration / self.max_duration)
            if num_segments <= 0:
                continue

            for seg_idx in range(min(num_segments, self.augment_factor)):
                noise_samples = random.sample(
                    noise_files,
                    min(len(noise_files), self.augment_factor)
                )

                for noise_file in noise_samples:
                    metadata = self.process_audio_pair(
                        source_file, noise_file,
                        seg_idx * self.max_duration,
                        save_idx
                    )
                    combinations.append(metadata)
                    save_idx += 1

        # Save metadata
        self.metadata['combinations'] = combinations
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_audio_files(self, subdir):
        path = os.path.join(self.musan_dir, subdir)
        files = []
        for root, _, filenames in os.walk(path):
            for fname in filenames:
                if fname.endswith('.wav'):
                    files.append(os.path.join(root, fname))
        return sorted(files)

if __name__ == "__main__":
    preprocessor = DatasetPreprocessor(
        musan_dir="D:/Backup/musan",
        output_dir="./preprocessed_dataset2",
        noise_ratio=0.3,
        augment_factor=3,
        max_duration=5,
    )
    preprocessor.preprocess_dataset()
