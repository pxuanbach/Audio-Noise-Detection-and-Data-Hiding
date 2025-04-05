import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, data_dir, normalize=True, cache_size=100):
        self.data_dir = Path(data_dir)

        # Load all data files
        self.features = []
        self.labels = []

        # Process speech files (label 0)
        speech_files = list((self.data_dir / 'speech').glob('*.npy'))
        for file in speech_files:
            self.features.append(np.load(str(file)))
            self.labels.append(0)

        # Process music files (label 1)
        music_files = list((self.data_dir / 'music').glob('*.npy'))
        for file in music_files:
            self.features.append(np.load(str(file)))
            self.labels.append(1)

        # Convert to numpy arrays
        self.features = np.array(self.features)  # Shape: [N, 2, 360, 360]
        self.labels = np.array(self.labels)

        # Add magnitude as third channel
        magnitudes = np.sqrt(self.features[:, 0]**2 + self.features[:, 1]**2)  # Calculate magnitude
        self.features = np.concatenate([
            self.features,
            magnitudes[:, np.newaxis, :, :]
        ], axis=1)  # New shape: [N, 3, 360, 360]

        if normalize:
            # Normalize each channel independently
            for i in range(self.features.shape[1]):  # Loop through real/imag/magnitude
                mu = np.mean(self.features[:, i])
                std = np.std(self.features[:, i])
                self.features[:, i] = (self.features[:, i] - mu) / (std + 1e-8)

        # Convert to tensors
        self.features = torch.FloatTensor(self.features)  # Shape: [N, 3, 360, 360]
        self.labels = torch.LongTensor(self.labels)

        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        features = self.features[idx]
        label = self.labels[idx]

        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = (features, label)

        return features, label
