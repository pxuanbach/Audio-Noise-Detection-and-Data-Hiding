import torch
from torch.utils.data import Dataset
import numpy as np
from .audio_transforms import AudioTransforms

class AudioDataset(Dataset):
    def __init__(self, features, labels, normalize=True, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment
        self.transforms = AudioTransforms()

        if normalize:
            # Normalize each feature type (channel) independently
            for i in range(self.features.shape[1]):  # Loop through channels
                mu = np.mean(self.features[:, i, :])
                std = np.std(self.features[:, i, :])
                self.features[:, i, :] = (self.features[:, i, :] - mu) / (std + 1e-8)

        # Convert to tensors - don't add extra dimension
        self.features = torch.FloatTensor(self.features)  # [batch, channels, features]
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]

        if self.augment:
            # Apply random augmentations during training
            if torch.rand(1) < 0.5:
                features = self.transforms.add_noise(features)
            if torch.rand(1) < 0.5:
                features = self.transforms.random_mask(features)
            if torch.rand(1) < 0.5:
                features = self.transforms.time_shift(features)

        return features, self.labels[idx]
