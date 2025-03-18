import torch
import numpy as np

class AudioTransforms:
    @staticmethod
    def add_noise(features, noise_factor=0.05):
        noise = torch.randn_like(features) * noise_factor
        return features + noise

    @staticmethod
    def random_mask(features, mask_prob=0.1):
        mask = torch.rand_like(features) > mask_prob
        return features * mask

    @staticmethod
    def time_shift(features, max_shift=2):
        shift = np.random.randint(-max_shift, max_shift + 1)
        return torch.roll(features, shifts=shift, dims=-1)
