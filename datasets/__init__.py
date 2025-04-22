import os

from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio


AUD_EXTENSIONS = ('.flac', '.wav', '.mp3', '.mp4')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def wav_loader(path):
    with open(path, 'rb') as f:
        sig, sr = torchaudio.load(f.name)
        return sig


class AudioToImageFolder(datasets.DatasetFolder):
    """A generic audio data loader where the images are arranged in this way: """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=wav_loader, is_valid_file=None):
        super(AudioToImageFolder, self).__init__(root, loader, AUD_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = wav_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
            hop_length = sample[1]
            sample = sample[0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path, hop_length

class SingleAudioLoader(Dataset):
    """Loader cho 1 file WAV duy nhất, trả về cả spectrogram và thông số."""

    def __init__(self, file_path, transform=None, target_transform=None, loader=wav_loader):
        self.file_path = file_path
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # Load WAV
        self.sample = self.loader(self.file_path)

    def __getitem__(self, index):
        sample = self.sample

        # Mặc định thông số nếu không có transform custom trả ra
        sr, hop_length, n_fft = 22050, 256, 1024

        if self.transform is not None:
            transformed = self.transform(sample)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                sample, params = transformed
                if isinstance(params, (tuple, list)) and len(params) == 3:
                    sr, hop_length, n_fft = params
            else:
                sample = transformed  # fallback nếu transform không trả ra param

        return sample, self.file_path, (sr, hop_length, n_fft)

    def __len__(self):
        return 1
