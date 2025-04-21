import os
import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path

AUD_EXTENSIONS = ('.flac', '.wav', '.mp3', '.mp4')

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def wav_loader(path):
    with open(path, 'rb') as f:
        sig, sr = torchaudio.load(f.name)
        return sig

class MUSANDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=wav_loader, is_valid_file=None, data_type='both'):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = []  # Will store (path, label) tuples

        is_valid_file = is_valid_file or (lambda x: has_file_allowed_extension(x, AUD_EXTENSIONS))

        # Collect samples based on data_type
        if data_type in ['both', 'speech']:
            speech_files = [f for f in (self.root / 'speech').glob('*') if is_valid_file(str(f))]
            self.samples.extend([(str(f), 0) for f in speech_files])

        if data_type in ['both', 'music']:
            music_files = [f for f in (self.root / 'music').glob('*') if is_valid_file(str(f))]
            self.samples.extend([(str(f), 1) for f in music_files])

        if len(self.samples) == 0:
            raise ValueError(f"No valid audio files found in {root} for data_type '{data_type}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
            hop_length = sample[1] if isinstance(sample, tuple) else None
            sample = sample[0] if isinstance(sample, tuple) else sample

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path, hop_length if hop_length else None
