import torch
import numpy as np
import json
from pathlib import Path

class MUSANDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dir):
        """
        Load preprocessed MUSAN dataset.
        Args:
            preprocessed_dir: Directory containing preprocessed data
        """
        self.preprocessed_dir = Path(preprocessed_dir)

        # Load metadata
        with open(self.preprocessed_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)

        self.combinations = self.metadata['combinations']
        self.params = self.metadata['params']

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        item = self.combinations[idx]

        # Load preprocessed spectrograms
        noisy_mel = np.load(self.preprocessed_dir / item['spec_path'])
        mask_mel = np.load(self.preprocessed_dir / item['mask_path'])

        # Convert to tensors
        noisy_mel = torch.tensor(noisy_mel, dtype=torch.float32).unsqueeze(0)
        mask_mel = torch.tensor(mask_mel, dtype=torch.float32).unsqueeze(0)

        # Create info dict
        info = {
            'speech_file': str(Path(item['source_file']).name),
            'noise_file': str(Path(item['noise_file']).name),
            'noisy_segments': item['noisy_segments'],
            'segment_times': item['segment_times']
        }

        return noisy_mel, mask_mel, info
