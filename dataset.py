import torch
from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import json

class MusanNoiseDataset(Dataset):
    def __init__(self, dataset_dir="musan_dataset", transforms=None):
        self.dataset_dir = dataset_dir
        self.file_list = [f for f in os.listdir(dataset_dir) if f.endswith("_clean_mel.npy")]
        self.transforms = transforms

        # Load metadata
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Lấy đường dẫn file
        clean_file = os.path.join(self.dataset_dir, self.file_list[idx])
        noisy_file = clean_file.replace("_clean_mel.npy", "_noisy_mel.npy")
        noise_file = clean_file.replace("_clean_mel.npy", "_noise.npy")
        label_file = clean_file.replace("_clean_mel.npy", "_labels.npy")

        # Đọc dữ liệu
        clean_mel = np.load(clean_file)
        noisy_mel = np.load(noisy_file)
        noise = np.load(noise_file)
        labels = np.load(label_file)

        # Chuyển thành tensor
        clean_mel = torch.tensor(clean_mel, dtype=torch.float32)
        noisy_mel = torch.tensor(noisy_mel, dtype=torch.float32)
        noise = torch.tensor(noise, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Get metadata
        sample_id = self.file_list[idx].replace("_clean_mel.npy", "")
        metadata = self.metadata[sample_id]

        # Apply transforms if provided
        if self.transforms is not None:
            if 'mel' in self.transforms:
                clean_mel = self.transforms['mel'](clean_mel)
                noisy_mel = self.transforms['mel'](noisy_mel)
            if 'label' in self.transforms:
                labels = self.transforms['label'](labels)
                labels = labels.float()  # Convert to float for BCE loss

        # Stack clean and noisy mel spectrograms along channel dimension
        noisy_mel = torch.stack([clean_mel, noisy_mel, clean_mel - noisy_mel], dim=0)  # Shape: [3, freq, time]

        return {
            "noisy_mel": noisy_mel,  # Shape: [3, freq, time]
            "labels": labels,  # Shape will be [time]
            "metadata": metadata
        }
