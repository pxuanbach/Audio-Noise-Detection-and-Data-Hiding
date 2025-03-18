import torch

class MelSpectrogramTransform:
    def __init__(self, max_length=None, hop_length=512, frame_size=1024):
        self.max_length = max_length
        self.hop_length = hop_length
        self.frame_size = frame_size

    def __call__(self, mel_spectrogram):
        if self.max_length is None:
            return mel_spectrogram

        curr_len = mel_spectrogram.shape[1]
        if curr_len < self.max_length:
            # Pad if shorter
            pad_length = self.max_length - curr_len
            return torch.nn.functional.pad(mel_spectrogram, (0, pad_length))
        # Truncate if longer
        return mel_spectrogram[:, :self.max_length]

class LabelTransform:
    def __init__(self, max_length=None, hop_length=512, frame_size=1024):
        self.max_length = max_length
        self.hop_length = hop_length
        self.frame_size = frame_size

    def __call__(self, labels):
        if self.max_length is None:
            return labels

        # Calculate target length based on model output size
        target_length = self.max_length // (self.frame_size // self.hop_length)

        if len(labels) < target_length:
            return torch.nn.functional.pad(labels, (0, target_length - len(labels)))
        return labels[:target_length]
