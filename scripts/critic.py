import torch
from torch import nn


class BasicCritic(nn.Module):
    """
    The BasicCritic module determines if an STFT representation is original or modified.

    Input: STFT features (2 channels - real & imaginary) [N, 2, 360, 360]
    Output: Binary prediction (0: original, 1: steganographic) [N, 1]
    """
    def _name(self):
        return "BasicCritic"

    def _conv2d(self, in_channels, out_channels):
        """Basic convolution block without padding - reduces spatial dimensions"""
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
        )

    def _build_models(self):
        """
        Progressive feature extraction architecture:
        - Layer 1: Initial feature extraction from STFT
        - Layer 2-3: Higher level feature processing
        - Layer 4: Final classification
        """
        self.conv1 = nn.Sequential(
            self._conv2d(self.channels_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size, 1)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def __init__(self, hidden_size, channels_size):
        """
        Args:
            hidden_size: Number of feature channels in hidden layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.channels_size = channels_size
        self._models = self._build_models()
        self.name = self._name()

    def forward(self, image):
        """
        Forward pass through critic:
        1. Extract features through conv layers
        2. Reduce spatial dimensions progressively
        3. Average predictions for final binary classification

        Args:
            image: STFT representation [N, 2, 360, 360]
        Returns:
            Binary prediction per sample [N]
        """
        x = self._models[0](image)
        x_1 = self._models[1](x)
        x_2 = self._models[2](x_1)
        x_3 = self._models[3](x_2)
        return torch.mean(x_3.view(x_3.size(0), -1), dim=1)
