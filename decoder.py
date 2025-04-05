import torch
from torch import nn


class BasicDecoder(nn.Module):
    """
    Modified decoder for STFT steganographic data:
    Input: Encoded STFT (2 channels - real & imaginary) [N, 2, H, W]
    Output: Decoded hidden data [N, data_depth, H, W]
    """

    def _conv2d(self, in_channels, out_channels):
        """Basic conv2d block with 3x3 kernel and padding"""
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        # First layer: Process STFT input
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),  # Changed from 3 to 2 channels
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
            self._conv2d(self.hidden_size, self.data_depth)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image):
        x = self._models[0](image)
        x_1 = self._models[1](x)
        x_2 = self._models[2](x_1)
        x_3 = self._models[3](x_2)
        return x_3

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()


class DenseDecoder(BasicDecoder):
    """
    Dense decoder with enhanced feature extraction:
    - Progressive feature fusion through dense connections
    - Efficient for complex STFT patterns
    - Each layer has access to all previous features
    """

    def _build_models(self):
        # Initial feature extraction from STFT
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )

        # Progressive feature fusion
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )

        # Dense connection with accumulated features
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        # Final layer to reconstruct hidden data
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3, self.data_depth),
            nn.Tanh()  # Added to normalize output
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image):
        # Progressive feature accumulation
        x = self._models[0](image)
        x_list = [x]

        # Dense connections
        x_1 = self._models[1](torch.cat(x_list, dim=1))
        x_list.append(x_1)

        x_2 = self._models[2](torch.cat(x_list, dim=1))
        x_list.append(x_2)

        # Final decoding with all accumulated features
        x_3 = self._models[3](torch.cat(x_list, dim=1))

        return x_3
