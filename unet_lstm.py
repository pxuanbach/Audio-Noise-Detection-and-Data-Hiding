import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, hidden_size=512, num_lstm_layers=2):
        super(UNetLSTM, self).__init__()

        # Encoding path
        self.enc1 = self.conv_block(input_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        self.enc5 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM part
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True)

        # Decoding path
        self.dec5 = self.conv_block(256 + hidden_size, 256)
        self.dec4 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.dec2 = self.conv_block(64, 32)
        self.dec1 = self.conv_block(32, 16)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Remove the unsqueeze since input is already [batch, channel, height, width]
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Global Average Pooling
        gap = self.global_avg_pool(e5)
        gap = gap.view(gap.size(0), gap.size(1))  # Flatten to [batch, channels]

        # LSTM
        lstm_out, _ = self.lstm(gap.unsqueeze(1))  # Add sequence dimension
        lstm_out = lstm_out[:, -1, :]  # Take last output

        # Decoding with skip connections
        d5 = self.upsample(torch.cat([e5, lstm_out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, e5.size(2), e5.size(3))], dim=1))
        d5 = self.dec5(d5)
        d4 = self.upsample(torch.cat([d5, e4], dim=1))
        d4 = self.dec4(d4)
        d3 = self.upsample(torch.cat([d4, e3], dim=1))
        d3 = self.dec3(d3)
        d2 = self.upsample(torch.cat([d3, e2], dim=1))
        d2 = self.dec2(d2)
        d1 = self.upsample(torch.cat([d2, e1], dim=1))
        d1 = self.dec1(d1)

        return self.final_conv(d1)
