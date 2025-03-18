import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetLSTM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_size=512):
        super(UNetLSTM, self).__init__()
        # Input channels = 3 (clean mel, noisy mel, difference)
        # Output channels = 1 (noise labels)

        # Encoding layers
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        self.enc5 = self.conv_block(128, 256)
        self.enc6 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)

        # Decoding layers
        self.up6 = self.upconv(512, 256)
        self.dec6 = self.conv_block(512, 256)

        self.up5 = self.upconv(256, 128)
        self.dec5 = self.conv_block(256, 128)

        self.up4 = self.upconv(128, 64)
        self.dec4 = self.conv_block(128, 64)

        self.up3 = self.upconv(64, 32)
        self.dec3 = self.conv_block(64, 32)

        self.up2 = self.upconv(32, 16)
        self.dec2 = self.conv_block(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Save input size for later use
        input_size = x.size()

        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))

        # Global Average Pooling and LSTM
        gap = self.global_avg_pool(e6).squeeze(-1).squeeze(-1)
        lstm_out, _ = self.lstm1(gap.unsqueeze(1))
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.squeeze(1).unsqueeze(-1).unsqueeze(-1)

        # Decoding with size adjustment
        # Calculate output padding for each upsampling
        d6 = self.up6(e6)
        d6 = F.interpolate(d6, size=e5.shape[2:])
        d6 = self.dec6(torch.cat([d6, e5], dim=1))

        d5 = self.up5(d6)
        d5 = F.interpolate(d5, size=e4.shape[2:])
        d5 = self.dec5(torch.cat([d5, e4], dim=1))

        d4 = self.up4(d5)
        d4 = F.interpolate(d4, size=e3.shape[2:])
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:])
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        # Final processing
        output = self.final_conv(d2)
        output = self.sigmoid(output)

        # Ensure output matches input time dimension
        output = F.adaptive_avg_pool2d(output, (1, input_size[-1]))
        output = output.squeeze(2)  # Remove frequency dimension

        return output
