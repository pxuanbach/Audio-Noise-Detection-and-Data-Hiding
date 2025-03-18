import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HybridCNNAttention(nn.Module):
    def __init__(self, input_channels=5, num_classes=10, dropout_rate=0.5):
        super().__init__()

        # CNN blocks without pooling
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),

            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),

            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128)
        )

        # Global context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1),
            nn.Sigmoid()
        )

        # Single attention head for small feature size
        self.attention = nn.MultiheadAttention(128, 4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # [batch, channels, height, 1]

        # CNN features
        x = self.cnn(x)  # [batch, 128, height, 1]

        # Apply global context
        context = self.global_context(x)
        x = x * context

        # Reshape for attention
        x = x.squeeze(-1).transpose(1, 2)  # [batch, height, channels]
        x = x.transpose(0, 1)  # [height, batch, channels]

        # Self-attention
        x, _ = self.attention(x, x, x)
        x = x.mean(0)  # [batch, channels]

        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
