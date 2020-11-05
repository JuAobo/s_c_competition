import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class EfficientDualPool(nn.Module):
    def __init__(self, num_classes, version='efficientnet-b0'):
        super().__init__()
        self.base = EfficientNet.from_pretrained(version, num_classes=num_classes)
        self.planes = self.base._fc.in_features
        self.fc = nn.Sequential(
            self.base._dropout,
            self.base._fc,
            )
        self.ca = ChannelAttention(self.planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        bs = x.shape[0]
        x = self.base.extract_features(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.base._bn1(x)
        x = self.base._avg_pooling(x)
        x = x.view(bs, -1)
        logits = self.fc(x)
        return logits
