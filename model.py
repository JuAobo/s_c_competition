import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientDualPool(nn.Module):
    def __init__(self, num_classes, version='efficientnet-b0'):
        super().__init__()
        self.base = EfficientNet.from_pretrained(version, num_classes=num_classes)
        self.planes = self.base._fc.in_features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce_layer = nn.Conv2d(self.planes*2, self.planes, 1)
        self.fc = nn.Sequential(
            self.base._dropout,
            self.base._fc,
            )

    def forward(self, x):
        bs = x.shape[0]
        x = self.base.extract_features(x)
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce_layer(x).view(bs, -1)
        logits = self.fc(x)
        return logits
