import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.soft = nn.Softmax(0)
        self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model .features(x)
        pooled_features = self.pooling_layer(features)
        conv = nn.Conv2d(in_channels=pooled_features.shape[1], out_channels=1, kernel_size=1)
        a = conv(pooled_features)
        a = a.squeeze()
        pooled_features = pooled_features.squeeze()
        a = torch.tanh(a)
        a = self.soft(a)
        pooled_features = pooled_features.permute(1,0)
        output = pooled_features * a
        output = torch.sum(output, dim=1)
        output = self.classifer(output)
        output = output.unsqueeze(0)
        return output, a




