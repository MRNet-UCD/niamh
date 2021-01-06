import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        self.classifer = nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        flattened_features = torch.max(features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output

