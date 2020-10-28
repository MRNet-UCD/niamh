import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        #self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        #pooled_features = self.pooling_layer(features)
        #pooled_features = pooled_features.view(pooled_features.size(0), -1)
        #flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        flattened_features = torch.max(features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output
