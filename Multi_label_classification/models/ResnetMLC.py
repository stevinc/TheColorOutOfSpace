import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class ResNetMLC(nn.Module):
    def __init__(self, in_channels=1, out_cls=2, resnet_version=18, pretrained=0, colorization=0):
        super(ResNetMLC, self).__init__()

        if resnet_version == 18:
            if pretrained:
                self.model = models.resnet18(pretrained=True)
            else:
                self.model = models.resnet18(pretrained=False)
        else:
            if pretrained:
                self.model = models.resnet50(pretrained=True)
            else:
                self.model = models.resnet50(pretrained=False)

        # SET THE CORRECT NUMBER OF INPUT CHANNELS
        self.colorization = colorization
        self.in_channels = in_channels
        if self.colorization:
            self.conv_1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if in_channels != 3:
            self.model.conv1 = self.conv_1

        # DEFINE THE FEATURE EXTRACTOR
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        # CLASSIFIER
        num_ftrs = self.model.fc.in_features
        self.classifier = nn.Linear(in_features=num_ftrs, out_features=out_cls)
        self.sigmoid = nn.Sigmoid()

    def set_weights_conv1(self):
        if self.colorization:
            self.feature_extractor[0] = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        # FEATURES-SPACE
        B, F = features.shape[:2]
        # MULTI-LABEL CLASSIFICATION
        out = self.classifier(features.view(B, F))
        return out, self.sigmoid(out).detach()
