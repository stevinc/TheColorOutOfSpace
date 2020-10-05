import torch
import torch.nn as nn
from torchvision import models

from Colorization.models.Decoder import resnet18_decoder


class ResNet18(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, pretrained=0, dropout=0.3):
        super(ResNet18, self).__init__()
        if pretrained:
            self.model = models.resnet18(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=False)
        # set the number of input channels
        if in_channels != 3:
            self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.conv1 = self.conv_1
        # feature extractor definition
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        # decoder definition
        self.decoder = resnet18_decoder(stride=2, out_channels=out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(spectral)
        features = self.dropout(features)
        recon = self.decoder(features)
        return recon


if __name__ == '__main__':
    x = torch.ones((16, 12, 128, 128)).cuda()
    ab = torch.ones((16, 12, 128, 128)).cuda()
    net = ResNet18(in_channels=9, out_channels=2, pretrained=0).cuda()
    recon = net(x)
    print("that's all folks!")