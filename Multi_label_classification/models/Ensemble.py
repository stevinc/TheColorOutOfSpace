import torch
import torch.nn as nn
from typing import Tuple


class EnsembleModel(nn.Module):
    def __init__(self, model_rgb, model_colorization, device=0):
        super(EnsembleModel, self).__init__()
        self.device = device
        self.features_rgb = model_rgb.feature_extractor
        self.rgb_classifier = model_rgb.classifier
        self.features_colorization = model_colorization.feature_extractor
        self.colorization_classifier = model_colorization.classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        # split the input tensor
        indices = torch.tensor([3, 2, 1])
        indices_spectral = torch.tensor([0, 4, 5, 6, 7, 8, 9, 10, 11])
        rgb = torch.index_select(input=x, dim=1, index=indices.to(self.device))
        spectral = torch.index_select(input=x, dim=1, index=indices_spectral.to(self.device))
        # FEATURES EXTRACTION
        features_rgb = self.features_rgb(rgb)
        out_rgb = self.rgb_classifier(features_rgb.view(B, 512))
        features_colorization = self.features_colorization(spectral)
        out_colorization = self.colorization_classifier(features_colorization.view(B, 512))
        # output concat
        out = torch.stack((out_rgb, out_colorization), 2)
        out = out.mean(2)
        # sigmoid concat
        out_sigmoid_rgb = self.sigmoid(out_rgb)
        out_sigmoid_colorization = self.sigmoid(out_colorization)
        out_sigmoid = torch.stack((out_sigmoid_rgb, out_sigmoid_colorization), 2)
        out_sigmoid = out_sigmoid.mean(2)
        return out, out_sigmoid.detach()
