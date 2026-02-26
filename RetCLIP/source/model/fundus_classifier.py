from torch import nn
import torch

from RetCLIP.source.model.vision_vit import VisionViT
from RetCLIP.source.model.linear_classifier import LinearClassifier


class FundusClassifier(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionViT,
        classifier: LinearClassifier,
    ):
        super(FundusClassifier, self).__init__()
        self.vision_encoder = vision_encoder
        self.classifier = classifier

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_encoder(image)
        logits = self.classifier(image_features)
        return logits
