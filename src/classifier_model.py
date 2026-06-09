import torch.nn as nn
from torchvision import models


class CityVillageClassifier(nn.Module):
    """ResNet-18 (pretrained) z jednym wyjściem: logit miasto(1) vs wieś(0)."""

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)
