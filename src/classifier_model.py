import torch.nn as nn
from torchvision import models

class CityVillageClassifier(nn.Module):
    def __init__(self):
        super(CityVillageClassifier, self).__init__()
        # ZMIANA: Dodajemy weights="DEFAULT" zamiast None
        # Model pobierze gotowe wagi, które świetnie rozumieją tekstury
        self.model = models.resnet18(weights="DEFAULT") 
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.model(x)