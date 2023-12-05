import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F


class Resnet18(nn.Module):
    def __init__(self, args=None):
        super(Resnet18, self).__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features=2)

    def forward(self, x):
        return self.model(x)
    
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):

        return x


class CNNModel(nn.Module):
    def __init__(self, args=None):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 30 * 40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output 2 floats
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.fc_layers(x)
        return x
    
