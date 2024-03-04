import torch
import torch.nn as nn

class CustomLossFunction(nn.Module):
    def __init__(self,alpha=0.4):
        super(CustomLossFunction, self).__init__()
        self.alpha=alpha

    def forward(self, predicted, target):
        # Custom loss calculation
        mean_throttle=torch.mean((predicted[:][0] - target[:][0]) ** 2)
        mean_angle=torch.mean((predicted[:][1] - target[:][1]) ** 2)
        loss = torch.mean(self.alpha*mean_throttle+(1-self.alpha)*mean_angle)
        return loss