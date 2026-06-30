import torch.nn as nn
import torch
from const import H, W

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ici tu déclares tes couches
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        self.Flatten = nn.Flatten()

        #to compute shape of in_features
        dummy = torch.zeros(1, 1, H, W)
        dummy = self.conv1(dummy)
        dummy = self.relu1(dummy)
        dummy = self.maxPool1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.relu2(dummy)
        dummy = self.maxPool2(dummy)
        in_features = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]

        self.Linear = nn.Linear(in_features, out_features=10)

    def forward(self, x):
        # ici tu décris le chemin du batch x à travers les couches
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = self.Flatten(x)
        x = self.Linear(x)
        return x