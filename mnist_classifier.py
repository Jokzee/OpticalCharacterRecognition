import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):

    def __init__(self, n_classes):
        super(MNISTClassifier, self).__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # batch x 8 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1),  # batch x 16 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)  # batch x 32 x 14 x 14
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # batch x 64 x 7 x 7
        )
        self.fcLayers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),  # batch x 256
            nn.Linear(256, 256),  # batch x 256
            nn.Linear(256, n_classes)  # batch x n_classes
        )

    def forward(self, inputs):
        outputs = self.convBlock1(inputs)
        outputs = self.convBlock2(outputs)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = self.fcLayers(outputs)
        return outputs
