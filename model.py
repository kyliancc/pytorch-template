import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.iterations = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(7*7*8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
