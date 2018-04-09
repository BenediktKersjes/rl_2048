from torch import nn as nn
from torch.nn import functional


class DQN(nn.Module):
    def __init__(self, input_channels):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, (2, 2))
        self.conv2 = nn.Conv2d(128, 128, (2, 2))
        self.linear = nn.Linear(512, 256)
        self.head = nn.Linear(256, 4)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = functional.relu(self.linear(x))
        return self.head(x)