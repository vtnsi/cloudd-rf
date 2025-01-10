
import torch
import torch.nn as nn
import torch.nn.functional as F

# From headley_modrec.py
class Team3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, (2, 16))
        self.conv2 = nn.Conv2d(16, 8, (1, 8))
        self.conv3 = nn.Conv2d(8, 4, (1, 4))
        #self.fc1 = nn.Linear(3996, 512)
        self.fc1 = nn.LazyLinear(out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.num_classes)

        self.activation = {}

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
