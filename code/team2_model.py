
import torch
import torch.nn as nn
import torch.nn.functional as F

class Team2Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 128, 8, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 8, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 8, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 1024, 8, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(1024),
        )

        self.fc1 = nn.LazyLinear(512, dtype=torch.float32)
        self.fc2 = nn.Linear(512, num_classes, dtype=torch.float32)
        # self.fc = nn.Sequential(
        #     nn.LazyLinear(512, dtype=torch.float32),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes, dtype=torch.float32),
        #     nn.ReLU(),
        # )

    def forward(self, x):
        x = x.squeeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        #return F.softmax(x, dim=1)
        return x
