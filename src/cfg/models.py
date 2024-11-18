import torch
import torch.nn as nn
import torch.nn.functional as F


class Team1Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 16))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 8))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 8))
        self.fc1 = nn.LazyLinear(out_features=1200)
        self.fc2 = nn.Linear(in_features=1200, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=65)
        self.fc4 = nn.Linear(in_features=65, out_features=self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

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

    def forward(self, x):
        x = x.squeeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Team3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, (2, 16))
        self.conv2 = nn.Conv2d(16, 8, (1, 8))
        self.conv3 = nn.Conv2d(8, 4, (1, 4))
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

class Team4Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.re1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.re2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.re3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.re4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.flat = nn.Flatten()
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(out_features=512)

        self.re5 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.re6 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.re1(self.conv1(x)))
        x = self.pool2(self.re2(self.conv2(x)))
        x = self.pool3(self.re3(self.conv3(x)))
        x = self.pool4(self.re4(self.conv4(x)))
        x = self.flat(x)
        x = self.drop1(x)
        x = self.re5(self.fc1(x))
        x = self.drop2(x)
        x = self.re6(self.fc2(x))
        x = self.fc3(x)
        return x
