
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
        #self.fc1 = nn.Linear(in_features=15360, out_features=1200)  # Fully Connected Layer
        self.fc1 = nn.LazyLinear(out_features=1200)
        self.fc2 = nn.Linear(in_features=1200, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=65)
        self.fc4 = nn.Linear(in_features=65, out_features=self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(f'after conv1: {x.shape}')
        x = self.pool(x)
        # print(f'after pool: {x.shape}')
        x = F.relu(self.conv2(x))
        # print(f'after conv2: {x.shape}')
        x = self.pool(x)
        # print(f'after pool: {x.shape}')
        x = F.relu(self.conv3(x))
        # print(f'after conv3: {x.shape}')
        x = self.pool(x)
        # print(f'after pool: {x.shape}')
        x = x.reshape(x.shape[0], -1)
        # print(f'after reshaping: {x.shape}')
        x = F.relu(self.fc1(x))
        # print(f'fc1: {x.shape}')
        x = F.relu(self.fc2(x))
        # print(f'fc2: {x.shape}')
        x = F.relu(self.fc3(x))
        # print(f'fc3: {x.shape}')
        #x = F.relu(self.fc4(x))
        x = self.fc4(x)
        # print(f'fc3: {x.shape}')

        return x
