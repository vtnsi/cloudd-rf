
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #self.fc1 = nn.Linear(128 * 256,512)  # I dont exactly know why it is 128x256, but I had to do some debugging and hardcode the required value
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
