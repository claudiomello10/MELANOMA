import torch

import torch.nn as nn


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=6, stride=2, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=6, stride=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 1)  # 32 * 9 * 9 = 2592
        self.fc2 = nn.Sigmoid()

    def forward(self, x):

        # Define the forward pass
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the tensor
        x = x.reshape(x.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(1)

        return x
