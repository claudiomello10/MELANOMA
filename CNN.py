# Description: This file contains the CNN architecture for the model.
import torch.nn as nn


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=4, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=6, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 54 * 54, 1)  # 64 * 54 * 54 = 186624
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
