import torch.utils
import torch.utils.data
import CNN
import torch
import pandas as pd
import torch.nn as nn


# Load the .pt file as a tensor
train_benign_tensor = torch.load("./train_benign_tensor.pt")
train_malignant_tensor = torch.load("./train_malignant_tensor.pt")

# Create labels for the data
train_benign_labels = torch.zeros(len(train_benign_tensor))
train_malignant_labels = torch.ones(len(train_malignant_tensor))

# Concatenate the data and labels
input_data = torch.cat((train_benign_tensor, train_malignant_tensor))
output_data = torch.cat((train_benign_labels, train_malignant_labels))

# Convert to float
input_data = input_data.float()
input_data = input_data.permute(0, 3, 1, 2)

output_data = output_data.float()


# Get the shape of the input data
data_shape = input_data.shape
print("Data shape:", data_shape)


# Create a TensorDataset
tensor_dataset = torch.utils.data.TensorDataset(input_data, output_data)

# Use the TensorDataset in the DataLoader
train_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=32, shuffle=True)


# Create a TensorDataset
tensor_dataset = torch.utils.data.TensorDataset(input_data, output_data)

# Use the TensorDataset in the DataLoader
train_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=32, shuffle=True)

model = CNN.CNN()

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs
num_epochs = 10


# Train the model
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 iterations
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}"
            )
