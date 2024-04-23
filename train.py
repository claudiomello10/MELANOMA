import torch.utils
import torch.utils.data
import CNN
import torch
import pandas as pd
import torch.nn as nn
from batch_dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the training parameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 512


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Cria uma instância do conjunto de dados personalizado
custom_dataset = CustomDataset("./melanoma.h5")

# Usa o conjunto de dados personalizado no DataLoader
train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create an instance of the CNN model
model = CNN.CNN()

# Move the model to the device
model.to(device)


# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Train the model

print("Training the model...\n")
epoch_losses = []
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    for data, label in tqdm(train_loader, desc="Batches", leave=False):
        # Move batch data to device (e.g. GPU)
        batch_data = data.to(device)
        batch_labels = label.to(device)

        # Forward pass
        outputs = model(batch_data)

        # Compute the loss
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_losses.append(loss.item())

# Close the dataset
custom_dataset.close()

# Print the final loss
print(f"\n\nFinal loss: {loss.item()}\n")

# Save the losses to a CSV file
loss_df = pd.DataFrame(epoch_losses, columns=["loss"])
loss_df.to_csv("losses.csv", index=False)

# Save the model
torch.save(model.state_dict(), "model.pth")
