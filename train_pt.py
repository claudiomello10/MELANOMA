import torch
from torch import nn, optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset
import datetime

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the benchmark to True
torch.backends.cudnn.benchmark = True

# Define the data directory
data_directory = "./data"

# Define the number of epochs
NUM_EPOCHS = 100
BATCH_SIZE = 2000


# Training data files and labels
train_data_files = [
    "train_tensor_0.pt",
    "train_tensor_30.pt",
    "train_tensor_45.pt",
    "train_tensor_60.pt",
    "train_tensor_90.pt",
]
train_labels_files = [
    "train_labels_0.pt",
    "train_labels_30.pt",
    "train_labels_45.pt",
    "train_labels_60.pt",
    "train_labels_90.pt",
]


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


# Define the model
model = CNN()

model.to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.BCELoss()


epoch_pb = tqdm(range(NUM_EPOCHS), desc="Epochs", leave=False)


# Train the model
model.train()


for data_file, label_file in zip(train_data_files, train_labels_files):
    print(f"Training with data file: {data_file} and label file: {label_file}")
    train_data = torch.load(os.path.join(data_directory, data_file)).float()
    train_labels = torch.load(os.path.join(data_directory, label_file)).float()

    dataset = TensorDataset(train_data, train_labels)

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True
    )
    epoch_losses = []
    for epoch in epoch_pb:

        batch_losses = []
        batch_pb = tqdm(train_loader, desc="Batches", leave=False)

        for data, label in batch_pb:
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
            batch_losses.append(loss.item())
            batch_pb.set_postfix({"Batch loss": loss.item()})
        epoch_pb.set_postfix({"Epoch loss": sum(batch_losses) / len(batch_losses)})
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    print(f"Epoch losses: {epoch_losses}")
print("\n\nFinished Training!\n\n")


# Get model name based on timestamp
model_name = datetime.datetime.now().strftime("%Y-%m-%d_%H") + ".pth"


# Save the model
try:
    os.makedirs("models")
except FileExistsError:
    pass
try:
    torch.save(model.state_dict(), f"models/{model_name}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
    try:
        print("Attempting to save a backup model...")
        torch.save(model.state_dict(), "BackupModel.pth")
    except Exception as e:
        print(f"An error occurred while saving the backup model: {e}")
        print("Please save the model manually!")
        print(f"Model state dict: {model.state_dict()}")
    else:
        print(
            "Backup model saved in the current directory as BackupModel.pth, please move it to the models directory!"
        )
else:
    print(f"Model saved successfully as {model_name} in the models directory!")
