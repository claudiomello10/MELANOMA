import torch
from torch import nn, optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset
import datetime
import csv
from CNN import CNN

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the benchmark to True
torch.backends.cudnn.benchmark = True

# Define the data directory
data_directory = "./data"

# Define the number of epochs
NUM_EPOCHS = 100
BATCH_SIZE = 300


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


# Define the model
model = CNN()

model.to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0009)

# Define the loss function
criterion = nn.BCELoss()


# Train the model
model.train()


file_pb = tqdm(
    zip(train_data_files, train_labels_files),
    desc="Files",
    leave=False,
    total=len(train_data_files),
)

file_losses = []
for data_file, label_file in file_pb:
    file_pb.set_postfix({"File": data_file})
    train_data = torch.load(os.path.join(data_directory, data_file)).float()
    train_labels = torch.load(os.path.join(data_directory, label_file)).float()

    dataset = TensorDataset(train_data, train_labels)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    epoch_losses = []
    epoch_pb = tqdm(range(NUM_EPOCHS), desc="Epochs", leave=False)
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
    file_losses.append(epoch_losses)
print("\n\nFinished Training!\n\n")

# Print the final loss of each file
for i, loss in enumerate(file_losses):
    print(f"Final loss of file {train_data_files[i]}: {loss[-1]}")


# Get model name based on timestamp
model_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the loss values
try:
    os.makedirs(f"/{data_directory}/training_losses")
except FileExistsError:
    pass
try:
    with open(
        f"{data_directory}/training_losses/file_losses_{model_name}.csv", "w"
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(file_losses)
except Exception as e:
    print(f"An error occurred while saving the loss values: {e}")
else:
    print("Loss values saved successfully in the losses directory!")


# Save the model
try:
    os.makedirs("models")
except FileExistsError:
    pass
try:
    torch.save(model.state_dict(), f"models/{model_name}" + ".pth")
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
