import torch
from CNN import CNN
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

# Define directories
DATA_DIRECTORY = "./data"
MODELS_DIRECTORY = "./models"
MODEL_NAME = "2024-04-24_12-54-36.pth"


# Load the model

model_state_dict = torch.load(f"{MODELS_DIRECTORY}/{MODEL_NAME}")

model = CNN()
model.load_state_dict(model_state_dict)
model.to(device)


# Load the test data
test_data = torch.load(f"{DATA_DIRECTORY}/test_tensor.pt").to(device)
test_labels = torch.load(f"{DATA_DIRECTORY}/test_labels.pt").to(device)

# Set the model to evaluation mode
model.eval()


# Forward pass
with torch.no_grad():
    outputs = model(test_data)

    # Apply softmax to the outputs
    predictions = torch.round(outputs)

    # Calculate the accuracy
    correct_predictions = (predictions == test_labels).sum().item()
    total_predictions = test_labels.size(0)
    accuracy = correct_predictions / total_predictions * 100

    # Print the accuracy
    print("Accuracy:", accuracy)
