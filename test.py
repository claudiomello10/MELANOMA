import torch
import h5py
from CNN import CNN

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

# Load the model
model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.to(device)


# Load the test data
h5_file = h5py.File("melanoma.h5", "r")
test_data = torch.tensor(h5_file["test"][-10:]).to(device)
test_labels = torch.tensor(h5_file["test_labels"][-10:]).to(device)

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
