import os
import cv2
import tqdm
import torch

train_benign_directory = "./melanoma/train/Benign"
train_malignant_directory = "./melanoma/train/Malignant"
csv_file = "train.csv"

# Get all image file names in the directory
train_benign_files = [
    file for file in os.listdir(train_benign_directory) if file.endswith(".jpg")
]
train_malignant_files = [
    file for file in os.listdir(train_malignant_directory) if file.endswith(".jpg")
]

# Get the images as matrixes
train_benign_images = [
    cv2.imread(os.path.join(train_benign_directory, file))
    for file in tqdm.tqdm(
        train_benign_files, desc="Reading training benign images", leave=False
    )
]
train_malignant_images = [
    cv2.imread(os.path.join(train_malignant_directory, file))
    for file in tqdm.tqdm(
        train_malignant_files, desc="Reading training malignant images", leave=False
    )
]


# Convert the images to tensors
train_benign_tensors = [torch.from_numpy(image) for image in train_benign_images]
train_malignant_tensors = [torch.from_numpy(image) for image in train_malignant_images]

# Stack the tensors along the batch dimension
train_benign_tensor = torch.stack(train_benign_tensors)
train_malignant_tensor = torch.stack(train_malignant_tensors)

# Print the shape of the tensors
print("Shape of benign tensor:", train_benign_tensor.shape)
print("Shape of malignant tensor:", train_malignant_tensor.shape)

# Save the tensors as files
torch.save(train_benign_tensor, "train_benign_tensor.pt")
torch.save(train_malignant_tensor, "train_malignant_tensor.pt")


test_benign_directory = "./melanoma/test/Benign"
test_malignant_directory = "./melanoma/test/Malignant"

# Get all image file names in the test directory
test_benign_files = [
    file for file in os.listdir(test_benign_directory) if file.endswith(".jpg")
]
test_malignant_files = [
    file for file in os.listdir(test_malignant_directory) if file.endswith(".jpg")
]

# Get the test images as matrices
test_benign_images = [
    cv2.imread(os.path.join(test_benign_directory, file))
    for file in tqdm.tqdm(
        test_benign_files, desc="Reading test benign images", leave=False
    )
]
test_malignant_images = [
    cv2.imread(os.path.join(test_malignant_directory, file))
    for file in tqdm.tqdm(
        test_malignant_files, desc="Reading test malignant images", leave=False
    )
]

# Convert the test images to tensors
test_benign_tensors = [torch.from_numpy(image) for image in test_benign_images]
test_malignant_tensors = [torch.from_numpy(image) for image in test_malignant_images]

# Stack the test tensors along the batch dimension
test_benign_tensor = torch.stack(test_benign_tensors)
test_malignant_tensor = torch.stack(test_malignant_tensors)

# Print the shape of the test tensors
print("Shape of test benign tensor:", test_benign_tensor.shape)
print("Shape of test malignant tensor:", test_malignant_tensor.shape)

# Save the test tensors as files
torch.save(test_benign_tensor, "benign_tensor_test.pt")
torch.save(test_malignant_tensor, "malignant_tensor_test.pt")
