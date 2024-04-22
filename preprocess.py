import os
import cv2
import tqdm
import torch
from torchvision import transforms

train_benign_directory = "./melanoma/train/Benign"
train_malignant_directory = "./melanoma/train/Malignant"
csv_file = "train.csv"

# Get all image file names in the directory for benign
train_benign_files = [
    file for file in os.listdir(train_benign_directory) if file.endswith(".jpg")
]


# Get the images as matrixes for benign
train_benign_images = [
    cv2.imread(os.path.join(train_benign_directory, file))
    for file in tqdm.tqdm(
        train_benign_files, desc="Reading training benign images", leave=False
    )
]

del train_benign_files

# Define the transformation for rotating the images
transform30 = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomRotation(30), transforms.ToTensor()]
)

trandform45 = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomRotation(45), transforms.ToTensor()]
)

transform60 = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomRotation(60), transforms.ToTensor()]
)

transform90 = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomRotation(90), transforms.ToTensor()]
)

transform0 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


# Apply the transformation to the benign images
train_benign_rotated30 = [
    transform30(image)
    for image in tqdm.tqdm(
        train_benign_images, desc="Rotating 30 degrees benign images", leave=False
    )
]


train_benign_rotated45 = [
    trandform45(image)
    for image in tqdm.tqdm(
        train_benign_images, desc="Rotating 45 degrees benign images", leave=False
    )
]

train_benign_rotated60 = [
    transform60(image)
    for image in tqdm.tqdm(
        train_benign_images, desc="Rotating 60 degrees benign images", leave=False
    )
]


train_benign_rotated90 = [
    transform90(image)
    for image in tqdm.tqdm(
        train_benign_images, desc="Rotating 90 degrees benign images", leave=False
    )
]

train_benign_images0 = [
    transform0(image)
    for image in tqdm.tqdm(
        train_benign_images, desc="No rotation benign images", leave=False
    )
]

del train_benign_images


# Stack the rotated benign tensors along the batch dimension
print("Stacking the benign tensors...")
train_benign_rotated30 = torch.stack(train_benign_rotated30)
train_benign_rotated45 = torch.stack(train_benign_rotated45)
train_benign_rotated60 = torch.stack(train_benign_rotated60)
train_benign_rotated90 = torch.stack(train_benign_rotated90)
train_benign_images0 = torch.stack(train_benign_images0)


# Stack all the benign tensors along the batch dimension
print("Concatenating all the benign tensors...")
train_benign_tensor = torch.cat(
    (
        train_benign_rotated30,
        train_benign_rotated45,
        train_benign_rotated60,
        train_benign_rotated90,
        train_benign_images0,
    )
)

del (
    train_benign_rotated30,
    train_benign_rotated45,
    train_benign_rotated60,
    train_benign_rotated90,
    train_benign_images0,
)


# Print the shape of the benign training tensors
print("Shape of benign tensor:", train_benign_tensor.shape)

# add the labels to the tensor
train_benign_labels = torch.zeros(len(train_benign_tensor))

# Add the labels to the tensor in the last dimension
train_benign_tensor = torch.cat(
    (train_benign_tensor, train_benign_labels.unsqueeze(1)), dim=1
)

del train_benign_labels

# Get all image file names in the directory for malignant
train_malignant_files = [
    file for file in os.listdir(train_malignant_directory) if file.endswith(".jpg")
]

# Get the images as matrixes for malignant
train_malignant_images = [
    cv2.imread(os.path.join(train_malignant_directory, file))
    for file in tqdm.tqdm(
        train_malignant_files, desc="Reading training malignant images", leave=False
    )
]

del train_malignant_files

# Apply the transformation to the malignant images
train_malignant_rotated30 = [
    transform30(image)
    for image in tqdm.tqdm(
        train_malignant_images, desc="Rotating 30 degrees malignant images", leave=False
    )
]

train_malignant_rotated45 = [
    trandform45(image)
    for image in tqdm.tqdm(
        train_malignant_images, desc="Rotating 45 degrees malignant images", leave=False
    )
]

train_malignant_rotated60 = [
    transform60(image)
    for image in tqdm.tqdm(
        train_malignant_images, desc="Rotating 60 degrees malignant images", leave=False
    )
]

train_malignant_rotated90 = [
    transform90(image)
    for image in tqdm.tqdm(
        train_malignant_images, desc="Rotating 90 degrees malignant images", leave=False
    )
]

train_malignant_images0 = [
    transform0(image)
    for image in tqdm.tqdm(
        train_malignant_images, desc="No rotation malignant images", leave=False
    )
]

del train_malignant_images

# Stack the rotated malignant tensors along the batch dimension
print("Stacking the malignant tensors...")
train_malignant_rotated30 = torch.stack(train_malignant_rotated30)
train_malignant_rotated45 = torch.stack(train_malignant_rotated45)
train_malignant_rotated60 = torch.stack(train_malignant_rotated60)
train_malignant_rotated90 = torch.stack(train_malignant_rotated90)
train_malignant_images0 = torch.stack(train_malignant_images0)

# Stack all the malignant tensors along the batch dimension
print("Concatenating all the malignant tensors...")
train_malignant_tensor = torch.cat(
    (
        train_malignant_rotated30,
        train_malignant_rotated45,
        train_malignant_rotated60,
        train_malignant_rotated90,
        train_malignant_images0,
    )
)

del (
    train_malignant_rotated30,
    train_malignant_rotated45,
    train_malignant_rotated60,
    train_malignant_rotated90,
    train_malignant_images0,
)

# Print the shape of the malignant training tensors
print("Shape of malignant tensor:", train_malignant_tensor.shape)

# Add the labels to the tensor
train_malignant_labels = torch.ones(len(train_malignant_tensor))

# Add the labels to the tensor in the last dimension
train_malignant_tensor = torch.cat(
    (train_malignant_tensor, train_malignant_labels.unsqueeze(1)), dim=1
)

del train_malignant_labels

# Concatenate the benign and malignant tensors
print("Concatenating the benign and malignant tensors...")
train_tensor = torch.cat((train_benign_tensor, train_malignant_tensor))

del train_benign_tensor, train_malignant_tensor

# Shuffle the tensor
train_tensor = train_tensor[torch.randperm(train_tensor.size()[0])]
train_tensor = train_tensor.permute(0, 3, 1, 2)


# Print the shape of the training tensors
print("Shape of training tensor:", train_tensor.shape)

# Save the training tensor as a file
torch.save(train_tensor, "train_tensor.pt")

del train_tensor


########################### Test data preprocessing ###########################

# Define the test directories
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

del test_benign_files

test_malignant_images = [
    cv2.imread(os.path.join(test_malignant_directory, file))
    for file in tqdm.tqdm(
        test_malignant_files, desc="Reading test malignant images", leave=False
    )
]

del test_malignant_files


# Convert the test images to tensors
test_benign_tensors = [
    transforms.ToTensor()(image)
    for image in tqdm.tqdm(
        test_benign_images, desc="Converting test benign images to tensors", leave=False
    )
]

test_malignant_tensors = [
    transforms.ToTensor()(image)
    for image in tqdm.tqdm(
        test_malignant_images,
        desc="Converting test malignant images to tensors",
        leave=False,
    )
]

del test_benign_images, test_malignant_images

# Stack the test tensors along the batch dimension
print("Stacking the test tensors...")
test_benign_tensor = torch.stack(test_benign_tensors)
test_malignant_tensor = torch.stack(test_malignant_tensors)

# Print the shape of the test tensors
print("Shape of test benign tensor:", test_benign_tensor.shape)
print("Shape of test malignant tensor:", test_malignant_tensor.shape)

# Add the labels to the test tensors
test_benign_labels = torch.zeros(len(test_benign_tensor))
test_malignant_labels = torch.ones(len(test_malignant_tensor))

# Add the labels to the test tensors in the last dimension
test_benign_tensor = torch.cat(
    (test_benign_tensor, test_benign_labels.unsqueeze(1)), dim=1
)

del test_benign_labels

test_malignant_tensor = torch.cat(
    (test_malignant_tensor, test_malignant_labels.unsqueeze(1)), dim=1
)

del test_malignant_labels

# Concatenate the test tensors
print("Concatenating the test tensors...")
test_tensor = torch.cat((test_benign_tensor, test_malignant_tensor))

del test_benign_tensor, test_malignant_tensor

# Shuffle the test tensor
test_tensor = test_tensor[torch.randperm(test_tensor.size()[0])]
test_tensor = test_tensor.permute(0, 3, 1, 2)

# Print the shape of the test tensor
print("Shape of test tensor:", test_tensor.shape)

# Save the test tensor as a file
torch.save(test_tensor, "test_tensor.pt")

del test_tensor
