import os
import cv2
import tqdm
import torch
import h5py
from torchvision import transforms


FILE_FORMAT = "h5"  # 'pt' or 'h5'


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


# Concatenate the benign and malignant tensors
print("Concatenating the benign and malignant tensors...")
train_tensor = torch.cat((train_benign_tensor, train_malignant_tensor))

# Shuffle the tensor and create labels
print("Shuffling the tensor and creating labels...")
train_labels = torch.cat(
    (
        torch.zeros(len(train_benign_tensor)),
        torch.ones(len(train_malignant_tensor)),
    )
)

del train_benign_tensor, train_malignant_tensor

# Shuffle the train_tensor along with its labels
print("Shuffling the train tensor and labels...")
indices = torch.randperm(len(train_tensor))
train_tensor = train_tensor[indices]
train_labels = train_labels[indices]

del indices

# Convert the tensor to float
print("Converting the tensor to float...")
train_tensor = train_tensor.float()
train_labels = train_labels.float()

# Permute the tensor
print("Permuting the tensor...")
train_tensor = train_tensor.permute(0, 3, 1, 2)


# Save the training tensors as files
print(f"Saving the training tensors as {FILE_FORMAT}...")
if FILE_FORMAT == "pt":
    torch.save(train_tensor, "train_tensor.pt")
    torch.save(train_labels, "train_labels.pt")
elif FILE_FORMAT == "h5":
    with h5py.File("train_tensor.h5", "w") as f:
        f.create_dataset("tensor", data=train_tensor)
    with h5py.File("train_labels.h5", "w") as f:
        f.create_dataset("labels", data=train_labels)
else:
    raise ValueError(f"Invalid file format: {FILE_FORMAT}")

del train_tensor, train_labels


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


# Concatenate the test tensors
print("Concatenating the test tensors...")
test_tensor = torch.cat((test_benign_tensor, test_malignant_tensor))

# Shuffle the test tensor and create labels
print("Shuffling the test tensor and creating labels...")
test_labels = torch.cat(
    (
        torch.zeros(len(test_benign_tensor)),
        torch.ones(len(test_malignant_tensor)),
    )
)


del test_benign_tensor, test_malignant_tensor

# Shuffle the test_tensor along with its labels
print("Shuffling the test tensor and labels...")
indices = torch.randperm(len(test_tensor))
test_tensor = test_tensor[indices]
test_labels = test_labels[indices]

del indices

# Convert the tensor to float
print("Converting the test tensor to float...")
test_tensor = test_tensor.float()
test_labels = test_labels.float()

# Permute the tensor
print("Permuting the test tensor...")
test_tensor = test_tensor.permute(0, 3, 1, 2)

# Save the test tensors as files
print("Saving the test tensors...")
torch.save(test_tensor, "test_tensor.pt")
torch.save(test_labels, "test_labels.pt")

del test_tensor, test_labels
