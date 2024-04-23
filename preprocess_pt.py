import os
import cv2
from tqdm import tqdm
import torch
from torchvision import transforms

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


nameDict = {
    transform30: "30",
    trandform45: "45",
    transform60: "60",
    transform90: "90",
    transform0: "0",
}


########################### Training data preprocessing ###########################


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
    for file in tqdm(
        train_benign_files, desc="Reading training benign images", leave=False
    )
]

train_malignant_images = [
    cv2.imread(os.path.join(train_malignant_directory, file))
    for file in tqdm(
        train_malignant_files, desc="Reading training malignant images", leave=False
    )
]


# Concatenate images and labes into a single list
train_images = train_benign_images + train_malignant_images


train_labels = [0] * len(train_benign_images) + [1] * len(train_malignant_images)

del train_benign_files, train_malignant_files


# Shuffle the images and labels
indices = torch.randperm(len(train_images))
train_images = [train_images[i] for i in indices]
train_labels = [train_labels[i] for i in indices]

del indices


for transform in tqdm(
    [transform30, trandform45, transform60, transform90, transform0],
    desc="Processing training images",
):
    images = [
        transform(image)
        for image in tqdm(
            train_images, desc=f"Transform: {nameDict[transform]}", leave=False
        )
    ]
    tensor = torch.stack(images)
    del images

    torch.save(tensor, f"./data/train_tensor_{nameDict[transform]}.pt")
    torch.save(
        torch.tensor(train_labels), f"./data/train_labels_{nameDict[transform]}.pt"
    )
    del tensor

del train_images, train_labels


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

# Get the images as matrixes
test_benign_images = [
    cv2.imread(os.path.join(test_benign_directory, file))
    for file in tqdm(test_benign_files, desc="Reading test benign images", leave=False)
]

test_malignant_images = [
    cv2.imread(os.path.join(test_malignant_directory, file))
    for file in tqdm(
        test_malignant_files, desc="Reading test malignant images", leave=False
    )
]

# Concatenate images and labes into a single list
test_images = test_benign_images + test_malignant_images

test_labels = [0] * len(test_benign_images) + [1] * len(test_malignant_images)

del test_benign_files, test_malignant_files

# Shuffle the images and labels
indices = torch.randperm(len(test_images))
test_images = [test_images[i] for i in indices]
test_labels = [test_labels[i] for i in indices]

del indices

for transform in [transform0]:
    images = [
        transform(image)
        for image in tqdm(test_images, desc="Transform: 0", leave=False)
    ]
    tensor = torch.stack(images)
    del images

    torch.save(tensor, f"./data/test_tensor.pt")
    torch.save(torch.tensor(test_labels), f"./data/test_labels.pt")
    del tensor
