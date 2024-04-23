import os
import tqdm
import torch
import torchvision
from torchvision import transforms
import h5py

# Define the directories for the training and test data
train_benign_directory = "./melanoma/train/Benign"
train_malignant_directory = "./melanoma/train/Malignant"
test_benign_directory = "./melanoma/test/Benign"
test_malignant_directory = "./melanoma/test/Malignant"

OUTPUT_NAME = "melanoma.h5"  # Output file name

# Get all image file names in the directory for benign
train_benign_files = [
    (file, 0) for file in os.listdir(train_benign_directory) if file.endswith(".jpg")
]

# Get all image file names in the directory for malignant
train_malignant_files = [
    (file, 1) for file in os.listdir(train_malignant_directory) if file.endswith(".jpg")
]


# Concatenate the benign and malignant file names
train_files = train_benign_files + train_malignant_files

# Shuffle the file names
import random

random.shuffle(train_files)


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


# Create an h5 file to store the training tensors
try:
    os.remove(OUTPUT_NAME)
except FileNotFoundError:
    pass
with h5py.File(OUTPUT_NAME, "x") as f:
    f.create_dataset("train", (0, 3, 224, 224), maxshape=(None, 3, 224, 224))
    f.create_dataset("train_labels", (0,), maxshape=(None,))
    for file, label in tqdm.tqdm(
        train_files, desc="Processing training images", leave=False
    ):
        directory = train_benign_directory if label == 0 else train_malignant_directory
        image = torchvision.io.read_image(
            os.path.join(directory, file),
            torchvision.io.image.ImageReadMode.RGB,
        )
        image30 = transform30(image)
        image45 = trandform45(image)
        image60 = transform60(image)
        image90 = transform90(image)
        image0 = transform0(image)

        # Resize the dataset to add 5 more images and labels
        f["train"].resize(f["train"].shape[0] + 5, axis=0)
        f["train"][-5:] = torch.stack([image30, image45, image60, image90, image0])
        f["train_labels"].resize(f["train_labels"].shape[0] + 5, axis=0)
        f["train_labels"][-5:] = torch.tensor([label] * 5)

    ########################### Test data preprocessing ###########################

    print("\nProcessing the test data...\n")
    # Get benign file names in the test directory
    test_benign_files = [
        (file, 0) for file in os.listdir(test_benign_directory) if file.endswith(".jpg")
    ]

    # Get malignant file names in the test directory
    test_malignant_files = [
        (file, 1)
        for file in os.listdir(test_malignant_directory)
        if file.endswith(".jpg")
    ]

    # Concatenate the benign and malignant file names
    test_files = test_benign_files + test_malignant_files

    # Shuffle the file names
    random.shuffle(test_files)

    f.create_dataset("test", (0, 3, 224, 224), maxshape=(None, 3, 224, 224))
    f.create_dataset("test_labels", (0,), maxshape=(None,))
    for file, label in tqdm.tqdm(
        test_files, desc="Processing test images", leave=False
    ):
        directory = test_benign_directory if label == 0 else test_malignant_directory
        image = torchvision.io.read_image(
            os.path.join(directory, file),
            torchvision.io.image.ImageReadMode.RGB,
        )

        image0 = transform0(image)

        # Resize the dataset to add 1 more image and label
        f["test"].resize(f["test"].shape[0] + 1, axis=0)
        f["test"][-1:] = torch.stack([image0])
        f["test_labels"].resize(f["test_labels"].shape[0] + 1, axis=0)
        f["test_labels"][-1:] = torch.tensor([label])


print("Data preprocessing is complete.")
