import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data_file = h5py.File(data_path, "r")

    def __len__(self):
        return len(self.data_file["train_labels"])

    def __getitem__(self, idx):
        data = torch.tensor(self.data_file["train"][idx])
        label = torch.tensor(self.data_file["train_labels"][idx])

        return (data, label)

    def close(self):
        self.data_file.close()
