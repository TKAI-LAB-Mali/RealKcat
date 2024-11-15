import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE

# Custom Dataset class for handling tensor data with labels
class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Load and prepare datasets, moving them to the specified device (e.g., GPU)
def load_and_prepare_datasets(train_path, val_path, test_path, device):
    # Load data from specified paths
    train_data, train_y1 = torch.load(train_path)
    val_data, val_y1 = torch.load(val_path)
    test_data, test_y1 = torch.load(test_path)

    # Move data to device (GPU or CPU)
    train_data, train_y1 = train_data.to(device), train_y1.to(device)
    val_data, val_y1 = val_data.to(device), val_y1.to(device)
    test_data, test_y1 = test_data.to(device), test_y1.to(device)

    # Create dataset objects
    train_dataset = TensorDataset(train_data, train_y1)
    val_dataset = TensorDataset(val_data, val_y1)
    test_dataset = TensorDataset(test_data, test_y1)

    return train_dataset, val_dataset, test_dataset

# Function for global standardization for two feature groups (X1 and X2)
def standardize_x_global_separate(data, global_mean_1, global_std_1, global_mean_2, global_std_2):
    X1, X2 = data[:, :1280], data[:, 1280:]
    global_std_1, global_std_2 = torch.clamp(global_std_1, min=1e-7), torch.clamp(global_std_2, min=1e-7)
    X1_standardized = (X1 - global_mean_1) / global_std_1
    X2_standardized = (X2 - global_mean_2) / global_std_2
    return torch.cat((X1_standardized, X2_standardized), dim=1).squeeze(1)

# Custom Dataset class for global standardization
class StandardizedDatasetGlobalSeparate(Dataset):
    def __init__(self, subset, global_mean_1, global_std_1, global_mean_2, global_std_2):
        self.subset = subset
        self.global_mean_1 = global_mean_1
        self.global_std_1 = global_std_1
        self.global_mean_2 = global_mean_2
        self.global_std_2 = global_std_2

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y1 = self.subset[idx]
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        x_standardized = standardize_x_global_separate(x, self.global_mean_1, self.global_std_1, self.global_mean_2, self.global_std_2)
        return x_standardized, y1

# Function to compute global mean and std for each feature group (first 1280 and the rest) from the train dataset
def get_train_stats_separate(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_batch = next(iter(train_loader))
    train_data, _ = train_batch
    X1, X2 = train_data[:, :1280], train_data[:, 1280:]
    global_mean_1, global_std_1 = X1.mean(), X1.std()
    global_mean_2, global_std_2 = X2.mean(), X2.std()
    return global_mean_1, global_std_1, global_mean_2, global_std_2

# Apply global standardization to the dataset using precomputed stats
def apply_global_standardization_separate(dataset, global_mean_1, global_std_1, global_mean_2, global_std_2):
    return StandardizedDatasetGlobalSeparate(dataset, global_mean_1, global_std_1, global_mean_2, global_std_2)
