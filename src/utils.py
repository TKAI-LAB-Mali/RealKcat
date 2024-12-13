import torch
import numpy as np
import random

# Check if GPU is available and return the appropriate device
def check_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a random seed for reproducibility across different modules and libraries
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Optional: Print and log information about the device for verification
def print_device_info(device):
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

# Utility function to convert a PyTorch tensor to numpy, handling device compatibility
def tensor_to_numpy(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

# Utility function to convert a numpy array to a PyTorch tensor and move it to the specified device
def numpy_to_tensor(array, device):
    return torch.tensor(array).to(device)
