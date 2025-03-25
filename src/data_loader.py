"""
Torch Dataset Def.
"""
import h5py
import torch
import sys
from typing import Tuple, Union
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
from cetacean_detection.utils.config import DataLoaderConfig

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        """
        Args:
            hdf5_file (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.hdf5_file = hdf5_file
        self.transform = transform

        # Open the file in read-only mode to avoid memory overhead
        with h5py.File(self.hdf5_file, 'r') as f:
            self.data_len = f['X'].shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Open HDF5 within __getitem__ for multiprocessing support
        with h5py.File(self.hdf5_file, 'r') as f:
            X = f['X'][index]
            y = f['y'][index]

        # Convert to Tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

def get_hdf5_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test data loaders from a single HDF5 file
    
    Args:
        hdf5_file (str): Path to the HDF5 file
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        test_ratio (float): Proportion of data for testing
        batch_size (int): Batch size for the data loaders
        transform (callable, optional): Transform to apply to the data
        random_seed (int): Seed for reproducible splits
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create the full dataset
    full_dataset = HDF5Dataset(config["hdf5_file"], transform=config["transform"])
    dataset_size = len(full_dataset)
    
    # Create indices for the splits
    indices = list(range(dataset_size))
    
    # Shuffle indices
    np.random.seed(config["random_seed"])
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(config["train_ratio"] * dataset_size)
    val_size = int(config["val_ratio"] * dataset_size)
    # test_size isn't needed since we'll just take the remainder
    
    # Create the splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Dynamically call the function specified in the config's "entry_function"
    entry_function = config.get("entry_function")
    if not entry_function:
        raise ValueError("The 'entry_function' key must be specified in the config dictionary.")
    
    # Ensure the function exists in the current module
    if entry_function not in globals():
        raise ValueError(f"The function '{entry_function}' is not defined in the current module.")
    
    # Call the function with the provided config
    return globals()[entry_function](config["config"])