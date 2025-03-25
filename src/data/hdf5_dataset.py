import h5py
import torch
from torch.utils.data import Dataset, DataLoader

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

        if self.transform:
            X = self.transform(X)

        return X, y

# Example Usage
def get_dataloader(hdf5_path, batch_size=64, num_workers=4, shuffle=True, transform=None):
    dataset = HDF5Dataset(hdf5_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader


from torch.utils.data import Subset, DataLoader
import numpy as np

def create_data_loaders(hdf5_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                        batch_size=32, transform=None, random_seed=42):
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
    full_dataset = HDF5Dataset(hdf5_file, transform=transform)
    dataset_size = len(full_dataset)
    
    # Create indices for the splits
    indices = list(range(dataset_size))
    
    # Shuffle indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    hdf5_file = r'C:\Users\NoahB\OneDrive\Desktop\cetacean_detection\nefsc_sbnms_200903_nopp6_ch10\processed\intial_run\hdf5\processed_data.h5'
    # Example: Load data
    dataloader = get_dataloader(hdf5_file)
    
    train_loader, val_loader, test_loader = create_data_loaders(hdf5_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                        batch_size=32, transform=None, random_seed=42)
    
    for X_batch, y_batch in dataloader:
        print(X_batch.shape, y_batch.shape)
        break