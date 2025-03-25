import os
import numpy as np
import h5py
import tqdm
def convert_npz_to_hdf5(npz_dir, output_file):
    # Collect all npz files
    npz_files = [os.path.join(root, file) 
                 for root, _, files in os.walk(npz_dir) 
                 for file in files if file.endswith('.npz')]
    print(f"Found {len(npz_files)} .npz files.")

    # Initialize HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        X_dataset = None
        y_dataset = None
        
        for i, npz_path in tqdm.tqdm(enumerate(npz_files)):
            data = np.load(npz_path)
            X, y = data['X'], data['y']
            
            # Create datasets on first iteration
            if X_dataset is None:
                X_shape = (0,) + X.shape[1:]  # Variable first dimension
                y_shape = (0,) + y.shape[1:]

                X_dataset = h5f.create_dataset('X', shape=X_shape, maxshape=(None,) + X.shape[1:], dtype=X.dtype)
                y_dataset = h5f.create_dataset('y', shape=y_shape, maxshape=(None,) + y.shape[1:], dtype=y.dtype)

            # Append data
            X_dataset.resize((X_dataset.shape[0] + X.shape[0]), axis=0)
            y_dataset.resize((y_dataset.shape[0] + y.shape[0]), axis=0)
            X_dataset[-X.shape[0]:] = X
            y_dataset[-y.shape[0]:] = y

            if i % 100 == 0:
                print(f"Processed {i}/{len(npz_files)} files...")

    print("Conversion complete. Data stored in:", output_file)

# Example usage
convert_npz_to_hdf5('nefsc_sbnms_200903_nopp6_ch10/processed/intial_run/images', 'processed_data.h5')