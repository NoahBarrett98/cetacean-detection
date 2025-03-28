import h5py
import numpy as np


def filter_hdf5_by_y(input_file, output_file, y_value_to_remove):
    """
    Removes all cases from an HDF5 file where the y array has a specified value.

    Parameters:
        input_file (str): Path to the input HDF5 file.
        output_file (str): Path to the output HDF5 file.
        y_value_to_remove: The value in the y array to filter out.
    """
    with h5py.File(input_file, 'r') as infile:
        X = infile['X'][:]
        y = infile['y'][:]

        # Find indices where y is not equal to the specified value
        valid_indices = np.where(y != y_value_to_remove)[0]

        # Filter X and y
        X_filtered = X[valid_indices]
        y_filtered = y[valid_indices]
        print("size of X:", len(X))
        print("size of X_filtered: ", len(X_filtered))
        print("size of y:", len(y))
        print("size of y_filtered: ", len(y_filtered))
        # Write filtered data to a new HDF5 file
        with h5py.File(output_file, 'w') as outfile:
            outfile.create_dataset('X', data=X_filtered)
            outfile.create_dataset('y', data=y_filtered)
            
if __name__ == "__main__":
    # remove the possibly detected cases: 
    filter_hdf5_by_y(input_file="nefsc_sbnms_200903_nopp6_ch10/processed/intial_run/hdf5/processed_data.h5", 
                     output_file="nefsc_sbnms_200903_nopp6_ch10/processed/intial_run/hdf5/processed_data_no_class_2.h5", 
                     y_value_to_remove= 2)