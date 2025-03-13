from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import re
from scipy.io import wavfile
from scipy.signal import spectrogram
from collections import defaultdict

"""
preprocessing dor dldce 2013 dataset

output file = .npy file, with entries X and y, where X is a (N, i, j) array amd y is a (N,) array
where N is the number of samples, and i, j are the dimensions of the outputted spectrograms.  
"""


def get_detections_for_clip(detection_log_path, wav_filename):
    """
    Get all detections from the detection log that fall within the timespan of a given WAV file.

    Args:
        detection_log_path (str): Path to the CSV detection log file
        wav_filename (str): Filename of the WAV file (format: NOPP6_EST_YYYYMMDD_HHMMSS_CH10.wav)

    Returns:
        pd.DataFrame: Filtered dataframe containing only detections within the WAV file's timespan
    """
    # Read the detection log
    df = pd.read_csv(detection_log_path)

    # Parse the WAV filename to get the start time
    datetime_str = wav_filename.split("_")[2:4]  # ['20090328', '000000']
    clip_start = pd.to_datetime(
        f"{datetime_str[0]}_{datetime_str[1]}", format="%Y%m%d_%H%M%S"
    )
    clip_end = clip_start + timedelta(minutes=15)

    # Convert detection timestamps to datetime objects and remove timezone info
    df["start_time"] = pd.to_datetime(df["Start_DateTime_ISO8601"]).dt.tz_localize(None)
    df["end_time"] = pd.to_datetime(df["End_DateTime_ISO8601"]).dt.tz_localize(None)
    # # Filter detections that fall within the clip's timespan
    mask = (df["start_time"] >= clip_start) & (df["start_time"] < clip_end)
    filtered_df = df[mask].copy()

    return filtered_df

def get_clip_time_delta_labels(detections_df: pd.DataFrame, wav_filename: str, time_delta: int = 3):
    
    match = re.search(r"(\d{8})_(\d{6})", wav_filename)
    date_part, time_part = match.groups()
    reference_time_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
    reference_time = pd.to_datetime(reference_time_str)
    
    # Compute time delta in seconds relative to the parsed start time
    time_start_delta = (detections_df["start_time"] - reference_time).dt.total_seconds().astype(int)
    
    durations = (detections_df["end_time"] - detections_df["start_time"]).dt.total_seconds()
    detections_df["sample_start_time"] = time_start_delta + np.maximum(np.floor(durations / 2) - 1, 0).astype(int)
    detections_df["sample_end_time"] = detections_df["sample_start_time"] + time_delta
    # get negative samples
    # 900: 15minutes = 900s
    return create_labeled_intervals(detections_df, "sample_start_time", "sample_end_time", time_range_start=0, time_range_end=900, time_delta=time_delta)

def create_labeled_intervals(df, start_col, end_col, time_range_start=0, time_range_end=900, time_delta=3):
    """
    Process a dataframe with time intervals, labeling existing intervals as positive 
    and adding missing intervals as negative.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time intervals
    start_col : str
        Column name for the start time of intervals
    end_col : str
        Column name for the end time of intervals
    time_range_start : int
        Start of the overall time range to check
    time_range_end : int
        End of the overall time range to check
    time_delta : int
        Size of each interval to check
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with labeled intervals including both original and added negative intervals
    """
    # Create a copy of the input dataframe
    original_df = df.copy()
    
    # Create a list to store the negative intervals
    negative_intervals = []
    
    # Iterate through the time range with the specified time_delta
    for i in range(time_range_start, time_range_end, time_delta):
        interval_start = i
        interval_end = i + time_delta
        
        # Check if this interval overlaps with any in the original dataframe
        overlaps = False
        for _, row in original_df.iterrows():
            # Check for overlap: 
            # Two intervals overlap if one's start is less than the other's end
            # and one's end is greater than the other's start
            if (interval_start < row[end_col] and interval_end > row[start_col]):
                overlaps = True
                break
        
        # If no overlap was found, add this as a negative interval
        if not overlaps:
            negative_intervals.append({
                start_col: interval_start,
                end_col: interval_end,
                'Detection_Confidence': 'Not Detected'
            })
    
    # Create a dataframe from the negative intervals
    negative_df = pd.DataFrame(negative_intervals)
    
    # Combine the original and negative dataframes
    combined_df = pd.concat([original_df, negative_df], ignore_index=True)
    
    # Sort by the start time
    combined_df = combined_df.sort_values(by=start_col).reset_index(drop=True)
    
    return combined_df[['sample_start_time', 'sample_end_time', 'Detection_Confidence']]
  
     
def compute_spectrogram(data, sample_rate, start_time, end_time):
    # compute offsets
    start_offset = int(start_time * sample_rate)
    end_offset = int(end_time * sample_rate)
    # Extract the segment of the WAV file corresponding to the positive detection
    data_segment = data[start_offset:end_offset]
    # Compute the spectrogram
    frequencies, times, Sxx = spectrogram(data_segment, fs=sample_rate, window='hann', 
                            nperseg=256,  # 128 ms window 0.128 * 2000 sr
                            noverlap=256 - 100)  # 50 ms advance 0.05 * 2000 sr
    freq_mask = (frequencies >= 39.06) & (frequencies <= 357.56)
    # limit freq
    frequencies = frequencies[freq_mask]
    Sxx = Sxx[freq_mask, :]
    # normalize
    Sxx /= np.sum(np.square(Sxx))
    
    return frequencies, times, Sxx

def generate_dataset(detection_log_path, wav_file, output_dir):
    detections = get_detections_for_clip(detection_log_path, os.path.basename(wav_file))
    labels = get_clip_time_delta_labels(detections, os.path.basename(wav_file))
    sample_rate, data = wavfile.read(wav_file)
    # make label dirs
    samples = defaultdict(lambda: defaultdict(list))
    # samples = {label: defaultdict(list) for label in labels['Detection_Confidence'].unique()}
    
    for i, row in labels.iterrows():
        frequencies, times, Sxx = compute_spectrogram(data, sample_rate, row['sample_start_time'], row['sample_end_time'])
        samples[row['Detection_Confidence']]['frequencies'].append(frequencies)
        samples[row['Detection_Confidence']]['times'].append(times)
        samples[row['Detection_Confidence']]['Sxx'].append(Sxx)
        
    for label, data in samples.items():
        filename = os.path.join(output_dir, f"{label}.npz")
        np.savez(filename, frequencies=data['frequencies'], times=data['times'], Sxx=data['Sxx'])
        
# Example usage:
if __name__ == "__main__":
    detection_log_path = r"C:\Users\NoahB\OneDrive\Desktop\cetacean_detection\nefsc_sbnms_200903_nopp6_ch10\detections\NEFSC_SBNMS_200903_NOPP6_CH10_upcall-detection-log.csv"
    wav_file = r"C:\Users\NoahB\OneDrive\Desktop\cetacean_detection\nefsc_sbnms_200903_nopp6_ch10\source-audio\NOPP6_EST_20090328_000000_CH10.wav"
    output_dir = r"C:\Users\NoahB\OneDrive\Desktop\cetacean_detection\nefsc_sbnms_200903_nopp6_ch10\processed"
    generate_dataset(detection_log_path, wav_file, output_dir)
