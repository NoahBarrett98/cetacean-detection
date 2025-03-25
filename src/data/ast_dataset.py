from cetacean_detection.src.data.pre_processor import get_detections_for_clip, get_clip_time_delta_labels
from scipy.io import wavfile
import os

def generate_ast_dataset(wav_file, config):
    detections = get_detections_for_clip(
        config.detection_log_path, os.path.basename(wav_file)
    )
    labels = get_clip_time_delta_labels(detections, os.path.basename(wav_file))
    sample_rate, data = wavfile.read(wav_file)
    X, y = [], []
    for i, row in labels.iterrows():
        start_offset = int(row["sample_start_time"] * sample_rate)
        end_offset = int(row["sample_end_time"] * sample_rate)
        # Extract the segment of the WAV file corresponding to the positive detection
        data_segment = data[start_offset:end_offset]
        label = config.label_mapping[row["Detection_Confidence"]]
        X.append(data_segment)
        y.append(label)
    return sample_rate, X, y