import glob
import os
import argparse
import pandas as pd
import tqdm
import mlflow
from typing import Dict, List

from pre_processor import PreProcessConfig, generate_dataset
from cetacean_detection.utils.config import Config, GeneralConfig

class PreProcessConfig(Config):
    detection_log_path: str
    processed_output_dir: str
    wav_dir: str
    label_mapping: Dict[str, int]  # mapping for dataset labels to ints
    figsize: List[int]  # fig size for generated spectrograms
    ylim: List[float]  # ylim for plt plot
    dpi: int  # dpi for generated spectrograms
    shading: str  # shading param for generated spectrograms
    window: str  # window for computing spectrogram
    nperseg: int  # window
    noverlap: int  # overlap window

def preprocess(config: PreProcessConfig) -> None:
    """
    access config using config.parameter
    """
    wav_files = glob.glob(os.path.join(config.wav_dir, "*.wav"))
    output_dir = os.path.join(config.processed_output_dir, config.identifier_tag)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'images'))
    already_processed = [
        os.path.basename(x).split(".")[0]
        for x in glob.glob(os.path.join(output_dir, "images", "*.npz"))
    ]
    wav_files = [f for f in wav_files if not any(a in f for a in already_processed)]
    labels_file = os.path.join(output_dir, "labels.csv")
    for wav_file in tqdm.tqdm(wav_files):
        label_df = generate_dataset(wav_file, output_dir, config)
        if os.path.isfile(labels_file):
            label_df = pd.concat(
                [pd.read_csv(labels_file).drop(columns=["Unnamed: 0"]), label_df]
            )
            label_df.to_csv(labels_file)
        else:
            label_df.to_csv(labels_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from YAML file")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    
    args = parser.parse_args()
    
    configs = Config.from_yaml(args.config)
    preprocess_config: PreProcessConfig = configs.get("preprocess", PreProcessConfig())
    general_config: GeneralConfig = configs.get("general", GeneralConfig())
    # run preprocessing
    preprocess(preprocess_config)
    # log run to mlflow
    mlflow.set_tracking_uri(general_config.mlflow_server_uri)
    mlflow.set_experiment(general_config.experiment_name)
    experiment = mlflow.get_experiment_by_name(general_config.experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_params(preprocess_config.__dict__)
        mlflow.log_params(general_config.__dict__)
        mlflow.end_run()