# Dataset

This project is playing around with an updated version of the DLDCE 2013 workshop dataset, updated by NOAA to include all Baleen whale calls. To pull  the data set gcloud cli must be installed, use the command: 

```
gsutil -m cp -r \
  "gs://noaa-passive-bioacoustic/dclde/2013/nefsc_sbnms_200903_nopp6_ch10" \
  .
```

# Running pipeline

## configs 

This project uses a yaml config system for running experiments. see ```cetacean_detection/utils/config/py``` for the definition of the config template, in ```cetacean_detection/configs/pipeline.yaml``` a sample template is provided. 

## mlflow

This project uses mlflow to track experiment runs. To set the mlflow tracking uri see the ```mlflow_server_uri``` and ```experiment_name``` params in the sample template. 

To run mlflow locally:  ```mlflow server --port {mlflow_Server_uri}```

## running pipeline step

To run a pipeline step you must provide the entry function with the path to the yaml config you'd like to use. E.g. ```python -m cetacean_detection.src.data.pre_processor --config cetacean_detection/configs/pipeline.yaml```