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

To run mlflow locally:  
```
mlflow server --port {mlflow_Server_uri}
```

## running pipeline step

To run a pipeline step you must provide the entry function with the path to the yaml config you'd like to use. E.g. 
```
python -m cetacean_detection.src.data.pre_processor --config cetacean_detection/configs/pipeline.yaml
```

## optimizations

In order to feasibly train on smaller GPUs, several different optimziations are being explored: 

* LoRA - Low Rank Adaptation - the use of LoRA particularly on the qkv layers is used to bring the trainable parameter size down to ~2% of the original models trainable parameters. 
* Mixed Precision - in order to speedup compute times, mixed precision is used during training. 
* Balanced sampling - Remove excess negative samples from training set (leads to better computational efficiency, but also a more balanced representation of the problem space.)

### run time analysis (10 batches): 
- no optimizations:
    - Self CPU time total: 105.053s
    Self CUDA time total: 114.283s
- with just lora:
    - Self CPU time total: 63.559s
    Self CUDA time total: 77.730s
- with lora plus mixed precision + pin_memory + 8 workers in data loader + model.half() precision and torch.backends.cudnn.benchmark = True:
    - Self CPU time total: 52.942s
    Self CUDA time total: 59.733s

## experiments

The first experiment will look at applying the Audio Spectrogram Transformer model to the DCLDE 2013 datast - this approach will only consider the Detected, and Not Detected cases, Possibly Detected will be left for later experimentation. 
* removing the Possibly Detected data will result in the reduction of samples from 196844 to 194493 cases. 
* given that the majority of these cases are negative samples, we will ensure that balanced sampling is used to downsample the negative class. 
  * problem is extremely imbalanced - 188588 negative vs.   5905 positive