import torch
import torch.profiler
from torch.cuda.amp import autocast, GradScaler
from peft import LoraConfig, get_peft_model
torch.backends.cudnn.benchmark = True
import tqdm
from cetacean_detection.src.evaluator import evaluate_classification
from cetacean_detection.src import optimizers
from typing import Tuple
from torch.utils.data import DataLoader
from cetacean_detection.src.data_loader import get_data_loaders
from cetacean_detection.src.models import get_model
from cetacean_detection.utils.config import flatten_dict
import mlflow
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_step(pbar, train_loader, val_loader, optimizer, model, criterion, scheduler):
    running_loss = 0.0
    scaler = GradScaler()
    
    for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader, 0)):
        # put on gpu
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        # Enable autocast for mixed precision
        with autocast():
            outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            loss = criterion(outputs, labels)

        # Use GradScaler to scale the loss and call backward
        scaler.scale(loss).backward()

        # Step the optimizer with scaled gradients
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    if scheduler:
        scheduler.step()
    torch.save(model, "C:/Users/NoahB/OneDrive/Desktop/cetacean_detection/experiments/run_0/model.pt")
    # get validation loss #
    with torch.no_grad():
        val_running_loss = 0.0
        for i, (inputs, labels) in enumerate(val_loader, 0):
            inputs, labels = inputs.cuda(), labels.cuda()
            # outputs = model(inputs)
            with autocast():
                outputs = torch.nn.functional.softmax(model(inputs), dim=1)
                loss = criterion(outputs, labels)
            val_running_loss += loss.item()

    pbar.set_description(f"Loss: {running_loss / len(train_loader):.6f}")
    print(f"running loss: {running_loss}")
    # train_results = evaluate_classification(
    #     model, train_loader
    # )
    val_results = evaluate_classification(
        model, val_loader
    )
    return val_results, running_loss, val_running_loss
        
def train_classification(data_loaders, model, config: dict) -> Tuple[float, float, torch.nn.Module]:
    """main training function for classification
    """
    # unpack data loaders
    train_loader, val_loader, test_loader = data_loaders
    # get optimizer
    optimizer = optimizers.__dict__[config.get("optimizer")](model, config.get("lr"))
    # set scheduler
    if config.get("use_scheduler"):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200], gamma=0.5
        )
    else:
        scheduler = None
    # train model
    pbar = tqdm.tqdm(range(config.get("num_epochs")))
    # CE for classification
    criterion = torch.nn.CrossEntropyLoss()
    # training
    for epoch in pbar:
        val_results, running_loss, val_running_loss = train_step(pbar, train_loader, val_loader, optimizer, model, criterion, scheduler)
        
        logs = {
            "loss/train": running_loss / len(train_loader),
            "loss/val": val_running_loss / len(val_loader),
            "accuracy/val": val_results["accuracy"], 

        }
        mlflow.log_metrics(logs, step=epoch)
        # mlflow.log_param("loss/train", running_loss / len(train_loader))
        # mlflow.log_param("accuracy/train", train_results["accuracy"])
        # mlflow.log_param("auc/train", train_results["auc"])
        # mlflow.log_param("loss/val", val_running_loss / len(val_loader))
        # mlflow.log_param("accuracy/val", val_results["accuracy"])
        # mlflow.log_param("auc/val", val_results["auc"])
    
    # evaluate model
    test_eval_results = evaluate_classification(model, test_loader)
    val_eval_results = evaluate_classification(model, val_loader)
    print("***************")
    print(type(model))
    print("***************")
    return test_eval_results, val_eval_results, model

def run_training_pipeline(data_loaders: Tuple[DataLoader, DataLoader, DataLoader], model: torch.nn.Module, config: dict) -> Tuple[float, float, torch.nn.Module]:
    # Dynamically call the function specified in the config's "entry_function"
    entry_function = config.get("entry_function")
    if not entry_function:
        raise ValueError("The 'entry_function' key must be specified in the config dictionary.")
    
    # Ensure the function exists in the current module
    if entry_function not in globals():
        raise ValueError(f"The function '{entry_function}' is not defined in the current module.")
    
    # Call the function with the provided config
    return globals()[entry_function](data_loaders, model, config["config"])

def model_trainer(config: dict):
    """
    general model trainer entry function
    """
    # log run to mlflow
    logging.info("starting mlflow run")
    mlflow.set_tracking_uri(config["general"]["mlflow_server_uri"])
    mlflow.set_experiment(config["general"]["experiment_name"])
    experiment = mlflow.get_experiment_by_name(config["general"]["experiment_name"])
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        logging.info("logging config")
        
        flat_config = flatten_dict(config)
        mlflow.log_params(flat_config)
        # preloaded data to prevent reloading
        logging.info("getting data loaders")
        data_loaders = get_data_loaders(config.get("data_loader"))
        
        # load model
        logging.info("getting model")
        model = get_model(config.get("model")).cuda().half() # use float 16 to benefit from mixed precision

        lora_config = LoraConfig(
            **config["model_trainer"]["config"]["LoRA_kwargs"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # run pipeline
        logging.info("running training pipeline")
        run_training_pipeline(data_loaders, model, config.get("model_trainer"))

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Load configuration from YAML file")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    if not config:
        raise ValueError("Config model_trainer entry not found")
    model_trainer(config)