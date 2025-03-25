import yaml
from typing import Dict,  Any, Type, TypeVar, Union

T = TypeVar("T", bound="Config")

def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)
    
    @classmethod
    def from_yaml(cls: Type[T], file_path: str) -> Dict[str, T]:
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        
        configs = {}
        for section, values in yaml_data.items():
            configs[section] = cls.from_dict(values.get("config", {}))
        
        return configs
class GeneralConfig(Config):
    mlflow_server_uri: str
    experiment_name: str

class DataLoaderConfig(Config):
    hdf5_file: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    transform: Union[str, None]
    random_seed: int
    
class AstTrainerConfig(Config):
    optimizer: str
    num_epochs: int
    use_scheduler: bool
    lr: float
    # ast model params
    ast_src_dir: str
    label_dim: int
    fstride: int
    tstride: int 
    input_fdim: int 
    input_tdim: int 
    imagenet_pretrain: bool
    audioset_pretrain: bool
    model_size: str
    verbose: bool