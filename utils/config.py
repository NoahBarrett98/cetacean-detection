import yaml
from typing import Dict,  Any, Type, TypeVar

T = TypeVar("T", bound="Config")

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