import torch.nn as nn
import sys

def get_ast_model(config: dict) -> nn.Module:
    # provide access to external ast dir
    sys.path.append(config.get("ast_src_dir"))
    from ast_models import ASTModel
    # load model
    return ASTModel(**config.get("AST_kwargs"))

def get_model(config: dict) -> nn.Module:
    # Dynamically call the function specified in the config's "entry_function"
    entry_function = config.get("entry_function")
    if not entry_function:
        raise ValueError("The 'entry_function' key must be specified in the config dictionary.")
    
    # Ensure the function exists in the current module
    if entry_function not in globals():
        raise ValueError(f"The function '{entry_function}' is not defined in the current module.")
    
    # Call the function with the provided config
    return globals()[entry_function](config["config"])