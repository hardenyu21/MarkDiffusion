from omegaconf import OmegaConf

def get_params(path: str):
    
    """load parameters for exoeriment from yaml file"""

    params = OmegaConf.load(path)

    return params

def parse_optim_params(params):

    optim_params = {}

    if "lr" in optim_params:
        optim_params["lr"] = optim_params.lr
    
    if "weight_decay" in optim_params:
        optim_params["weight_decay"] = optim_params.weight_decay