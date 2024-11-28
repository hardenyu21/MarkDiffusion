from omegaconf import OmegaConf
import math

def get_params(path: str):
    
    """load parameters for exoeriment from yaml file"""

    params = OmegaConf.load(path)

    return params

def parse_optim_params(params):

    optim_params = {}

    if "lr" in params:
        optim_params["lr"] = params.lr
    
    if "weight_decay" in params:
        optim_params["weight_decay"] = params.weight_decay

def adjust_learning_rate(optimizer, step, steps, warmup_steps, blr, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < warmup_steps:
        lr = blr * step / warmup_steps 
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr