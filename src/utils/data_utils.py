from typing import List
import numpy as np
import torch

def list_to_str(key: List[int]):
    """
    convert list watermark key to str
    """
    return ''.join(str(bit) for bit in key)

def str_to_list(key: str):
    """
    convert list watermark key to str
    """
    return [int(item) for item in key]

def list_to_numpy(key: List[int]):
    """
    convert list watermark key to numpy NDAaray
    """
    return np.array(key)

def list_to_torch(key: List[int]):
    """
    convert list watermark key to torch Tensor
    """
    return torch.tensor(key, dtype = torch.float32)