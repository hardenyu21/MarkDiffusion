"""The code is modified based on https://github.com/facebookresearch/stable_signature """

import argparse
import json
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple, Callable, Iterable
import torch.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from utils.param_utils import parse_optim_params

class Trainer():

    ##TODO: add comments
    """
    Trainer class for fine-tuning VAE models in Stable Diffusion.

    This class handles loading the pre-trained VAE model, preparing the dataloaders,
    and building the optimizer for training or fine-tuning.

    Attributes:
        vae (AutoencoderKL): The pre-trained VAE model.
        finetuned_vae (AutoencoderKL): A deepcopy of the pre-trained VAE for fine-tuning.
        params (argparse.Namespace): Command-line arguments parsed using argparse.
    """

    def __init__(self, params: OmegaConf) -> None:

        self.params = params

        #generate key before sed seed
        self.watermark_key = self._generate_key(params.num_bits)
        #set seed
        self._seed_all(params.seed)

        #Initialize vae decoder and msg_decoder
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5") #vae is the same for different versions of stable diffusion
        for model_params in self.vae.parameters():
            model_params.requires_grad = False
        self.finetuned_vae = deepcopy(self.vae)
        for decoder_params in self.finetuned_vae.decoder.parameters():
            decoder_params.requires_grad = True
        self.vae.eval()
        self.msg_decoder = self._load_msg_decoder()
        self._to(self.device)
        
        #dataset
        self.train_loader, self.val_loader = self._get_dataloader()
        self.optimizer = self._build_optimizer()


    def _seed_all(self, seed: int) -> None:
        "Set the random seed for reproducibility"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    def _generate_key(self, num_bits: int) -> None:

        bits = random.choice([0, 1], k = num_bits)
        return bits
        
    def _get_dataloader(self) -> tuple[DataLoader, DataLoader]:
        
        raise NotImplementedError

    def _build_optimizer(self) -> torch.optim.optimizer:
        
        optimizer = self.params.optimizer
        optim_params = parse_optim_params(self.params)

        torch_optimizers = sorted(name for name in torch.optim.__dict__
            if name[0].isupper() and not name.startswith("__")
            and callable(torch.optim.__dict__[name]))
        if hasattr(torch.optim, optimizer):
            return getattr(torch.optim, optimizer)(self.finetuned_vae.decoder.parameters(), **optim_params)
        raise ValueError(f'Unknown optimizer "{optimizer}", choose among {str(torch_optimizers)}')

    def _load_msg_decoder(self) -> nn.Module:

        ckpt_path = self.params.msg_decoder_path
        return torch.jit.load(ckpt_path)
    
    def _to(self, device: str):

        self.vae.to(device)
        self.finetuned_vae.to(device)
        self.msg_decoder.to(device)
        
    def train_per_key(self):

        raise NotImplementedError
        

    def train(self) -> None:
        
        num_keys = self.params.num_keys
        for key_index in range(num_keys):
            self.train_per_key()

        raise NotImplementedError

    def evaluate(self) -> None:

        raise NotImplementedError
    
    @property
    def device(self):

        return "cuda" if torch.cuda.is_available() else "cpu"