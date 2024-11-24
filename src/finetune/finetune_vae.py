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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from src.utils.util import parse_params

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, help="Path to the training data directory", required=True)
    aa("--val_dir", type=str, help="Path to the validation data directory", required=True)

    group = parser.add_argument_group('Model parameters')
    aa("--msg_decoder_path", type=str, default= "models/hidden/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type =int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default = 42)

    return parser

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

    def __init__(self, params: argparse.Namespace) -> None:

        self.params = params

        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5") #vae is the same for different versions of stable diffusion
        for model_params in self.vae.parameters():
            model_params.requires_grad = False
        self.finetuned_vae = deepcopy(self.vae)
        for decoder_params in self.finetuned_vae.decoder.parameters():
            decoder_params.requires_grad = True
        self.vae.eval()
        self.msg_decoder = self._load_msg_decoder()
        self._to(self.device)
        
        self.train_loader, self.val_loader = self._get_dataloader()
        self.optimizer = self._build_optimizer()


    def _seed_all(self) -> None:
        "Set the random seed for reproducibility"

        seed = self.params.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    def _get_dataloader(self) -> tuple[DataLoader, DataLoader]:
        
        raise NotImplementedError

    def _build_optimizer(self):
        
        optim_params = parse_params(self.params.optimizer)
        optimizer = optim_params.pop("name", None)
        torch_optimizers = sorted(name for name in torch.optim.__dict__
            if name[0].isupper() and not name.startswith("__")
            and callable(torch.optim.__dict__[name]))
        if hasattr(torch.optim, optimizer):
            return getattr(torch.optim, optimizer)(self.finetuned_vae.decoder.parameters(), **optim_params)
        raise ValueError(f'Unknown optimizer "{optimizer}", choose among {str(torch_optimizers)}')

    def _load_msg_decoder(self) -> nn.Module:

        ckpt_path = self.params.msg_decoder_path
        return torch.jit.load(ckpt_path)
    
    def train_per_key(self):

        raise NotImplementedError
    
    def _to(self, device: str):

        self.vae.to(device)
        self.finetuned_vae.to(device)
        self.msg_decoder.to(device)
        

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