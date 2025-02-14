from typing import Callable, Optional, Union
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from torch import nn
from torchvision import transforms
from PIL import Image
from src.utils.data_utils import torch_to_str, default_transform
import torch
        

class WaterMarkedStableDiffusionPipeline():

    """
    Pipeline for watermarked text-to-image generation using Stable Diffusion.

    Args:
        model_card(str):
            The model_id of the model from huggingface.
            - "CompVis/stable-diffusion-v1-1"
            - "CompVis/stable-diffusion-v1-2"
            - "CompVis/stable-diffusion-v1-3"
            - "CompVis/stable-diffusion-v1-4"
            - "runwayml/stable-diffusion-v1-5"


        method(str): 
            The watermarking method.
            - 'SS': 'Stable Signature'
            - 

    Example:
        >>> device = 'cuda' if torch.cuda.is_availabel() else 'cpu'
        >>> WPipeline = WaterMarkedStableDiffusionPipeline("runwayml/stable-diffusion-v1-5").to(device)
        >>> images = WPipeline("A sunny beach")
    """

    transform= default_transform()

    def __init__(self,
                 model_card: str,
                 method: str = 'SS',
                 ):
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_card)
        self.method = method

        if method == 'SS':
                 
            self.vae = self.pipeline.vae
            stat_dict =  torch.load("ckpts/finetune_vae/checkpoint_000.pth", map_location = 'cpu')
            self.vae.load_state_dict(stat_dict)
            self.msg_decoder = torch.jit.load('ckpts/msg_decoder/dec_48b_whit.torchscript.pt')  
            self.msg_decoder.eval()
    
    def to(self, device: str):
        """move the modules to device"""
        self.pipeline.to(device)
        if hasattr(self, 'msg_decoder') and self.msg_decoder is not None:
            self.msg_decoder.to(device)

    @torch.no_grad()
    def __call__(self, prompts: str, return_msg: bool = False):

        images = self.pipeline(prompts)[0]
        transformed_images = torch.stack([self.transform(image) for image in images])
        messages = self.msg_decoder(transformed_images.to(self.device))
        if return_msg:
            return images, torch_to_str(messages)
        return images
    
    @property
    def device(self):
        return self.pipeline.device
        