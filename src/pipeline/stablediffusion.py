from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from torch import nn
from torchvision import transforms
from PIL import Image

import torch
        

class WaterMarkedStableDiffusionPipeline():

    """
    Pipeline for watermarked text-to-image generation using Stable Diffusion.

    Args:
        model_card (str):
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
        >>> WPipeline.generate("A sunny beach")
    """

    transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                           std=[0.5, 0.5, 0.5])])

    def __init__(self,
                 model_card: str,
                 method: str = 'SS',
                 ):
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_card)
        self.vae = self.pipeline.vae
        stat_dict =  torch.load("ckpts/finetune_vae/checkpoint_000.pth", map_location = 'cpu')
        self.vae.load_state_dict(stat_dict)
        #self.unet = self.pipeline.unet
        #self.text_encoder = self.pipeline.text_encoder

        if method == 'SS':
                 
            #TODO: replace path of vae decoder and msg decoder
            #self.vae.decoder = torch.jit.load('path/to/ckpt')
            self.msg_decoder = torch.jit.load('ckpts/msg_decoder/dec_48b_whit.torchscript.pt')  
            self.msg_decoder.eval()
    

    def to(self, device: str):
        """move the modules to device"""

        self.pipeline.to(device)
        self.msg_decoder.to(device)

    @torch.no_grad()
    def generate(self, prompts: str, return_msg: bool = False):

        images = self.pipeline(prompts)[0]
        transformed_images = torch.stack([self.transform(image) for image in images])
        messages = self.msg_decoder(transformed_images.to(self.device))
        if return_msg:
            return images, messages
        return images
    
    @property
    def device(self):
        return self.pipeline.device
        

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

if __name__ == '__main__':

    """

    torch.manual_seed(42)
    pipe = WaterMarkedStableDiffusionPipeline("CompVis/stable-diffusion-v1-4", "SS", fine_tune = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(pipe.vae.encoder.training) 
    print(pipe.vae.decoder.training)
    print(pipe.unet.training)  
    print(pipe.text_encoder.training) 
    """

    #pipe.to(device)

    #prompts = ["A sunny beach", "A snowy mountain"]

    #images, messages = pipe.generate(prompts)
    """
    for i, img in enumerate(images):
        img.save(f"test{i + 2}.png")
        msg = (messages[i] > 0).tolist()
        msg = msg2str(msg)
        print(msg)
    """
    import warnings
    warnings.filterwarnings('ignore')
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_pipe = WaterMarkedStableDiffusionPipeline("runwayml/stable-diffusion-v1-5")
    w_pipe.to(device)
    images, msgs = w_pipe.generate('A man is walking on the road', return_msg = True)
    for i, img in enumerate(images):
        img.save(f"test{i}.png")
        msg = (msgs[i] > 0).tolist()
        print(msg)
    
    #print(pipeline.vae)