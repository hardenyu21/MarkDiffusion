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
            - "CompVis/stable-diffusion-v1-4"


        method(str): 
            The watermarking method.
            - 'SS': 'Stable Signature(https://openaccess.thecvf.com/content/ICCV2023/papers/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.pdf)'
            - 
        
        fine_tune(bool):
            Indicates whether some modules of the pipeline are fine-tuned.
            - If method == 'SS': fine tune the vae decoder 
            - 
    """

    transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    def __init__(self,
                 model_card: str,
                 method: str,
                 fine_tune: bool = False
                 ):
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_card)
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.modules = [self.vae, self.unet, self.text_encoder]

        if method == 'SS':
            if fine_tune: 
                for module in self.modules:
                    for params in module.parameters():
                        params.requries_grad = False
                for params in self.vae.decoder:
                    params.requires_grad = True
            
            else:
                self.vae.decoder = torch.jit.load('path/to/ckpt')
           
            #TODO: replace path
            self.msg_decoder = torch.jit.load('/hpc2hdd/home/yhuang489/DiffusionMark/MarkDiffusion/extractor/ckpts/dec_48b_whit.torchscript.pt')  
            self.msg_decoder.eval()
    

    def to(self, device: str):

        self.pipeline.to(device)
        self.msg_decoder.to(device)

    
    def generate(self, prompts: str):

        images = self.pipeline(prompts)[0]
        transformed_images = torch.stack([self.transform(image) for image in images])
        messages = self.msg_decoder(transformed_images.to(self.device))

        return images, messages
    
    @property
    def device(self):
        return self.pipeline.device

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

if __name__ == '__main__':

    torch.manual_seed(42)
    pipe = WaterMarkedStableDiffusionPipeline("CompVis/stable-diffusion-v1-4", "SS")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe.to(device)

    prompts = ["A sunny beach", "A snowy mountain"]

    images, messages = pipe.generate(prompts)

    for i, img in enumerate(images):
        img.save(f"test{i + 2}.png")
        msg = (messages[i] > 0).tolist()
        msg = msg2str(msg)
        print(msg)
