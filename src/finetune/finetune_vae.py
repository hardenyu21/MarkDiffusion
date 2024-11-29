"""The code is modified based on https://github.com/facebookresearch/stable_signature """

#TODO: save the output
import os
import sys
sys.path.append(os.getcwd())
import json
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
from omegaconf import OmegaConf
from src.utils.param_utils import parse_optim_params, adjust_learning_rate
from src.utils import data_utils
from src.loss.loss_provider import LossProvider
from src.utils.log_utils import MetricLogger


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
        self.watermark_key = self._generate_key(params.num_keys, params.num_bits)
        #set seed
        self._seed_all(params.seed)

        #Initialize vae decoder and msg_decoder
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder = "vae") #vae is the same for different versions of stable diffusion
        self.finetuned_vae_decoders = self._build_finetuned_vae_decoder(params.num_keys)   ##requires_grad = True
        self._freeze()
        self.msg_decoder = self._load_msg_decoder(params)
        self._to(self.device)
        
        #dataset
        self.train_loader, self.val_loader = self._get_dataloader(params)
        
        ##optimizer and learning scheduler
        self.optimizers = []
        for i in range(params.num_keys):
            optimizer = self._build_optimizer(params, i)
            self.optimizers.append(optimizer)
                                   
        ##loss function
        self.loss_w, self.loss_i = self._loss_fn(params)
    
    def _generate_key(self, num_keys: int, num_bits: int) -> list[list]:

        """
        generate watermark keys based before set the random seed.
        """        
        watermark_keys = []
        for _ in range(num_keys):
            bits = random.choices([0, 1], k = num_bits)
        watermark_keys.append(bits)
        return watermark_keys
    
    def _seed_all(self, seed: int) -> None:
        "Set the random seed for reproducibility"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    def _build_finetuned_vae_decoder(self, num_keys: int):

        finetuned_vae_decoders = []
        for _ in range(num_keys):
            decoder = deepcopy(self.vae.decoder)
            finetuned_vae_decoders.append(decoder)
        return finetuned_vae_decoders
    
    def _freeze(self):

        for model_params in self.vae.parameters():
            model_params.requires_grad = False
        self.vae.eval()

    def _load_msg_decoder(self, params) -> nn.Module:

        """load msg decoder from checkpoint"""
        ckpt_path = params.msg_decoder_path
        try:
            msg_decoder = torch.jit.load(ckpt_path)
        except:
            raise KeyError(f"No checkpoint found in {ckpt_path}")
        for msg_decoder_params in msg_decoder.parameters():
            msg_decoder_params.requires_grad = False
        msg_decoder.eval()

        return msg_decoder
        
    def _get_dataloader(self, params) -> Tuple[DataLoader, DataLoader]:

        transform = data_utils.vqgan_transform(params.img_size)
        num_train = params.steps * params.batch_size
        train_loader = data_utils.get_dataloader(params.train_dir, transform = transform, 
                                      num_imgs = num_train, batch_size = params.batch_size,
                                      shuffle = True, collate_fn = None)
        val_loader = data_utils.get_dataloader(params.val_dir, transform = transform, 
                                    num_imgs = params.num_val, batch_size = params.batch_size,
                                    shuffle = False, collate_fn = None)
        
        return train_loader, val_loader

    def _build_optimizer(self, params, i: int) -> torch.optim.Optimizer:
        
        optimizer = params.optimizer
        optim_params = parse_optim_params(params)

        torch_optimizers = sorted(name for name in torch.optim.__dict__
            if name[0].isupper() and not name.startswith("__")
            and callable(torch.optim.__dict__[name]))
        if hasattr(torch.optim, optimizer):
            return getattr(torch.optim, optimizer)(self.finetuned_vae_decoders[i].parameters(), **optim_params)
        raise ValueError(f'Unknown optimizer "{optimizer}", choose among {str(torch_optimizers)}')
    
    def _loss_fn(self, params):
        
        """
        get loss function, the loss function and weights copy from https://github.com/SteffenCzolbe/PerceptualSimilarity
        """

        print(f'>>> Creating losses...')
        print(f'Losses: {params.loss_w} and {params.loss_i}...')
        if params.loss_w == 'mse':        
            loss_w = lambda decoded, keys, temp = 10.0: torch.mean((decoded * temp - (2 * keys- 1))**2) # b k - b k
        elif params.loss_w == 'bce':
            loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')
        else:
            raise NotImplementedError
    
        if params.loss_i == 'mse':
            loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
        elif params.loss_i == 'watson-dft':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('Watson-DFT', colorspace = 'RGB', pretrained = True, reduction = 'sum')
            loss_percep = loss_percep.to(self.device)
            loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif params.loss_i == 'watson-vgg':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('Watson-VGG', colorspace = 'RGB', pretrained = True, reduction = 'sum')
            loss_percep = loss_percep.to(self.device)
            loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif params.loss_i == 'ssim':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
            loss_percep = loss_percep.to(self.device)
            loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        else:
            raise NotImplementedError
        
        return loss_w, loss_i
    
    def _to(self, device: str):

        self.vae.to(device)
        for decoder in self.finetuned_vae_decoders:
            decoder.to(device)
        self.msg_decoder.to(device)
        
    def _train_per_key(
                       self,
                       params,
                       key: list[int],
                       decoder: nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       vqgan_to_imnet: transforms) -> dict:

        """
        fine_tune vae decoder for one watermark key
        """
        key = data_utils.list_to_torch(key)
        header = 'Train'
        metric_logger = MetricLogger(delimiter="  ")
        decoder.train()
        for step, imgs in enumerate(metric_logger.log_every(self.train_loader, params.log_freq, header)):
            imgs = imgs.to(self.device)
            keys = key.repeat(imgs.shape[0], 1)
        
            adjust_learning_rate(optimizer, step, params.steps, params.warmup_steps, params.lr)
            # encode images
            imgs_z = self.vae.encode(imgs).latent_dist.sample() # b c h w -> b z h/f w/f

            # decode latents with original and finetuned decoder
            imgs_d0 = self.vae.decode(imgs_z).sample # b z h/f w/f -> b c h w
            imgs_w = decoder.decode(imgs_z).sample # b z h/f w/f -> b c h w

            # extract watermark
            decoded = self.msg_decoder(vqgan_to_imnet(imgs_w)) # b c h w -> b k

            # compute loss
            lossw = self.loss_w(decoded, keys)
            lossi = self.loss_i(imgs_w, imgs_d0)
            loss = params.lambda_w * lossw + params.lambda_i * lossi

            # optim step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log stats
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            log_stats = {
                "iteration": step,
                "loss": loss.item(),
                "loss_w": lossw.item(),
                "loss_i": lossi.item(),
                "psnr": data_utils.psnr(imgs_w, imgs_d0).mean().item(),
                # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
                "bit_acc_avg": torch.mean(bit_accs).item(),
                "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
                "lr": optimizer.param_groups[0]["lr"],
                }
            for name, loss in log_stats.items():
                metric_logger.update(**{name:loss})
            if (step + 1) % params.log_freq == 0:
                print(json.dumps(log_stats))
        
        print("Averaged {} stats:".format('train'), metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}        

    @torch.no_grad()
    def evaluate(
                self,
                params,
                key: list[int],
                decoder: nn.Module,
                vqgan_to_imnet: transforms
                ) -> dict:
        
        header = 'Eval'
        metric_logger = MetricLogger(delimiter="  ")
        decoder.eval()
        for step, imgs in enumerate(metric_logger.log_every(self.val_loader, params.log_freq, header)):
        
            imgs = imgs.to(self.device)

            imgs_z = self.vae.encode(imgs).latent_dist.sample() # b c h w -> b z h/f w/f

            imgs_d0 = self.vae.decode(imgs_z).sample # b z h/f w/f -> b c h w
            imgs_w = decoder.decode(imgs_z).sample # b z h/f w/f -> b c h w
        
            keys = key.repeat(imgs.shape[0], 1)

            log_stats = {
                "iteration": step + 1,
                "psnr": data_utils.psnr(imgs_w, imgs_d0).mean().item(),
                # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            }
            attacks = {
                'none': lambda x: x,
                'crop_01': lambda x: data_utils.center_crop(x, 0.1),
                'crop_05': lambda x: data_utils.center_crop(x, 0.5),
                'rot_25': lambda x: data_utils.rotate(x, 25),
                'rot_90': lambda x: data_utils.rotate(x, 90),
                'resize_03': lambda x: data_utils.resize(x, 0.3),
                'resize_07': lambda x: data_utils.resize(x, 0.7),
                'brightness_1p5': lambda x: data_utils.adjust_brightness(x, 1.5),
                'brightness_2': lambda x: data_utils.adjust_brightness(x, 2),
                'jpeg_80': lambda x: data_utils.jpeg_compress(x, 80),
                'jpeg_50': lambda x: data_utils.jpeg_compress(x, 50),
            }
            for name, attack in attacks.items():
                imgs_aug = attack(vqgan_to_imnet(imgs_w))
                decoded = self.msg_decoder(imgs_aug) # b c h w -> b k
                diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                word_accs = (bit_accs == 1) # b
                log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
                log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
            for name, loss in log_stats.items():
                metric_logger.update(**{name:loss})
                print("Averaged {} stats:".format('eval'), metric_logger)
            
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def train(self) -> None:
        
        for i in range(len(self.watermark_key)):
            train_status =  self._train_per_key(self.params, self.watermark_key[i], self.finetuned_vae_decoders[i],
                                                self.optimizers[i], data_utils.vqgan_to_imnet())
            val_status = self.evaluate(self.params, self.watermark_key[i], self.finetuned_vae_decoders[i],
                                                                    data_utils.vqgan_to_imnet())
    
    @property
    def device(self):

        return "cuda" if torch.cuda.is_available() else "cpu"
    

if __name__ == '__main__':
    
    loader = data_utils.get_dataloader("/hpc2hdd/home/yhuang489/MSCOCO/train2017", 
                            transform = data_utils.vqgan_transform(256), num_imgs = 100, 
                            collate_fn=None)

    model = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder = "vae")
    