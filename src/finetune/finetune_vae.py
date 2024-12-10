"""
This code implement the watermarking method proposed by the papaer

The Stable Signature: Rooting Watermarks in Laten tDiffusion Models (ICCV 2023)

and is modified based on https://github.com/facebookresearch/stable_signature 
"""

import os
import sys
sys.path.append(os.getcwd())
import random
import numpy as np
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from src.utils.param_utils import  get_params, parse_optim_params, adjust_learning_rate
from src.utils import data_utils
from src.loss.loss_provider import LossProvider
from src.utils.log_utils import MetricLogger, OutputWriter

class Trainer():

    """
    Trainer class for fine-tuning VAE models in Stable Diffusion.

    This class is responsible for handling the entire fine-tuning workflow, including:
    - Loading the pre-trained VAE model.
    - Generating and managing watermark keys.
    - Initializing and freezing parts of the model as required.
    - Preparing datasets and dataloaders for training and validation.
    - Configuring optimizers and learning rate schedulers.
    - Defining and applying custom loss functions for training.
    - Logging training and validation statistics, including saving fine-tuned models.

    Methods:
        - `_generate_key`: Generates random watermark keys based on the specified number of keys and bit length.
        - `_seed_all`: Sets the random seed for reproducibility.
        - `_build_finetuned_vae`: Creates fine-tuned VAE instances, ensuring only the decoder is trainable.
        - `_freeze`: Freezes the pre-trained VAE to prevent updates during training.
        - `_load_msg_decoder`: Loads the message decoder for watermark extraction.
        - `_get_dataloader`: Prepares training and validation dataloaders from the specified dataset.
        - `_build_optimizer`: Builds optimizers for training, configured for fine-tuning specific VAE decoders.
        - `_loss_fn`: Configures the loss functions for image and watermark reconstruction.
        - `_train_per_key`: Handles the training process for a single watermark key.
        - `_evaluate`: Evaluates the fine-tuned VAE decoder using various robustness tests.
        - `_save`: Saves the fine-tuned model, training statistics, and watermark keys.
        - `train`: Orchestrates the training process for all watermark keys.

    Attributes:
        - `vae` (AutoencoderKL): The pre-trained VAE model used as a base for fine-tuning.
        - `finetuned_vaes` (list[AutoencoderKL]): A list of fine-tuned VAE models, one for each watermark key.
        - `params` (OmegaConf): Configuration parameters for training.
        - `output_dir` (str): Directory for saving logs and outputs.
        - `model_dir` (str): Directory for saving fine-tuned model checkpoints.
        - `watermark_key` (list[list[int]]): A list of binary watermark keys.
        - `msg_decoder` (nn.Module): The message decoder model for extracting watermarks.
        - `train_loader`, `val_loader` (DataLoader): Data loaders for training and validation datasets.
        - `optimizers` (list[torch.optim.Optimizer]): A list of optimizers for each fine-tuned VAE decoder.
        - `loss_w`, `loss_i`: Custom loss functions for watermark and image reconstruction, respectively.
        - `device` (str): The device (CPU or GPU) used for computation.
    """


    def __init__(self, params: OmegaConf) -> None:

        self.params = params

        self.output_dir = os.path.join(os.getcwd(), params.output_dir)
        self.model_dir = os.path.join(os.getcwd(), params.model_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.log_file = os.path.join(self.output_dir, params.log_file)
        self.writer = OutputWriter(self.log_file)
        #generate key before sed seed
        self.watermark_key = self._generate_key(params.num_keys, params.num_bits)
        #set seed
        self._seed_all(params.seed)

        #Initialize vae decoder and msg_decoder
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder = "vae") #vae is the same for different versions of stable diffusion
        self.finetuned_vaes = self._build_finetuned_vae(params.num_keys)   ##requires_grad = True
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
        generate watermark keys before set the random seed.
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

    def _build_finetuned_vae(self, num_keys: int):

        finetuned_vaes = []
        for _ in range(num_keys):
            vae = deepcopy(self.vae)
            for vae_params in vae.parameters():
                vae_params.requires_grad = False
            for vae_decoder_params in vae.decoder.parameters():
                vae_decoder_params.requires_grad = True
            finetuned_vaes.append(vae)
        return finetuned_vaes
    
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
            return getattr(torch.optim, optimizer)(self.finetuned_vaes[i].parameters(), **optim_params)
        raise ValueError(f'Unknown optimizer "{optimizer}", choose among {str(torch_optimizers)}')
    
    def _loss_fn(self, params):
        
        """
        get loss function, the loss function and weights copy from https://github.com/SteffenCzolbe/PerceptualSimilarity
        """

        print(f'>>> Creating losses')
        print(f'Losses: {params.loss_w} and {params.loss_i}')
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
        for vae in self.finetuned_vaes:
            vae.to(device)
        self.msg_decoder.to(device)
        
    def _train_per_key(
                       self,
                       params,
                       key: list[int],
                       vae: AutoencoderKL,
                       optimizer: torch.optim.Optimizer, 
                       vqgan_to_imnet: transforms) -> dict:

        """
        fine_tune vae decoder for one watermark key
        """
        key = data_utils.list_to_torch(key)
        #header = 'Train'
        metric_logger = MetricLogger(delimiter = '\n', window_size = params.log_freq)
        vae.train()
        for step, imgs in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            keys = key.repeat(imgs.shape[0], 1).to(self.device)
        
            adjust_learning_rate(optimizer, step, params.steps, params.warmup_steps, params.lr)
            # encode images
            imgs_z = self.vae.encode(imgs).latent_dist.mode() # b c h w -> b z h/f w/f

            # decode latents with original and finetuned decoder
            imgs_d0 = self.vae.decode(imgs_z).sample # b z h/f w/f -> b c h w
            imgs_w = vae.decode(imgs_z).sample # b z h/f w/f -> b c h w

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
                "loss": loss.item(),
                "loss_w": lossw.item(),
                "loss_i": lossi.item(),
                #"psnr": data_utils.psnr(imgs_w, imgs_d0).mean().item(),
                # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
                "bit_acc_avg": torch.mean(bit_accs).item(),
                "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
                #"lr": optimizer.param_groups[0]["lr"],
                }
            for name, meter in log_stats.items():
                metric_logger.update(**{name: meter})
            if (step + 1) % params.log_freq == 0:
                print(f'Step {step + 1} | {params.steps}')
                for name, meter in metric_logger.meters.items():
                    print(f"    {name}: {meter.avg}")
        
        print(f"Final train stats: the value in () indicates the averaged stats")
        print(metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}        

    @torch.no_grad()
    def _evaluate(
                self,
                key: list[int],
                vae: AutoencoderKL,
                vqgan_to_imnet: transforms
                ) -> dict:
        
        metric_logger = MetricLogger(delimiter = "\n", fmt = "{global_avg:4f}")
        key = data_utils.list_to_torch(key)
        vae.eval()
        for imgs in tqdm.tqdm(self.val_loader):
        
            imgs = imgs.to(self.device)

            imgs_z = self.vae.encode(imgs).latent_dist.mode() # b c h w -> b z h/f w/f

            imgs_d0 = self.vae.decode(imgs_z).sample # b z h/f w/f -> b c h w
            imgs_w = vae.decode(imgs_z).sample # b z h/f w/f -> b c h w
        
            keys = key.repeat(imgs.shape[0], 1).to(self.device)

            log_stats = {
                #"iteration": step + 1,
                "psnr": data_utils.psnr(imgs_w, imgs_d0).mean().item(),
                # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            }
            imgs_w, imgs_d0 = torch.clamp(imgs_w, -1, 1), torch.clamp(imgs_d0, -1, 1)
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
                #'jpeg_80': lambda x: data_utils.jpeg_compress(x, 80),
                #'jpeg_50': lambda x: data_utils.jpeg_compress(x, 50),
            }
            for name, attack in attacks.items():
                imgs_aug = attack(vqgan_to_imnet(imgs_w))
                decoded = self.msg_decoder(imgs_aug) # b c h w -> b k
                diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                word_accs = (bit_accs == 1) # b
                log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
                log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
            for name, meter in log_stats.items():
                metric_logger.update(**{name: meter})
            
        print("Averaged eval stats")
        print(metric_logger)
            
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
    def _save(self, index: int, **kargs) -> None:

        watermark_key = data_utils.list_to_str(self.watermark_key[index])
        model_path = os.path.join(self.model_dir, f"checkpoint_{index:03d}.pth")
        metadata = {'key': watermark_key,
                    'saved_path': model_path,
                    'train_stats': {**{f'train_{k}': v for k, v in kargs['train_stats'].items()}},
                    'val_stats': {**{f'val_{k}': v for k, v in kargs['val_stats'].items()}},
                    }
        log_metadata = {f'Finetuned VAE {index + 1}': metadata}
        torch.save(self.finetuned_vaes[index].state_dict(), model_path)
        self.writer.write_dict(log_metadata)
    
    def train(self) -> None:
        
        for i in range(len(self.watermark_key)):
            print(f'>>> Training for {i + 1}-th Watermark Key')
            train_stats =  self._train_per_key(self.params, self.watermark_key[i], self.finetuned_vaes[i],
                                                self.optimizers[i], data_utils.vqgan_to_imnet())
            print(f'>>> Evaluation for {i + 1}-th Watermark Key')
            val_stats = self._evaluate(self.watermark_key[i], self.finetuned_vaes[i],
                                                                data_utils.vqgan_to_imnet())
            self._save(i, train_stats = train_stats, val_stats = val_stats)

    @property
    def device(self):

        return "cuda" if torch.cuda.is_available() else "cpu"
    

if __name__ == '__main__':

    """
    
    loader = data_utils.get_dataloader("/hpc2hdd/home/yhuang489/MSCOCO/train2017", 
                            transform = data_utils.vqgan_transform(256), num_imgs = 100, 
                            collate_fn=None)

    model = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder = "vae")
    """
    import warnings
    warnings.filterwarnings("ignore")
    params_path = "src/utils/yamls/finetune_vae.yaml"

    trainer = Trainer(params = get_params(params_path))
    trainer.train()