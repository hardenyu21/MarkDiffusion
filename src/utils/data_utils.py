import os
import numpy as np
import torch
from typing import List, Any
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import is_image_file, default_loader
import json
import functools
from augly.image import functional as aug_functional
from torchvision.transforms import functional

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

normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                           std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                           std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
    
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], 
                                             std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5

def default_transform():

    return transforms.Compose([
                transforms.ToTensor(),
                normalize_img])

def img_transform(img_size: int):

    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_img])

def vqgan_transform(img_size: int):

    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_vqgan
                ])

def vqgan_to_imnet():
    
    return transforms.Compose([unnormalize_vqgan, normalize_img])

def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

class COCODataset(Dataset):

    """
    COCO Dataset for both training and validation sets, only contains images
    """

    def __init__(self,
                 data_dir: str,
                 transform: transforms = None) -> None:
        
        if transform:
            self.transform = transform
        self.images = self._get_image_paths(data_dir)

    @functools.lru_cache()
    def _get_image_paths(self, path:str):
        """
        get the image paths
        """
        paths = []
        for current_path, _, files in os.walk(path):
            for filename in files:
                paths.append(os.path.join(current_path, filename))
        return sorted([fn for fn in paths if is_image_file(fn)])

    def __getitem__(self, index: int) -> Any:
        assert 0 <= index < len(self), "invalid index"
        image = default_loader(self.images[index])
        if self.transform:
            image = self.transform(image)
        
        return image

    def __len__(self) -> int:

        return len(self.images)

def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch

def get_dataloader(data_dir: str, 
                   transform: transforms, 
                   num_imgs: int = None, 
                   batch_size: int = 4,
                   num_workers: int = 4,
                   shuffle: bool = False, 
                   collate_fn: Any = collate_fn):
    
    """ Get dataloader"""
    dataset = COCODataset(data_dir, transform=transform)
    if num_imgs is not None:
        assert num_imgs < len(dataset)
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))

    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, 
                      num_workers = num_workers, collate_fn = collate_fn)


if __name__ == '__main__':


   
    loader = get_dataloader("/hpc2hdd/home/yhuang489/MSCOCO/val2017", 
                            transform = img_transform(256), num_imgs = 100, 
                            collate_fn=None)

    for x in loader:
        print(x.shape)
        break