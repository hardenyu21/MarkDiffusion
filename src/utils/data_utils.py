import os
import numpy as np
import torch
from typing import List, Any
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import is_image_file, default_loader
import json
import functools

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

def default_transform():

    return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def img_transform(img_size: int):

    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
    unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                           std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean
    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_img])

def vqgan_transform(img_size: int):

    normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                           std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
    unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], 
                                             std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5

    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_vqgan
                ])

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