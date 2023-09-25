import logging
import os.path
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LeatherDataset(Dataset):
    def __init__(self, visual_dir: str, tactile_dir: str, masks_dir: str, resize_shape=None):
        if resize_shape is None:
            resize_shape = [256, 256]
        self.visual_dir = Path(visual_dir)
        self.tactile_dir = Path(tactile_dir)
        self.masks_dir = Path(masks_dir)
        self.resize_shape = resize_shape

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.resize_shape)
        ])

        self.visual_filenames = [splitext(file)[0] for file in listdir(self.visual_dir)]
        logging.info(f'Creating dataset with {len(self.visual_filenames)} examples')

    def __len__(self):
        return len(self.visual_filenames)

    def __getitem__(self, idx):
        visual_filename = self.visual_filenames[idx]
        tactile_filename = visual_filename + 'z'
        visual_path = os.path.join(self.visual_dir, visual_filename + '.jpg')
        tactile_path = os.path.join(self.tactile_dir, tactile_filename + '.jpg')
        mask_path = os.path.join(self.masks_dir, visual_filename + '.png')
        visual = Image.open(visual_path).convert('RGB')
        tactile = Image.open(tactile_path).convert('L')
        mask_path = Image.open(mask_path)
        visual = self.img_transform(visual)
        tactile = self.img_transform(tactile)
        mask = self.mask_transform(mask_path)

        return {'visual': visual, 'tactile': tactile, 'mask': np.array(mask)}
