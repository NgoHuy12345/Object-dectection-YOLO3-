import zipfile
import gzip
import os
import io
import pickle
from dataclasses import dataclass

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

def prepare_data(DATA_FILE='coco128.zip'):
    DATA_ROOT = '/content/drive/My Drive/YOLOv3/data/{}'
    DATA_PATH = DATA_ROOT.format(DATA_FILE)
    train_zip = zipfile.ZipFile(DATA_PATH)
    file_names_image = [name for name in train_zip.namelist() if 'jpg' in name]
    file_names_label = [name for name in train_zip.namelist() if 'txt' in name and 'README' not in name]
    images = [[name, Image.open(train_zip.open(name))] for name in file_names_image]
    labels = [[name, train_zip.open(name).read().decode('UTF-8')] for name in file_names_label]
    images.sort(key=lambda x: x[0])
    labels.sort(key=lambda x: x[0])
    [img[1].load() for img in images]
    images = list(map(lambda x: x[1], images))
    labels = list(map(lambda x: x[1], labels))
    labels = list(map(lambda x: x.split('\n'), labels))
    return images, labels    


class MemoryImageDataset(Dataset):
    """
    Dataset for memory images
    """

    def __init__(self, images, targets, transform=None):
        super().__init__()
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = self.targets[idx]
        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]

        return img, y


@dataclass
class EasyTransforms:
    train = A.Compose([
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomCrop(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    val = A.Compose([
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.CenterCrop(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    test = A.Compose([
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.CenterCrop(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(always_apply=True)
    ])