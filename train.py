import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MemoryImageDataset, prepare_data, EasyTransforms


class DataModule(pl.LightningDataModule):
    def __init__(self, images, targets, args: argparse.Namespace):
        super()__init__()
        self.images = images
        self.targets = targets
        self.args = args
        self.train_images, self.val_images, self.train_targets, self.val_targets = None, None, None, None
        self.train, self.val = None, None

    def prepare_data(self):
        self.train_images, self.val_images, self.train_targets, self.val_targets = \
            train_test_split(self.images, self.targets, test_size=self.args.test_size)

    def setup(self, stage=None):
        self.train = MemoryImageDataset(self.train_images, self.train_targets, transform=EasyTransforms.train)
        self.val = MemoryImageDataset(self.val_images, self.val_targets, transform=EasyTransforms.val)

    def train_loader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MyDataModule")
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--test_size', type=float, default=0.3)
        parser.add_argument('--num_workers', type=int, default=16)
        return parent_parser


















        
        