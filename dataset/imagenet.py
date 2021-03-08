from pathlib import Path

from torchvision import datasets, transforms

from .folder2lmdb import ImageFolderLMDB


class ImageNet:

    def __init__(self, location):
        self.name = 'imagenet'
        self.data_dir = Path(location)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_split(self, split='train'):
        return datasets.ImageFolder(
            self.data_dir / split, transform=self.transform)


class ImageNetLMDB:

    def __init__(self, location):
        self.name = 'imagenet_lmdb'
        self.data_dir = Path(location)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_split(self, split='train'):
        return ImageFolderLMDB(
            str(self.data_dir / f'{split}.lmdb'), transform=self.transform)