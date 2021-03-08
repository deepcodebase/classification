from pathlib import Path

from torchvision import datasets, transforms


class MNIST:

    def __init__(self, location):
        self.name = 'mnist'
        self.data_dir = Path(location)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    def get_split(self, split='train'):
        return datasets.MNIST(
            self.data_dir, train=split == 'train', download=True,
            transform=self.transform)