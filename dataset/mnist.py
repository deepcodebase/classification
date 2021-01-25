from pathlib import Path

from torchvision import datasets, transforms


class MNIST:

    def __init__(self, data_root):
        self.name = 'mnist'
        self.data_root = Path(data_root)
        self.data_dir = self.data_root / self.name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    def get_split(self, split='train'):
        return datasets.MNIST(
            self.data_dir, train=split == 'train', download=True,
            transform=self.transform)