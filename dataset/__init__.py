import torch

from .mnist import MNIST
from .imagenet import ImageNet, ImageNetLMDB

def get_dataset(name, location, split='train'):
    if name.lower() == 'mnist':
        return MNIST(location).get_split(split)
    elif name.lower() == 'imagenet':
        return ImageNet(location).get_split(split)
    elif name.lower() == 'imagenet_lmdb':
        return ImageNetLMDB(location).get_split(split)
    else:
        raise NotImplementedError


class Prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target