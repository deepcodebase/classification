from hpman.m import _

from .mnist import MNIST

def get_dataset(name, split='train'):
    if name.lower() == 'mnist':
        return MNIST(_('data_root', '/data')).get_split(split)
    else:
        raise NotImplementedError