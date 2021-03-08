import torch.nn.functional as F
from .metrics import Metric, compute_acc


def get_loss(name):
    if name in dir(F):
        return getattr(F, name)
    else:
        raise NotImplementedError