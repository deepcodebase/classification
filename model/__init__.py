from torchvision import models

from .cnn import BaseNet


def get_model(name, pretrained=False):
    if name.lower() == 'basenet':
        return BaseNet()
    if name in dir(models):
        model_func = getattr(models, name)
        return model_func(pretrained=pretrained)
    else:
        raise NotImplementedError
