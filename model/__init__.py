from .cnn import BaseNet


def get_model(name):
    if name.lower() == 'basenet':
        return BaseNet()
    else:
        raise NotImplementedError
