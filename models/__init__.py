#from models.vgg16s import dilated_vgg16
from models.resnet import *

def _get_model_instance(name):
    """get_model_instance
    :param name:
    """
    return {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }[name]

def get_model(name, nclass):
    model = _get_model_instance(name)(n_classes=nclass)
    model.load_pretrained()
    return model
