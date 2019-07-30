#from models.vgg16s import dilated_vgg16
from models.siamese import *

def _get_model_instance(name):
    """get_model_instance
    :param name:
    """
    return {
        'siamese': SiameseNet(SalakhNet)
    }[name]

def get_model(name):
    model = _get_model_instance(name)
    return model
