import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import weights_init
import torchvision.models as models



class Siamese(nn.Module):
    def __init__(self, nlayers=5, n_hidden=128, learned_billinear=False):
        super(Siamese, self).__init__()
