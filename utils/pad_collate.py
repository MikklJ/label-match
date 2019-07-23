import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn.functional as F

class PadCollate: 
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=1):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        
        max_len = max(map(lambda x: x[2].shape[self.dim], batch))
        # pad according to max_len 
        zs = list(map(lambda x: pad_tensor(x[2], pad=max_len, dim=self.dim), batch))
        # stack all
        zs = torch.stack(zs)
        xs = torch.stack(list(map(lambda x: x[0], batch)))
        ys = torch.stack(list(map(lambda x: x[1], batch)))
        return xs, ys, zs

    def __call__(self, batch):
        return self.pad_collate(batch)
