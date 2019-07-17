import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn.functional as F

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def show_data(images, labels):
    images = images.numpy()
    labels['mapCityCl'] = labels['mapCityCl'].numpy()
    for j in range(images.shape[0]):
        img = images[j].transpose(1, 2, 0)
        # do un-normalization
        img[:, :, 0] = 256 * (img[:, :, 0] * 0.229 + 0.485)
        img[:, :, 1] = 256 * (img[:, :, 1] * 0.224 + 0.456)
        img[:, :, 2] = 256 * (img[:, :, 2] * 0.225 + 0.406)
        img[img > 255] = 255
        img = img.astype(np.uint8)

        lab = labels['mapCityCl'][j].astype(np.uint8)

        img = img / 5 * 4
        img[:,:,0] = img[:, :, 0] + lab * 50

        plt.imshow(img)
        plt.show()

def getTops(pred):
    #p = F.softmax(pred, dim = 1)
    #tops = torch.topk(pred, 1, dim=1)
    #tops = tops[1][0][0].cpu().data.numpy().astype(np.uint8)
    tops = pred.data.max(1)[1].cpu().numpy()
    tops = tops[0].astype(np.uint8)
    #tops[tops==34] = 0
    return tops

def get_train_labels(labels):
    label_map = {label[1]: label[2] if label[2] != 255 else 19 for label in labels }
    label_map[-1] = 19
    return label_map

