import os
import numpy as np
import torch
import torch.nn.functional as F
#from utils._utils import getTops

def save_loss_path(outdir, arch, opt, l_rate, epoch, n_epoch, iter, loss):
    save_loss_path = outdir+"/losses"
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path)
    save_loss = save_loss_path + "/{}.txt".format(epoch)
    with open(save_loss,'a') as f:
       loss_txt = "Epoch [%d/%d] Iter %d Loss: %.4f" % (epoch+1, n_epoch, iter, loss) + "\n"
       f.write(loss_txt)

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim = 1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.nll_loss(log_p, target, weight=weight, reduction="sum")
    if size_average:
        loss /= target.size(0)
        # loss /= mask.data.sum()
    return loss

"""
def unnormalize(img):
    img = img.copy()
    img[:, 0, :, :] = (img[:, 0, :, :] * 0.229 + 0.485)
    img[:, 1, :, :] = (img[:, 1, :, :] * 0.224 + 0.456)
    img[:, 2, :, :] = (img[:, 2, :, :] * 0.225 + 0.406)
    return img
"""
def label_mapper(labels):
    return {label[1]: np.array(label[7]) for label in labels}

def filtered_label_mapper(labels):
    mapper = {label[2]:np.array(label[7]) for label in labels if (label[2] != 255 and label[2] != -1) }
    mapper[19] = np.array((0, 0, 0))
    return mapper
        
def color_mapper(input, colors):
    out = np.zeros((input.shape[0], 3, input.shape[2], input.shape[3]), dtype=np.uint8).reshape((-1, 3))
    temp = input.transpose((0, 2, 3, 1)).reshape((-1,))
    for i in range(len(colors)-1):
        out[temp == i,:] = colors[i]
    out = out.reshape((input.shape[0], input.shape[2], input.shape[3], 3)).transpose((0, 3, 1, 2))
    return out

def gt_mapper(input, colors):
    input = input.cpu().data.numpy().astype(np.float32).reshape((-1, 1, input.shape[1], input.shape[2]))
    return color_mapper(input, colors)

def pred_mapper(pred, colors):
    preds = getTops(pred)
    input = preds.astype(np.float32).reshape((1, 1, preds.shape[0], preds.shape[1]))
    return color_mapper(input, colors)