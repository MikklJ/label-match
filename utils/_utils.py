import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn.functional as F

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.starprintswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def load_partial_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = dict(list(pretrained_dict.items())[18:])
    pretrained_dict = OrderedDict([(list(model_dict.items())[i][0], v) for i, (k, v) in enumerate(pretrained_dict.items())])
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    return model.load_state_dict(pretrained_dict)

# Padding utils

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    catted = torch.cat([vec.float(), torch.zeros(*pad_size)], dim=dim)
    return catted


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]  

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def get_img(path):
    img = np.load(path).astype(np.float32).transpose(2, 0, 1)
    return img.astype(np.uint8)

def get_gt_img(path):
    img = np.array(Image.open(path)).transpose(2, 0, 1)
    return img.astype(np.uint8)

def unpack_raw(image):
    # pack Bayer image to 4 channels
    h, w = image.shape
    expanded_image = np.expand_dims(image, axis=0)
    compact = np.concatenate(
        [
            expanded_image[:,0:h:2, 0:w:2],  # Gr
            expanded_image[:,0:h:2, 1:w:2],  # R
            expanded_image[:,1:h:2, 0:w:2],  # B
            expanded_image[:,1:h:2, 1:w:2]   # Gb
        ], axis=0)
    return compact

def unnormalize(img):
    img = img.copy()
    return (img*255.0).astype(np.uint8)[0]

def new_unnormalize(img):
    img = img.copy()
    img = np.clip((img*255.0), 0, 255).astype(np.uint8)[0]
    return img

def unnormalize_to_jpg(img):
    img = img.copy()
    img[:, 0, :, :] = (img[:, 0, :, :] * 0.229 + 0.485)
    img[:, 1, :, :] = (img[:, 1, :, :] * 0.224 + 0.456)
    img[:, 2, :, :] = (img[:, 2, :, :] * 0.225 + 0.406)
    return (img*255.0).astype(np.uint8)[0]

def unnormalize_2(img):
    img = img.copy()
    return (img*1023.0).astype(np.uint8)[0]

def unnormalize_boxes(labels, img_shape):
    boxes = torch.zeros_like(labels[:,1:])
    boxes[:, 0] = img_shape[1]*(labels[:,1] - labels[:,3]/2)
    boxes[:, 1] = img_shape[0]*(labels[:,2] - labels[:,4]/2)
    boxes[:, 2] = img_shape[1]*(labels[:,1] + labels[:,3]/2)
    boxes[:, 3] = img_shape[0]*(labels[:,2] + labels[:,4]/2)
    return boxes

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def flow_to_img(flow):
    flow = flow.copy()[0]
    c, h, w =  flow.shape
    c = 3
    hsv = np.zeros((h, w, c), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 2] = 255
    mag, ang = cv2.cartToPolar(flow[0, ...], flow[1, ...])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb.transpose(2, 0, 1)

class Unnormalize(object):
    """ Revert normalized images to their original rgb values"""
    def __init__(self, normalization):
        self.means = normalization['mean']
        self.stds = normalization['std']

    def __call__(self, img):
        img = img.copy()
        img[:, 0, :, :] = (img[:, 0, :, :] * self.stds[0] + self.means[0])
        img[:, 1, :, :] = (img[:, 1, :, :] * self.stds[1] + self.means[1])
        img[:, 2, :, :] = (img[:, 2, :, :] * self.stds[2] + self.means[2])
        return img[0]

def show_data(images, labels, normalization):
    images = images.numpy()
    labels['clear'] = labels['clear'].numpy()
    for j in range(images.shape[0]):
        img = images[j].transpose(1, 2, 0)
        # do un-normalization
        img[:, :, 0] = 256 * (img[:, :, 0] * normalization['std'][0] + normalization['mean'][0])
        img[:, :, 1] = 256 * (img[:, :, 1] * normalization['std'][1] + normalization['mean'][1])
        img[:, :, 2] = 256 * (img[:, :, 2] * normalization['std'][2] + normalization['mean'][2])
        img[img > 255] = 255
        img = img.astype(np.uint8)

        lab = labels['clear'][j].transpose(1, 2, 0).astype(np.uint8)

        img = img / 5 * 4
        lab = lab / 5 * 4
        #img[:,:,0] = img[:, :, 0] + lab * 50
        plt.imshow(img)
        plt.imshow(lab)
        plt.show()