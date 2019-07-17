import os, sys
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from loader.bdd_utils import *
from PIL import Image
import numpy as np
import json

sys.path.append(".."),
from utils._utils import PadCollate, get_config

def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def get_train_labels(labels):
    label_map = {label[0]: label[2] for label in labels }
    return label_map

class CroppedDataset(data.Dataset):
    def __init__(self, root="/home/ege/datasets/bdd100k", split="train", dataset_size='100k', noisy=False, alterations={}, img_size=(720 , 1280), val_percent=0.2, test_percent=0.1, resize=False, crop=False, crop_size=None):
        self.data = []
        self.img_size = img_size
        self.resize = resize
        self.crop = crop
        self.crop_size = crop_size
        self.resize = alterations['resize']
        self.jitter = alterations['jitter']
        self.normalize = alterations['normalize']
        self.split = split
        self.noisy = noisy
        if self.noisy:
            self.gauss_mean = alterations['gaussian_noise'][0]
            self.gauss_std = alterations['gaussian_noise'][1]
        self.label_map = get_train_labels(bdd_labels)

        if self.split == "train":
            labels_file = root + "/labels/"+ "bdd100k_labels_images_train.json"
            new_split = "train"
        else:
            labels_file = root + "/labels/"+ "bdd100k_labels_images_val.json"
            new_split = "val"
        
        with open(labels_file, 'r') as f:
            labels = json.load(f)

        # standard normalization for the pretrained networks
        if 'train' in self.split:
            self.color_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=self.jitter['brightness'], contrast=self.jitter['contrast'], saturation=self.jitter['saturation'], hue=self.jitter['hue']),
            ])
            self.transforms = transforms.Compose([
                #transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize['mean'], std=self.normalize['std'])
            ])
        else:
            self.transforms = transforms.Compose([
                #transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize['mean'], std=self.normalize['std'])])

        images_folder = root + "/images/" + dataset_size + "/" + new_split
        images_list = [os.path.join(images_folder, image) for image in os.listdir(images_folder)]
        images_list = images_list[:30000] #30000
        for image in images_list:
            img_name = image.split("/")[-1]
            label_list = [label for label in labels if label["name"] == img_name]
            if label_list:
                label_dict = label_list[0]
            else:
                print("missing label for image")
                continue
            bare_json_name = img_name.replace("jpg", "json")
            json_name = root+"/new_labels/"+new_split+"/"+bare_json_name
            if not os.path.isfile(json_name):
                with open(json_name, 'w') as outfile:
                    json.dump(label_dict, outfile)
            self.data.append({
                'image': image,
                'label': json_name
            })

        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):

        noise_img = Image.open(self.data[index]['image'])
        if self.split == "train":
            noise_img = self.color_transforms(noise_img)
        noise_img = np.array(noise_img).astype(np.float32)
        gt_img = np.copy(noise_img)
        with open(self.data[index]['label'], 'r') as f:
            bbs = json.load(f)

        new_bbs = []
        for bb in bbs["labels"]:
            if "box2d" in bb.keys():
                if bb["category"] == "motor":
                    bb["category"] = "motorcycle"
                if bb["category"] == "bike":
                    bb["category"] = "bicycle"
                if self.label_map[bb["category"]] != 255:
                    new_bbs.append(np.array([[self.label_map[bb["category"]], (bb["box2d"]["x2"] + bb["box2d"]["x1"])/2.0, \
                                    (bb["box2d"]["y2"] + bb["box2d"]["y1"])/2.0, \
                                    bb["box2d"]["x2"] - bb["box2d"]["x1"],  \
                                    bb["box2d"]["y2"] - bb["box2d"]["y1"]]]).astype(np.float32))


        annot = np.concatenate(new_bbs, axis=0)

        #if 'train' in self.split and np.random.uniform() > 0.5: # do random flip
        #    img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #    labels['mapCityCl'] = labels['mapCityCl'][:,::-1].copy()
        #    # TODO: need to take care of polynomial form of the labels
        #    if 'polys' in labels:
        #    	raise NotImplementedError

        got_shape = False
        noise_imgs, gt_imgs = [], []

        pad, padded_h, padded_w = None, None, None
        #for noise_img, gt_img in zip([np.load(img_name).astype(np.float32) for img_name in img_paths], [np.load(gt_name).astype(np.float32) for gt_name in gt_paths]):
        if not got_shape:
            h, w, _ = noise_img.shape
            y0 = np.random.randint(h - self.crop_size[0])
            x0 = np.random.randint(w - self.crop_size[1])
            got_shape = True
        if self.crop:
            noise_img = noise_img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            #noise_img = noise_img/255.0
            gt_img = gt_img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            #gt_img = gt_img/255.0
        else:
            diff = np.abs(h-w)
            pad = ((diff//2, diff//2), (0,0), (0, 0)) if h <= w else ((0, 0), (diff//2, diff/2), (0, 0))
            noise_img = np.pad(noise_img, pad, 'constant', constant_values=128) #/255.
            padded_h, padded_w, _ = noise_img.shape
        if self.resize != 1.0:
            noise_img = np.resize(noise_img, (int(noise_img.shape[1] * self.resize), int(noise_img.shape[2] * self.resize), noise_img.shape[0]))
            gt_img = np.resize(gt_img, (int(gt_img.shape[1] * self.resize), int(gt_img.shape[2] * self.resize), gt_img.shape[0]))
        noise_img = np.transpose(noise_img, (2, 0, 1))
        gt_img = np.transpose(gt_img, (2, 0, 1))
        #noise_img = torch.from_numpy(padded_img).float()
        #noise_imgs.append(noise_img)
        #gt_imgs.append(gt_img)


        #for annot in bbs:
            # Adjust with added padding

        if self.crop:
            #annot = annot[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            m1 = (x0 < annot[:, 1]) * (y0 < annot[:, 2])
            m2 = ( (x0+self.crop_size[1]) > annot[:, 1]) * ( (y0+self.crop_size[0]) > annot[:, 2])
            mask =  m1 * m2
            #if not mask.any():
            #    continue
                #annotations.append(torch.from_numpy(np.zeros_like(annot)))
                # #print(np.zeros_like(annot))
                
            matching_boxes = annot[mask,:].copy()
            x1 = matching_boxes[:, 1] - (matching_boxes[:, 3]/2)
            y1 = matching_boxes[:, 2] - (matching_boxes[:, 4]/2)
            x2 = matching_boxes[:, 1] + (matching_boxes[:, 3]/2)
            y2 = matching_boxes[:, 2] + (matching_boxes[:, 4]/2)    
            x1 = np.maximum(x1, x0)
            x1 -= x0
            y1 = np.maximum(y1, y0)
            y1 -= y0
            x2 = np.minimum(x2 , x0+self.crop_size[1])
            x2 -= x0
            y2 = np.minimum(y2, y0+self.crop_size[0])
            y2 -= y0
            #ux1 = np.where(x2 < x0+self.crop_size[1], np.where(x1 > x0 , x1, np.where(x1 < x2, x0, 0)), )
            matching_boxes[:, 1] = np.clip(((x1 + x2) / 2) / float(self.crop_size[1]), 0, 0.99)
            matching_boxes[:, 2] = np.clip(((y1 + y2) / 2) / float(self.crop_size[0]), 0, 0.99)
            matching_boxes[:, 3] = (x2 - x1) / float(self.crop_size[1])
            matching_boxes[:, 4] = (y2 - y1) / float(self.crop_size[0])
            annotation = torch.from_numpy(matching_boxes)
        else:
            x1 = annot[:, 1] - (annot[:, 3]/2)
            y1 = annot[:, 2] - (annot[:, 4]/2)
            x2 = annot[:, 1] + (annot[:, 3]/2)
            y2 = annot[:, 2] + (annot[:, 4]/2)
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            annot[:, 1] = np.clip(((x1 + x2) / 2) / float(padded_w), 0, 0.99)
            annot[:, 2] = np.clip(((y1 + y2) / 2) / float(padded_h), 0, 0.99)
            annot[:, 3] = (x2 - x1) / float(padded_w)
            annot[:, 4] = (y2 - y1) / float(padded_h)
            annotation = torch.from_numpy(annot) 
        #annotations = [torch.from_numpy(np.load(annotation_path).copy()) for annotation_path in annotation_paths]
        #return noise_img,annotations
        
        #return torch.from_numpy(np.array(noise_imgs)), torch.from_numpy(np.array(gt_imgs)), annotation
        #print(noise_img.shape)
        if self.noisy:
            noise_img = noise_img/255.0
            noise = np.random.normal(self.gauss_mean, self.gauss_std, noise_img.shape)
            noise_img += noise
            noise_img = np.clip(noise_img, 0, 1)
            noise_img = torch.from_numpy(noise_img).float()
        else:
            noise_img = noise_img/255.0
            noise_img = np.minimum(noise_img, 1.0)
            noise_img = torch.from_numpy(noise_img).float()

        gt_img = gt_img/255.0
        gt_img = np.minimum(gt_img, 1.0)
        gt_img = torch.from_numpy(gt_img).float()
        return self.transforms(noise_img), self.transforms(gt_img), annotation

def load_data(dataset, batch_size, num_workers, split='train', deterministic=False, shuffle=False):
    """
    Load the denoise dataset.
    """


    #idx = dataset.indx
    #sampler = SubsetRandomSampler(idx)

    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                        pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=PadCollate(dim=0))

    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('--config', nargs='?', type=str, default='../configs/recurrent-yolo-bdd.yaml', help='Specify yaml config file to use')
    args = parser.parse_args()
    config = get_config(args.config)
    train_config = config['data'][0]
    loader = BDDDataset(split=train_config['split'], root=train_config['root'], img_size=train_config['img_size'], alterations=train_config['alterations'], crop=train_config['crop'], crop_size=train_config['crop_size'])
    #loader = LoaderWrapper(noisyLoader, batch_size=train_config['batch_size'])
    trainloader = data.DataLoader(loader, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=train_config['shuffle'], collate_fn=PadCollate(dim=0))
    for i, (images, gts, labels) in enumerate(trainloader):
        if i == 10:
            print(images)
            print(gts)
            print(labels)
            break

