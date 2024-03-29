from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from loader.bdd_utils import *
from PIL import Image

import os, sys
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import json

sys.path.append(".."),
from utils._utils import get_config, unnormalize

def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def get_train_labels(labels):
    label_map = {label[0]: label[2] for label in labels }
    return label_map

"""
Make a training dataset for the siamese neural network

1/2 of the dataset is a pair of images from the same class,
and 1/2 is a pair of images from different classes
"""
class CroppedDataset(data.Dataset):
    def __init__(self, root="/home/ege/datasets", split="train", dataset_size='100k', noisy=False, alterations={}, img_size=(720 , 1280), val_percent=0.2, test_percent=0.1, resize=False, crop=False, crop_size=None):
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
        self.label_map = get_train_labels(michael_labels)

        # Use 2871_labels to get labels
        if self.split == "train":
            labels_file = root + "2871_labels.json"
            new_split = "train"
        else:
            labels_file = root + "2871_labels.json"
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

        images_folder = root + "michael_processed"
        
        # We create two lists to store data from the labels file, one for the paths to the images and one for their labels
        images_list = [key.replace("michael/", "ege/") for key in labels]
        label_list = [label for key, label in labels.items()]
                                   
        # Split the dataset into the first 2000 for training, last ~800 for validation
        if new_split == "train":
            images_list = images_list[:2000]
            label_list = label_list[:2000]
        elif new_split == "val":
            images_list = images_list[2000:]
            label_list = label_list[2000:]
        
        """
        Prepares the dataset to have 2000 "same" pairs and 2000 "different" pairs
        """
        for i in range(len(images_list)):
            # Select a random image to put into a "same" and "diff" pair
            while True:
                rand_index = int(np.random.choice(np.arange(len(images_list)), size=1, replace=True))
                
                # Creates lists of indices within the list of images that have the same/different label
                same_indices = [index for index, image in enumerate(images_list) if label_list[index] == label_list[rand_index]]
                diff_indices = [index for index, image in enumerate(images_list) if label_list[index] != label_list[rand_index]]
                
                if len(same_indices) > 1:
                    break
                    
            # Selects an image of the same class as the random image
            pair_index = np.random.choice(same_indices, size=1, replace=False)
            self.data.append({
                'image_1': images_list[rand_index],
                'image_2': images_list[pair_index[0]],
                'label_1': label_list[rand_index],
                'label_2': label_list[pair_index[0]]
            })
            
            # Selects an image of a different class as the random image
            pair_index_2 = np.random.choice(diff_indices, size=1, replace=False)
            self.data.append({
                'image_1': images_list[rand_index],
                'image_2': images_list[pair_index_2[0]],
                'label_1': label_list[rand_index],
                'label_2': label_list[pair_index_2[0]]
            })
            
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        # Get raw images and labels from image filepaths
        img_1 = Image.open(self.data[index]['image_1'])
        img_2 = Image.open(self.data[index]['image_2'])
        label_1 = self.data[index]["label_1"]
        label_2 = self.data[index]["label_2"]
        
        # Get dimensions of images
        width_1, height_1 = img_1.size
        width_2, height_2 = img_2.size
        
        # Rescales images so that the heights of images are 160 px
        if self.crop:
            img_1 = img_1.resize((int(width_1 * (160 / height_1)), 160), resample=Image.NEAREST, box=None)
            img_2 = img_2.resize((int(width_2 * (160 / height_2)), 160), resample=Image.NEAREST, box=None)
        
        # Get new dimensions of images
        width_1, height_1 = img_1.size
        width_2, height_2 = img_2.size
        
        # Crop images so that the width is a multiple of 32
        if self.crop:
            img_1 = img_1.crop((0, 0, width_1 - width_1 % 32, height_1))
            img_2 = img_2.crop((0, 0, width_2 - width_2 % 32, height_2))
            
        # Resizing of images to be 105x105 for easy loading into the siamese network
        img_1 = img_1.resize((105, 105), resample=Image.NEAREST, box=None)
        img_2 = img_2.resize((105, 105), resample=Image.NEAREST, box=None)

        # Convert images to np arrays
        img_1 = np.array(img_1).astype(np.float32)
        img_2 = np.array(img_2).astype(np.float32)
        
        # Convert image labels to numbers
        try:
            label_1 = [el.id for el in michael_labels if el.name == label_1][0]
            label_2 = [el.id for el in michael_labels if el.name == label_2][0]
        except IndexError:
            print(label_1, label_2)
            exit(1)
        
        # Convert labels to arrays
        label_1 = (np.array([label_1]) == np.arange(50)).astype(np.int32)
        label_2 = (np.array([label_2]) == np.arange(50)).astype(np.int32)
        
        # Transpose image arrays 
        img_1 = np.transpose(img_1, (2, 0, 1))
        img_2 = np.transpose(img_2, (2, 0, 1))

        # Normalize image arrays
        img_1 = img_1/255.0
        img_1 = np.minimum(img_1, 1.0)
        img_2 = img_2/255.0
        img_2 = np.minimum(img_2, 1.0)
        
        # Convert image arrays to tensors
        img_1 = torch.from_numpy(img_1).float()
        img_2 = torch.from_numpy(img_2).float()
        
        # Convert labels to tensors
        label_1 = torch.from_numpy(label_1).float()
        label_2 = torch.from_numpy(label_2).float()
        
        return img_1, img_2, label_1, label_2

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

"""
Run this file as main to test out the data loader
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config file')
    #parser.add_argument('--config', nargs='?', type=str, default='../configs/label-match.yaml', help='Specify yaml config file to use')
    #args = parser.parse_args()
    #config = get_config(args.config)

    config = get_config('../config/train/label-match.yaml')
    train_config = config['train_dataloaders'][0]
    dataset = CroppedDataset(split=train_config['split'], root=train_config['root'], img_size=train_config['img_size'], alterations=train_config['alterations'], crop=train_config['crop'], crop_size=train_config['crop_size'])
    # TODO: debug dataset initialization
    #loader = LoaderWrapper(noisyLoader, batch_size=train_config['batch_size'])
    trainloader = data.DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=train_config['shuffle'])
    
    counter = 0
    
    same_counter = 0
    diff_counter = 0
    
    for i, (image_1, image_2, label_1, label_2) in enumerate(trainloader):
        if i < 10:
            #print(unnormalize(image_1.cpu().data.numpy()).squeeze().shape, ";", label_1)
            #print(unnormalize(image_2.cpu().data.numpy()).squeeze().shape, ";", label_2, "\n")
            new_im = Image.fromarray(unnormalize(image_1.cpu().data.numpy()).squeeze())
            new_im.save('test.png')
            #new_im = Image.fromarray(unnormalize(image_2.cpu().data.numpy()).squeeze())
            #new_im.save('test2.png')
            pass
        
        # Check for same-diff pair balance
        if torch.all(torch.eq(label_1, label_2)):
            same_counter+=1
        elif torch.all(torch.eq(label_1, label_2)) == False:
            diff_counter+=1
        break
        
    print("Same", same_counter)
    print("Diff", diff_counter)
    