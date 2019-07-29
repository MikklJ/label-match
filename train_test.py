import os
import sys
import time
from time import strftime
import matplotlib as mpl
# mpl.use('Agg') # avoid using gtk
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
# custom stuff
from models import get_model
from loader import get_loader
from tqdm import tqdm
from utils._utils import get_config
#from utils.evaluator import CityscapesEvaluator
#from cityscapesscripts.helpers import labels
from utils.train_utils import *
from tensorboardX import SummaryWriter
import loader.cropped_dataset as cd
from loader.bdd_utils import michael_labels 

def train(args):
    # Get config from label-match.yaml
    config = get_config(args.config)
    #print(config)
    
    # Get device info
    device = torch.device("cuda:"+str(config['gpuid']) if config["cuda"] else "cpu")
    #print("CPU/GPU:",torch.cuda.get_device_name(0))
    
    # Get label mapping
    train_label_map = cd.get_train_labels(michael_labels)
    #print(train_label_map)
    
    # Get config for training and validation runs
    train_config = config['train_dataloaders'][0]
    val_config = config['val_dataloaders'][0]
    
    data_loader = get_loader(train_config['dataset'])
    
    loader = data_loader(split=train_config['split'], root=train_config['root'], img_size=train_config['img_size'], alterations=train_config['alterations']) 
    
    trainloader = data.DataLoader(loader,  batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=train_config['shuffle'])
    
    # Create validation loader
    if config['val_epoch'] > 0:
        data_loader = get_loader(val_config['dataset'])
        loader2 = data_loader(split=val_config['split'], root=val_config['root'], img_size=val_config['img_size'], alterations=val_config['alterations']) 
        
        valloader = data.DataLoader(loader2, batch_size=val_config['batch_size'], num_workers=val_config['num_workers'], shuffle=val_config['shuffle'])
    
    model = get_model(config['arch'], train_label_map)
    model = model.to(device)
    
    # Set up optimizer
    if config['opt'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['l_rate'], momentum=config['momentum'], weight_decay=config['w_decay']) 
        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    elif config['opt'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['l_rate'], weight_decay=config['w_decay'])
    
    # Training
    for epoch in range(config['n_epoch']):
        start = time.time()
        for i, (images, dlabels) in enumerate(trainloader):
            images = images.to(device)
            tlabels = dlabels['mapCityCl'].to(device)
            iter = len(trainloader)*epoch + i
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.0
            # Find cost
            loss += cross_entropy2d(outputs, tlabels) #,loss_weights[nclass])
            loss.backward()
            optimizer.step()

            if iter % config['viz_loss_n_iter'] == 0 and iter != 0:
                # Update loss
                writer.add_scalar('Loss', loss.item(), iter)

            if iter % config['viz_image_n_iter'] == 0:
                # Update images
                writer.add_image('Image', unnormalize(images[0:1].cpu().data.numpy()), iter)
                writer.add_image('GT', gt_mapper(tlabels[0:1], colors), iter)
                writer.add_image('Prediction', pred_mapper(outputs[0:1], colors), iter)

            if (iter + 1) % config['viz_console_n_iter'] == 0:
                end = time.time()
                print("Epoch [%d/%d] Iter %d Loss: %.4f" % (epoch+1, config['n_epoch'], iter, loss.item()), end - start)
                start = time.time()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('--config', nargs='?', type=str, default='config/train/label-match.yaml', help='Specify yaml config file to use')  
    args = parser.parse_args()
    train(args)