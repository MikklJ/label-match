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
#import loader.bdd_utils
import loader.cropped_dataset as cd

from torch.autograd import Variable
from torch.utils import data
# custom stuff
from models import get_model
from loader import get_loader
from tqdm import tqdm
from utils._utils import get_config, unnormalize
from utils.train_utils import *
from tensorboardX import SummaryWriter
from loader.bdd_utils import michael_labels 
from time import sleep
"""
Backbone Cellar Training

"""

def train(args):
    config = get_config(args.config)
    device = torch.device("cuda:"+str(config['gpuid']) if config["cuda"] else "cpu")
    #print(torch.cuda.get_device_name(device))
    
    train_label_map = cd.get_train_labels(michael_labels)
    #nlabels = len({k:v for k,v in michael_labels.items()})
    train_config = config['train_dataloaders'][0]
    val_config = config['val_dataloaders'][0]
    
    data_loader = get_loader(train_config['dataset'])
    loader = data_loader(split=train_config['split'], root=train_config['root'], img_size=train_config['img_size'], alterations=train_config['alterations'], crop=train_config['crop'], crop_size=train_config['crop_size']) #train_label_map=train_label_map
    trainloader = data.DataLoader(loader,  batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=train_config['shuffle'])
    
    if config['val_epoch'] > 0:
        data_loader = get_loader(val_config['dataset'])
        loader2 = data_loader(split=val_config['split'], root=val_config['root'], img_size=val_config['img_size'], alterations=val_config['alterations']) #train_label_map=train_label_map
        valloader = data.DataLoader(loader2, batch_size=val_config['batch_size'], num_workers=val_config['num_workers'], shuffle=val_config['shuffle'])

    # Get # of labels in Cityscapes dataset to create the channels
    #nlabels = len(labels.labels)
    model = get_model(config['arch'])
    model = model.to(device)
    #evaluator = CityscapesEvaluator(nlabels)

    if config['opt'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['l_rate'], momentum=config['momentum'], weight_decay=config['w_decay']) 
        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    elif config['opt'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['l_rate'], weight_decay=config['w_decay'])

    base_model = args.config.split("/")[-1].split(".")[0].split("-")
    base_model.append(str(config['l_rate']))
    current_time = strftime("%Y-%m-%d-%H-%M", time.localtime())
    outdir = "outputs/train/" + current_time + "/" + "_".join(base_model)
    val_outdir = outdir.replace("train", "test")
    
    # TensorboardX VISUALISATION
    writer = SummaryWriter()
    print("Open visualization at localhost:6006")
    #colors = label_mapper(labels.labels)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    loss_alg = torch.nn.BCELoss()
        
    for epoch in range(config['n_epoch']):
        start = time.time()
        
        for i, (image_1, image_2, label_1, label_2) in enumerate(trainloader):
            
            
            image_1 = image_1.to(device)
            image_2 = image_2.to(device)
            label_1 = label_1.to(device)
            label_2 = label_2.to(device)
            
            iter = len(trainloader)*epoch + i
            
            optimizer.zero_grad()
            #print(label_1.shape)
            distance_hat = model(image_1, image_2)
            loss = 0.0
            if torch.all(torch.eq(label_1, label_2)):
                loss += F.binary_cross_entropy_with_logits(distance_hat, torch.tensor([1.0]).to(device)) #,loss_weights[nclass])
            else:
                loss += F.binary_cross_entropy_with_logits(distance_hat, torch.tensor([0.0]).to(device)) #,loss_weights[nclass])
            
            #sleep(1)
            loss.backward()
            optimizer.step()

            if iter % config['viz_loss_n_iter'] == 0 and iter != 0:
                # Update loss
                writer.add_scalar('Loss', loss.item(), iter)

            if iter % config['viz_image_n_iter'] == 0:
                # Update images
                writer.add_image('Image_1', unnormalize(image_1.cpu().data.numpy()).squeeze(), iter)
                writer.add_image('Image_2', unnormalize(image_2.cpu().data.numpy()).squeeze(), iter)

            if (iter + 1) % config['viz_console_n_iter'] == 0:
                end = time.time()
                print("Epoch [%d/%d] Iter %d Loss: %.4f" % (epoch+1, config['n_epoch'], iter, loss.item()), end - start)
                start = time.time()
        
        ## EVALUATION
        """
        if config['val_epoch'] > 0 and ((epoch+1) % config['val_epoch'] == 0):
            # Transfer evaluation functions to a different file and import
            model.eval()
            for i, (image, vlabel) in tqdm(enumerate(valloader)):
                image = image.to(device)
                label_map = vlabel['mapCityCl'][0]
                label_id = vlabel['label_id'][0]
                prediction = model(image)
                evaluator.toPNG(prediction, label_map.cpu().data.numpy(), label_id, val_outdir)
    
            score = evaluator.run_cityscapes_evaluator(val_outdir)
            evaluator.delete(val_outdir)
            print("Validation IOU score: " + str(score*100), "Epoch: "+str(epoch+1))
            writer.add_scalar('Validation IOU score', score*100, (epoch+1))
            model.train()
        """

        if (epoch+1) % config['save_model_n_epoch'] == 0 and ( epoch+1 != config['n_epoch']):
            save_model_path = outdir+"/models"

            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)

            PATH = os.path.join(save_model_path, "{}.pt".format(epoch+1))
            torch.save(model.state_dict(), PATH)

    PATH = os.path.join(save_model_path, "final.pt")
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('--config', nargs='?', type=str, default='config/train/label-match.yaml', help='Specify yaml config file to use')  
    args = parser.parse_args()
    train(args)