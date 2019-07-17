import os
import sys
import time
import matplotlib as mpl
# mpl.use('Agg') # avoid using gtk
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
# custom stuff
from models import get_model
from loader import get_loader
from tqdm import tqdm
from utils._utils import get_config, get_train_labels
from utils.evaluator import CityscapesEvaluator
from cityscapesscripts.helpers import labels

def eval(args):
    config = get_config(args.config)
    device = torch.device("cuda:"+str(config['gpuid']) if config["cuda"] else "cpu")

    #train_label_map = get_train_labels(labels.labels)
    #nlabels = len({k:v for k,v in train_label_map.items() if v != 19}) + 1

    val_config = config['val_dataloaders'][0]
    data_loader = get_loader(val_config['dataset'])
    loader = data_loader(split=val_config['split'], root=val_config['root'], img_size=val_config['img_size'], alterations=val_config['alterations']) # train_label_map=train_label_map
    valloader = data.DataLoader(loader, batch_size=val_config['batch_size'], num_workers=val_config['num_workers'], shuffle=val_config['shuffle'])

    base_model = args.config.split("/")[-1].split(".")[0].split("-")
    base_model.append(str(config['l_rate']))
    PATH = "outputs/train/" + args.date + "/" + "_".join(base_model) + "/models/final.pt"
    
    nlabels = len(labels.labels)
    model = get_model(config['arch'], nlabels).to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    evaluator = CityscapesEvaluator(nlabels)
    outdir = "outputs/test/" + args.date + "/"+"_".join(base_model)

    for i, (image, vlabel) in tqdm(enumerate(valloader)):
        image = image.to(device)
        
        label_map = vlabel['mapCityCl'][0]
        label_id = vlabel['label_id'][0]
        # Generate pngs from numpy array and save for evaluation
        prediction = model(image)
        evaluator.toPNG(prediction, label_map.cpu().data.numpy(), label_id, outdir)
        
    score = evaluator.run_cityscapes_evaluator(outdir)
    evaluator.delete(outdir)
    print("avg IOU: " + str(score*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation routine')
    parser.add_argument('--config', nargs='?', type=str, default='config/train/resnet18-cityscapes.yaml', help='Specify yaml config file to use') 
    parser.add_argument('--date', nargs='?', type=str, default='2018-12-18-16-33' , help='Specify the date for your model in this format: Year-Month-Day-Hour-Minute with numbers and hypen between them')
    args = parser.parse_args()
    eval(args)