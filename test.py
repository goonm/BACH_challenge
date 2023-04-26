import os
import cv2
import argparse
import numpy as np

from models import ResNet101
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset, DataLoader

from utils import poly_lr, set_random
from datasets import BACHDataset
from batchgenerators.utilities.file_and_folder_operations import *

from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--img_height", type=int, default=224, help="size of image height")
parser.add_argument("--img_width", type=int, default=224, help="size of image width")
parser.add_argument("--out_size", type=int, default=1)
parser.add_argument("--stage", type=str, default='Test')
opt = parser.parse_args()
print(opt)

bach_test = subfiles('bach_data/test')

test_trs = iaa.Sequential([
    iaa.Resize({"height": opt.img_height, "width": opt.img_width})
])

# dataset & dataloader
test_dataset=BACHDataset(bach_test, transform=test_trs, stage=opt.stage)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#device = torch.device('cuda:'+str(opt.gpu_id))
device = torch.device('cpu')


########################### First model #################
model_First = ResNet101(out_size=opt.out_size)
model_First = model_First.to(device)

checkpoint = torch.load('weight/model_First.model', map_location=torch.device('cpu'))
curr_state_dict_keys = list(model_First.state_dict().keys())
new_state_dict = OrderedDict()
for k, value in checkpoint['state_dict'].items():
    key = k
    if key not in curr_state_dict_keys and key.startswith('module.'):
        key = key[7:]
    new_state_dict[key] = value
model_First.load_state_dict(new_state_dict)
model_First=model_First.eval()

with torch.no_grad():
    for data in test_loader: 
        img = data['image'].to(device)
        out = model_First(img)
        pred_First = torch.round(torch.sigmoid(out.squeeze()))
#############################################################

########################### Second model #################
model_Second = ResNet101(out_size=opt.out_size)
model_Second = model_Second.to(device)

checkpoint = torch.load('weight/model_Second.model', map_location=torch.device('cpu'))
curr_state_dict_keys = list(model_Second.state_dict().keys())
new_state_dict = OrderedDict()
for k, value in checkpoint['state_dict'].items():
    key = k
    if key not in curr_state_dict_keys and key.startswith('module.'):
        key = key[7:]
    new_state_dict[key] = value
model_Second.load_state_dict(new_state_dict)
model_Second=model_Second.eval()

with torch.no_grad():
    for data in test_loader: 
        img = data['image'].to(device)
        out = model_Second(img)
        pred_Second = torch.round(torch.sigmoid(out.squeeze()))
#############################################################

########################### Third model #################
model_Third = ResNet101(out_size=opt.out_size)
model_Third = model_Third.to(device)

checkpoint = torch.load('weight/model_Third.model', map_location=torch.device('cpu'))
curr_state_dict_keys = list(model_Third.state_dict().keys())
new_state_dict = OrderedDict()
for k, value in checkpoint['state_dict'].items():
    key = k
    if key not in curr_state_dict_keys and key.startswith('module.'):
        key = key[7:]
    new_state_dict[key] = value
model_Third.load_state_dict(new_state_dict)
model_Third=model_Third.eval()

with torch.no_grad():
    for data in test_loader: 
        img = data['image'].to(device)
        out = model_Third(img)
        pred_Third = torch.round(torch.sigmoid(out.squeeze()))
#############################################################

if pred_First.item()==0:
    print('prediction : Normal')
else:
    if pred_Second.item()==0:
        print('prediction : Benign')
    else:
        if pred_Third.item()==0:
            print('prediction : In situ')
        else:
            print('prediction : Invasive')


