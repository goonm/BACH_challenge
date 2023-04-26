import os
import cv2
import argparse

import numpy as np
from time import time

from models import ResNet101
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset, DataLoader

from utils import poly_lr, set_random

from datasets import BACHDataset
from batchgenerators.utilities.file_and_folder_operations import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--img_height", type=int, default=224, help="size of image height")
parser.add_argument("--img_width", type=int, default=224, help="size of image width")
parser.add_argument("--out_size", type=int, default=1)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--stage", type=str, default='First')
opt = parser.parse_args()
print(opt)

set_random(opt.random_seed)

# data load
if opt.stage=='First':
    bach_train = subfiles('bach_data/normal_train')+subfiles('bach_data/benign_train')+subfiles('bach_data/insitu_train')+subfiles('bach_data/invasive_train')
    bach_val = subfiles('bach_data/normal_val')+subfiles('bach_data/benign_val')+subfiles('bach_data/insitu_val')+subfiles('bach_data/invasive_val')

if opt.stage=='Second':
    bach_train = subfiles('bach_data/benign_train')+subfiles('bach_data/insitu_train')+subfiles('bach_data/invasive_train')
    bach_val = subfiles('bach_data/benign_val')+subfiles('bach_data/insitu_val')+subfiles('bach_data/invasive_val')

if opt.stage=='Third':
    bach_train = subfiles('bach_data/insitu_train')+subfiles('bach_data/invasive_train')
    bach_val = subfiles('bach_data/insitu_val')+subfiles('bach_data/invasive_val')

# data augmentation
trs = iaa.Sequential([
    iaa.Resize({"height": opt.img_height, "width": opt.img_width}),
    iaa.Rot90([0,1,2,3]),
    iaa.ScaleX((0.8, 1.2)),
    iaa.ScaleY((0.8, 1.2)),
    iaa.Fliplr(0.5),
])
test_trs = iaa.Sequential([
    iaa.Resize({"height": opt.img_height, "width": opt.img_width})
])

# dataset & dataloader
tr_dataset=BACHDataset(bach_train, transform=trs, stage=opt.stage)
val_dataset=BACHDataset(bach_val, transform=test_trs, stage=opt.stage)

tr_loader = DataLoader(tr_dataset, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

# Create weight directories
os.makedirs("weight", exist_ok=True)

# Losses
loss = torch.nn.BCEWithLogitsLoss()
#device = torch.device('cuda:'+opt.gpu_ids)
device = torch.device('cpu')

# model
model = ResNet101(out_size=opt.out_size)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

def train():
    model.train()

    loss_li=[]
    for data in tr_loader:
        optimizer.zero_grad() 

        img = data['image'].to(device)
        label = data['class'].to(device)
        
        out = model(img)
        loss_tr = loss(out.squeeze(), label.float())
        
        loss_tr.backward()
        optimizer.step()
        
        loss_li.append(loss_tr)
        
    return loss_li

def test(loader):
    model.eval()

    loss_total_li=[]
    acc_0_step=[]
    acc_1_step=[]
    
    for data in loader: 
        img = data['image'].to(device)
        label = data['class'].to(device)
        
        out = model(img)
        loss_test = loss(out.squeeze(), label.float())

        pred = torch.round(torch.sigmoid(out.squeeze()))
        #############################################
        up_0 = ((pred==label)&(label==0)).sum().detach().cpu()
        down_0 = (label==0).sum().detach().cpu()
        
        if down_0!=0:
            acc_0 = up_0/down_0
            acc_0_step.append(acc_0)
            
        up_1 = ((pred==label)&(label==1)).sum().detach().cpu()
        down_1 = (label==1).sum().detach().cpu()
        
        if down_1!=0:
            acc_1 = up_1/down_1
            acc_1_step.append(acc_1)

        loss_total_li.append(loss_test.detach().cpu())
        
    return acc_0_step, acc_1_step, loss_total_li

tr_acc_0_li=[]
tr_acc_1_li=[]


val_acc_0_li=[]
val_acc_1_li=[]


tr_loss_li=[]
val_loss_li=[]

val_best=99999
for epoch in range(1, opt.n_epochs):
    t1 = time()
    _ = train()
    optimizer.param_groups[0]['lr'] = poly_lr(epoch, opt.n_epochs, opt.lr, 0.9)

    with torch.no_grad():
        tr_acc_0_step, tr_acc_1_step, tr_loss = test(tr_loader)
        val_acc_0_step, val_acc_1_step, val_loss = test(val_loader)
    
    tr_acc_0_li = np.array(tr_acc_0_step).mean()
    tr_acc_1_li = np.array(tr_acc_1_step).mean()
    
    val_acc_0_li = np.array(val_acc_0_step).mean()
    val_acc_1_li = np.array(val_acc_1_step).mean()
    
    tr_loss_li.append(np.array(tr_loss).mean())
    val_loss_li.append(np.array(val_loss).mean())
    
    print(f'\n Epoch: {epoch:03d}')
    print('Training time for one epoch: %.1f'%(time()-t1))
    print(f'Train loss: {np.array(tr_loss).mean()}, Val loss: {np.array(val_loss).mean()}, val_acc_0_li: {val_acc_0_li}, val_acc_1_li: {val_acc_1_li}')
    
    if np.array(val_loss).mean() < val_best:
        fname = 'weight/model_'+opt.stage+'.model'
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        optimizer_state_dict = optimizer.state_dict()
        save_this = {
            'epoch':epoch,
            'state_dict':state_dict,
            'optimizer_state_dict':optimizer_state_dict,
            'tr_total':tr_loss_li,
            'val_total':val_loss_li,
        }
        torch.save(save_this, fname)
        val_best = np.array(val_loss).mean()
        print('save model epoch :', epoch)