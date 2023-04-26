import cv2
import torch
from torch.utils.data import Dataset

class BACHDataset(Dataset):
    def __init__(self, path, transform=None, stage='First'):
        self.path = path
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img = cv2.imread(self.path[idx])
        
        if self.transform:
            img = torch.tensor(self.transform(image=img))
        
        # BACH challenge mean/std
        if 'bach_data' in self.path[idx]:
            r_mean, r_std = 214.87577695210774, 30.14312735507004
            g_mean, g_std = 157.32969636201858, 48.073358360414474
            b_mean, b_std = 182.82091190656027, 42.34797739968601
            
        # BISQUE challenge mean/std
        elif 'bisque_data' in self.path[idx]:
            r_mean, r_std = 221.0088387475225, 29.97674072430344
            g_mean, g_std = 176.92862840473944, 63.750058814331624
            b_mean, b_std = 202.75758687537683, 53.92914829098324   
            
        # BreakHist challenge mean/std
        elif 'breakhis_data' in self.path[idx]:
            r_mean, r_std = 197.02152705054561, 28.770535045481093
            g_mean, g_std = 166.37173975777955, 39.28754332179039
            b_mean, b_std = 204.88393495073086, 29.09631775581356
        
        r = (img[:,:,0]-r_mean)/r_std
        g = (img[:,:,1]-g_mean)/g_std
        b = (img[:,:,2]-b_mean)/b_std
            
            
        img_new = torch.cat([r[:,:,None],g[:,:,None],b[:,:,None]], dim=-1) 
        img_new = torch.swapaxes(img_new, 0, 2)
        
        if self.stage=='First':
            if 'normal' in self.path[idx]:
                clss=0
            elif ('benign' in self.path[idx]) or ('insitu' in self.path[idx]) or ('invasive' in self.path[idx]):
                clss=1

        elif self.stage=='Second':
            if 'benign' in self.path[idx]:
                clss=0
            elif ('insitu' in self.path[idx]) or ('invasive' in self.path[idx]):
                clss=1

        elif self.stage=='Third':
            if 'insitu' in self.path[idx]:
                clss=0
            elif 'invasive' in self.path[idx]:
                clss=1

        elif self.stage=='Test':
            clss=99

        
        sample = {'image': img_new, 'class': clss}
        return sample