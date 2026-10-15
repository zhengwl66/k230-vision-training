"""Train a small supervised pose image classifier using the detector's pretrained stem."""
import copy,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,str(ROOT))
MAP={2:0,3:1,5:2}
SIDE=96

class PoseNet(nn.Module):
    def __init__(self,stem):
        super().__init__()
        self.gray=nn.Conv2d(3,3,1,bias=False)
        with torch.no_grad():self.gray.weight.fill_(1./3)
        self.gray.weight.requires_grad_(False)
        self.stem=stem
        self.pool=nn.AdaptiveAvgPool2d((4,4))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128*4*4,96),nn.ReLU(),nn.Dropout(.20),nn.Linear(96,6))
    def forward(self,x):return self.head(self.pool(self.stem(self.gray(x))))

def crop_tensor(image,augment=False):
    if augment:
        if random.random()<.5:image=ImageOps.mirror(image)
        # Physical in-plane rotations are present in captures; only small image
        # perturbations are used because a camera tilt is not a pose label.
        image=image.rotate(random.uniform(-8,8),resample=Image.Resampling.BILINEAR,fillcolor=(190,190,190))
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.65,1.40))
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.75,1.25))
        if random.random()<.5:
            image=image.resize((max(24,image.width//2),max(24,image.height//2)),Image.Resampling.BILINEAR)
    image=image.resize((SIDE,SIDE),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image,dtype=np.float32).transpose(2,0,1).copy()/255.)

class Captures(Dataset):
    def __init__(self,rows,augment=False):
        self.rows=rows;self.augment=augment
        self.images=[]
        for row in rows:
            image=Image.open(row['image']).convert('RGB').crop(tuple(row['box']))
            self.images.append(image)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];label=MAP[row['class_id']]*2+int(row['label']>0)
        return crop_tensor(self.images[index],self.augment),label

def evaluate(model,loader,device):
    model.eval();outputs=[];labels=[]
    with torch.inference_mode():
        for x,y in loader:
            outputs.append(model(x.to(device)).cpu());labels.append(y)
    return torch.cat(outputs).numpy(),torch.cat(labels).numpy()
