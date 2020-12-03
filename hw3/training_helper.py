import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import copy
from torchvision.utils import save_image
import PIL
import skimage.io
import multiprocessing as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
import random

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def getLambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-100.*p)) - 1.


class hybridImgData(Dataset):
    def __init__(self, src_root, tgt_root, transform = None):
        src_filenames = glob.glob(os.path.join(src_root, '*.png'))
        tgt_filenames = glob.glob(os.path.join(tgt_root, '*.png'))
        while len(tgt_filenames) < len(src_filenames):
            tgt_filenames += tgt_filenames
        while len(tgt_filenames) > len(src_filenames):
            src_filenames += src_filenames
        dataLength = min(len(tgt_filenames), len(src_filenames))
        tgt_filenames = tgt_filenames[:dataLength]
        src_filenames = src_filenames[:dataLength]
        self.src_len = len(src_filenames)
        self.filenames = src_filenames + tgt_filenames
        self.len = len(self.filenames)
        self.trans = transform
        
    def __getitem__(self, index):
        #print(self.filenames[index])
        img = Image.open(self.filenames[index])
        img = img if self.trans == None else self.trans(img)
        return (img, 0) if index < self.src_len else (img,  1)
    
    def __len__(self):
        return self.len
    
    
class labelImgData(Dataset):
    def __init__(self, root, transform = None):
        self.root = root
        filenames = sorted(glob.glob(os.path.join(self.root, '*.png')))
        self.trans = transform
        self.len = len(filenames)
        dataType = root.split('/')[-1]
        fn = root+".csv"
        labels = np.array(pd.read_csv(fn)['label'])
        self.fnLabelList = [(filenames[i], labels[i]) for i in range(self.len)]
        self.discardedfnLabelList = []
        
    def __getitem__(self, index):
        fn, label = self.fnLabelList[index]
        img = Image.open(fn)
        img = img.convert('RGB')
        img = img if self.trans == None else self.trans(img)
        return img, label
    
    def rngDiscarding(self, reserveCnt = 5000):
        reserveCnt = min(reserveCnt, self.len)
        self.fnLabelList += self.discardedfnLabelList
        random.shuffle(self.fnLabelList)
        self.discardedfnLabelList = self.fnLabelList[reserveCnt:]
        self.fnLabelList = self.fnLabelList[:reserveCnt]
        self.len = len(self.fnLabelList)
        
    def restoreFromDiscarding(self):
        self.fnLabelList += self.discardedfnLabelList
        self.discardedfnLabelList = []
        self.len = len(self.fnLabelList)
    
    def __len__(self):
        return self.len
    
class imgData(Dataset):
    def __init__(self, root):
        self.root = root
        self.filenames = sorted(glob.glob(os.path.join(self.root, '*.npy')))
        self.len = len(self.filenames) 
    
    def __getitem__(self, index):
        return np.load(self.filenames[index])
    
    def __len__(self):
        return self.len
    
def saveModel(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def loadModel(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
def getCudaDevice(cudaNum = 1, torchSeed = 123):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(torchSeed)
    deviceName = "cuda:"+str(cudaNum)
    device = torch.device(deviceName if use_cuda else "cpu")
    #device = torch.device('cpu')
    print('Device used:', device)
    return device