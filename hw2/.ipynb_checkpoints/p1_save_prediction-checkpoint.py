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
import argparse
import csv
import sys


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def getDevice():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    #device = torch.device('cpu')
    print('Device used:', device)
    return device

def getImg(fn):
    trans = transforms.ToTensor()
    return trans(Image.open(fn)).numpy()

def getLabelAndIndexFromFileName(fn):
    _fn = fn.split('/').pop()
    _list = _fn.split("_")
    label = int(_list[0])
    index = int(_list[1].split(".")[0])
    return label, index

def getPureFileName(fn):
    _fn = fn.split('/').pop()
    return _fn

def getPrediction(filepath, model_name):
    fnImgList = sorted(glob.glob(os.path.join(filepath, '*.png')))

    device = getDevice()
    model =  Net()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9, weight_decay = 1e-3)
    load_checkpoint(model_name, model, optimizer)
    
    model.eval()
    predList = []
    with torch.no_grad():
        for iteration, fn in enumerate(fnImgList):
            if iteration % 10 == 0:
                print('number of predictions completed =', iteration+1)
            
            label, index = getLabelAndIndexFromFileName(fn)
            img = getImg(fn)
            img = np.array([img])
            img = torch.from_numpy(img)
            img = img.to(device)
            output = model(img)
            pred = output.max(1, keepdim=True)[1]
            predList.append((getPureFileName(fn), pred.cpu().numpy()[0][0], label))
        return predList

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            models.vgg19_bn(pretrained=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 50)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    model_name = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    predList = getPrediction(input_dir, model_name)
    correct = sum([label == pred for fn, pred, label in predList])
    percentage = correct*100.0/len(predList)
    print('accuracy =',percentage)
    csvFn = os.path.join(output_dir,'test_pred.csv')
    with open(csvFn, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, args in enumerate(predList):
            fn, pred, label = args
            writer.writerow({'image_id': fn, 'label': pred})
    print('csv file writen as',csvFn)
        
    