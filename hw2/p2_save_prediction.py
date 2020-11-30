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
import sys

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def getImg(fn):
    trans = transforms.ToTensor()
    return trans(Image.open(fn)).numpy()

def getDevice():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    #device = torch.device('cpu')
    print('Device used:', device)
    return device
    
def getMaxLabel(output):
    width, height = output.shape[2], output.shape[3]
    labelList = [np.argmax(output[0,:,i,j]) for i in range(width) for j in range(height)]
    labelCnt = np.zeros(7)
    for label in labelList:
        labelCnt[label] += 1
    print(labelCnt[:6])
    return np.reshape(labelList, (width, height))

def getPrediction(filepath, model_name):
    fnImgList = sorted(glob.glob(os.path.join(filepath, '*.jpg')))
    
    device = getDevice()
    model = fcn16s(7)
    if model_name == 'p2_fcn32.pth':
        model = fcn32s(7)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0002, betas=(0.9, 0.999))
    load_checkpoint(model_name, model, optimizer)
    
    imgList = [getImg(fn) for fn in  fnImgList]
    print('images successfully loaded')
    
    model.eval()
    predList = []
    with torch.no_grad():
        for index, img in enumerate(imgList):
            if index % 10 == 0:
                print('number of predictions completed =', index+1)
            img = np.array([img])
            img = torch.from_numpy(img)
            img = img.to(device)
            output = model(img)
            pred = output.max(1, keepdim=True)[1]
            predList.append(pred.cpu())
            
    print('prediction completed')
    return predList


def encodeLabels(pred):
    pred = pred[0][0].numpy()
    width, height = pred.shape
    predRGB = np.zeros((width, height, 3))
    decode = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
    for w in range(width):
        for h in range(height):
            predRGB[w][h] = decode[pred[w][h]]
    
    return predRGB.astype(np.uint8)

def getFileName(index):
    fnString = str(index)
    while len(fnString) < 4:
        fnString = "0" + fnString
    return fnString + "_mask.png"

class fcn32s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn32s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, num_classes, 64 , 32 , 0, bias=False),
        )
    def  forward (self, x) :        
        x = self.vgg.features(x)
        x = self.vgg.classifier(x)
        return x

class fcn16s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn16s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.to_pool4 = nn.Sequential(*list(self.vgg.features.children())[:24])
        self.to_pool5 = nn.Sequential(*list(self.vgg.features.children())[24:])
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, 512, 4 , 2 , 0, bias=False)
            )
        self.upsample16 = nn.ConvTranspose2d(512, num_classes, 16 , 16 , 0, bias=False)
        
    def forward (self, x) :        
        pool4_output = self.to_pool4(x) #pool4 output size torch.Size([64, 512, 16, 16])
        x = self.to_pool5(pool4_output)
        x = self.vgg.classifier(x)    # 2xconv7 output size torch.Size([64, 512, 16, 16])
        x = self.upsample16(x+pool4_output)
        return x

if __name__ == '__main__':
    model_name = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    predList = getPrediction(input_dir, model_name)
    
    pool = mp.Pool(mp.cpu_count())
    npList = pool.map(encodeLabels, predList)
    
    print('label encoding completed')
    
    for index, pred in enumerate(npList):
        fn = getFileName(index)
        skimage.io.imsave(os.path.join(output_dir,fn), pred)
        if index % 100 == 0:
            print('number of savings completed =', index+1)
        
    