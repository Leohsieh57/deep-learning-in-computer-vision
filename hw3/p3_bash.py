import dann_model as dann
import training_helper as th
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
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import math
import sys

img_transform = transforms.Compose([
    transforms.Resize(28),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

if __name__ == '__main__':
    device = th.getCudaDevice(cudaNum = 0, torchSeed = 123)
    my_net = dann.CNNModel().to(device)
    optimizer = optim.Adam(my_net.parameters(), lr=1e-3)
    
    tgt_dir = sys.argv[1]
    tgt_name = sys.argv[2]
    output_dir = sys.argv[3]
    
    my_net.eval()
    fn = tgt_name+'.pth'
    th.loadModel(fn , my_net, optimizer)
    data = th.labelImgData(root=tgt_dir, transform=img_transform)
    test_loader = DataLoader(data, batch_size=1024, shuffle=False, num_workers=0)
    
    my_net.eval()
    labelList = []
    for batch_idx, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        output, _ = my_net(img)
        _, pred = torch.max(output, 1)
        labelList += pred.cpu().numpy().tolist()
        
    with open(output_dir, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, label in enumerate(labelList):
            fn = str(i)
            while len(fn) < 5:
                fn = '0'+fn
            fn += '.png'
            writer.writerow({'image_name': fn, 'label': label})
    print('csv file writen as',output_dir)