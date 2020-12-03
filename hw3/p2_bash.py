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
from torch.autograd import Variable
from models import  Generator
import training_helper as th
import skimage.io
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import csv
import sys



if __name__ == '__main__':
    device = th.getCudaDevice(cudaNum = 0, torchSeed = 123)
    G = Generator().to(device)
    G.eval()
    
    noise_name = sys.argv[1]
    model_name = sys.argv[2]
    output_dir = sys.argv[3]
    
    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    th.loadModel(model_name,G,optimizerG)
    #print('noise_name',' model_name', 'output_dir')
    #print(noise_name, model_name, output_dir)

    fixed_noise = torch.load(noise_name)
    results = G(fixed_noise.to(device)).cpu().data
    torchvision.utils.save_image(results, output_dir, nrow=8)
    print('photo saved to',output_dir)
        
    