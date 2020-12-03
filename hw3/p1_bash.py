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
from models import VAE, loss_function
import training_helper as th
import skimage.io
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import csv
import sys


if __name__ == '__main__':
    device = th.getCudaDevice(cudaNum = 0, torchSeed = 123)

    model = VAE(100, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    noise_name = sys.argv[1]
    model_name = sys.argv[2]
    output_dir = sys.argv[3]
    #print('noise_name',' model_name', 'output_dir')
    #print(noise_name, model_name, output_dir)
    with torch.no_grad():
        th.loadModel(model_name, model, optimizer)
        fixed_noise = torch.load(noise_name)
        results = model.decode(fixed_noise.to(device)).cpu()
    torchvision.utils.save_image(results, output_dir, nrow=8)
    print('photo saved to',output_dir)
        
    