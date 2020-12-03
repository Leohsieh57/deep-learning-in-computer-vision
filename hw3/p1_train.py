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


#preperations
trainset = th.imgData(root='hw3_data/p1_npy')
trainset_loader = DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=0)
device = th.getCudaDevice(cudaNum = 1, torchSeed = 123)


# training
latent_size = 100
with torch.no_grad():
    rand_variable = Variable(torch.randn(32,latent_size)).to(device)
model = VAE(latent_size, device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# since the cuda draws segmentation error in random eps with unknown reasons, 
#i complete the training by executing the code repeatively 
th.loadModel('p1_latest.pth', model, optimizer)
model.train()
train_MSE = np.load("hw3_data/p1_plot_npy/train_MSE.npy").tolist()
train_KLD = np.load("hw3_data/p1_plot_npy/train_KLD.npy").tolist()

#train_MSE = []
#train_KLD = []
while len(train_MSE) < 200:
    print('Epoch:',len(train_MSE))
    MSE_loss, KLD_loss = 0.0, 0.0
    for batch_idx, data in enumerate(trainset_loader):
        data = data.to(device)
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, KLD, MSE = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        MSE_loss += float(MSE.data)
        KLD_loss += float(KLD.data)  
    print("training Recon Loss:",MSE_loss/(12288*len(trainset)))
    print("training KLD_loss:", KLD_loss/len(trainset))
    print('')
    train_MSE.append(MSE_loss/(12288*len(trainset)))
    train_KLD.append(KLD_loss/len(trainset))
    th.saveModel('p1_latest.pth',model,optimizer)
    np.save("hw3_data/p1_plot_npy/train_MSE.npy", train_MSE)
    np.save("hw3_data/p1_plot_npy/train_KLD.npy", train_KLD)