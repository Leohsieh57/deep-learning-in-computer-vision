import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import datetime
import os, sys
from matplotlib.pyplot import imshow, imsave
import torchvision.models as models
from torch.autograd import Function
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature_src = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5),
                                         nn.BatchNorm2d(64),
                                         nn.MaxPool2d(2),
                                         nn.ReLU(True),
                                         nn.Conv2d(64, 50, kernel_size=5),
                                         nn.BatchNorm2d(50),
                                         nn.Dropout2d(),
                                         nn.MaxPool2d(2),
                                         nn.ReLU(True)
                                        )
        self.feature_tgt = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5),
                                         nn.BatchNorm2d(64),
                                         nn.MaxPool2d(2),
                                         nn.ReLU(True),
                                         nn.Conv2d(64, 50, kernel_size=5),
                                         nn.BatchNorm2d(50),
                                         nn.Dropout2d(),
                                         nn.MaxPool2d(2),
                                         nn.ReLU(True)
                                        )
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha = 0, domain = 'source'):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature_src(input_data) if domain == 'source' else self.feature_tgt(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None