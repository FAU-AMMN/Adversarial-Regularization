import torch.nn as nn
import torch
import math
from ClassFiles import util as ut
import torch.nn.functional as F
import numpy as np


import sys, os
sys.path.append(os.path.abspath('../FourierImaging/'))

import fourierimaging as fi

from fourierimaging.modules import TrigonometricResize_2d, conv_to_spectral


class ConvNetClassifier_nostride(nn.Module):

    def __init__(self, size, colors):
        
        super(ConvNetClassifier_nostride, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.colors = colors

        self.resize = TrigonometricResize_2d

        self.conv1 = nn.Conv2d(self.colors,16,5,padding="same", padding_mode='circular')
        self.conv2 = nn.Conv2d(16,32,5,padding="same", padding_mode='circular')
        self.conv3 = nn.Conv2d(32,32,5,padding="same", padding_mode='circular')
        #resize
        self.conv4 = nn.Conv2d(32,64,5, padding="same", padding_mode='circular')
        #resize
        self.conv5 = nn.Conv2d(64,64,5, padding="same", padding_mode='circular')
        #resize
        self.conv6 = nn.Conv2d(64,128,5, padding="same", padding_mode='circular')
        #resize

        self.adapAvgPool = nn.AdaptiveAvgPool2d((8,8))

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(128 * 128/(16*16)) #64x64
        self.dimensionality = finishing_size * 128 #8192
        
        self.fc1 = nn.Linear(self.dimensionality, 256)
        self.fc2 = nn.Linear(256, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
                
        x = x.type(torch.FloatTensor)
        x = x.to(self.device)
        x = self.conv1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.conv3(x)
        new_size = [int(np.ceil(x.shape[-2]/2)), int(np.ceil(x.shape[-1]/2))] 
        x = self.resize([new_size[0], new_size[1]])(x)

        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv4(x)
        new_size = [int(np.ceil(x.shape[-2]/2)), int(np.ceil(x.shape[-1]/2))] 
        x = self.resize([new_size[0], new_size[1]])(x)

        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv5(x)
        new_size = [int(np.ceil(x.shape[-2]/2)), int(np.ceil(x.shape[-1]/2))] 
        x = self.resize([new_size[0], new_size[1]])(x)

        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv6(x)
        new_size = [int(np.ceil(x.shape[-2]/2)), int(np.ceil(x.shape[-1]/2))] 
        x = self.resize([new_size[0], new_size[1]])(x)

        x = nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.adapAvgPool(x)
        x = torch.reshape(x, (-1,self.dimensionality))    
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        output = self.fc2(x)

        return output 
    
class Spectral_withResize(ConvNetClassifier_nostride):
    """
    FNO layer
    """
    def __init__(self, size, colors):
        super(Spectral_withResize,self).__init__(size, colors)
        
        self.conv1 = conv_to_spectral(self.conv1, size, in_shape = size)
        self.conv2 = conv_to_spectral(self.conv2, size, in_shape = size)
        self.conv3 = conv_to_spectral(self.conv3, size, in_shape = size)
        #resize
        size = (int(size[0]/2),int(size[1]/2))
        self.conv4 = conv_to_spectral(self.conv4, size, in_shape = size)
        #resize
        size = (int(size[0]/2),int(size[1]/2))
        self.conv5 = conv_to_spectral(self.conv5, size, in_shape = size)
        #resize
        size = (int(size[0]/2),int(size[1]/2))
        self.conv6 = conv_to_spectral(self.conv6, size, in_shape = size)
        #resize
        
class Spectral_FromTrainedConv(ConvNetClassifier_nostride):
    """
    Model uses CNN trained weights and converts CNN layers to FNO
    """
    def __init__(self, size, colors):
        super(Spectral_FromTrainedConv,self).__init__(size, colors)
    
    def convert_to_spectral(self, input_size): 
        input_size = self.size #if inputSize != self.size, then FNO performs like CNN
        self.conv1 = conv_to_spectral(self.conv1, input_size)#, in_shape = input_size )
        self.conv2 = conv_to_spectral(self.conv2, input_size)
        self.conv3 = conv_to_spectral(self.conv3, input_size)
        #resize
        size = (int(input_size [0]/2),int(input_size [1]/2))
        self.conv4 = conv_to_spectral(self.conv4, size)#, in_shape = size )
        #resize
        size = (int(size[0]/2),int(size[1]/2))
        self.conv5 = conv_to_spectral(self.conv5, size)
        #resize
        size = (int(size[0]/2),int(size[1]/2))
        self.conv6 = conv_to_spectral(self.conv6, size)
        #resize     
