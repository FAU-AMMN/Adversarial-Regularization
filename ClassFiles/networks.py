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

class Conv2dSame(nn.Conv2d):
    #strided convolution
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode='circular'
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class resblock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,kernel_size=5,padding= "same")
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size=5,padding= "same")
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(x)
        out += residual
        out = self.relu2(out)

        return out

class ConvNetClassifier(nn.Module):

    def __init__(self, size, colors):
        
        super(ConvNetClassifier, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.colors = colors
        self.conv1 = nn.Conv2d(self.colors,16,5,padding="same", padding_mode='circular')
        self.conv2 = nn.Conv2d(16,32,5,padding="same", padding_mode='circular')
        self.conv3 = Conv2dSame(32,32,5,stride=2)

        self.conv4 = Conv2dSame(32,64,5, stride=2)
        self.conv5 = Conv2dSame(64,64,5, stride=2)
        self.conv6 = Conv2dSame(64,128,5, stride=2)

        self.adapAvgPool = nn.AdaptiveAvgPool2d((8,8))

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(128 * 128/(16*16))
        self.dimensionality = finishing_size * 128
        
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
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv4(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv5(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv6(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.adapAvgPool(x)
        x = torch.reshape(x, (-1,self.dimensionality))    
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        output = self.fc2(x)

        return output 
    

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
        finishing_size = int(128 * 128/(16*16))
        self.dimensionality = finishing_size * 128
        
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