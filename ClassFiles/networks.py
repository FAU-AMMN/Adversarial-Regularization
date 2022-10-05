import torch.nn as nn
import torch
import math
from ClassFiles import util as ut
import torch.nn.functional as F

#from abc import ABC, abstractmethod

class Conv2dSame(nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
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

class ConvNetClassifier(nn.Module):

    def __init__(self, size, colors):
        
        super(ConvNetClassifier, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.colors = colors
        self.conv1 = nn.Conv2d(3,16,5,padding="same")
        self.conv2 = nn.Conv2d(16,32,5,padding="same")
        self.conv3 = Conv2dSame(32,32,5, stride = 2)
        self.conv4 = Conv2dSame(32,64,5, stride=2)
        self.conv5 = Conv2dSame(64,64,5, stride=2)
        self.conv6 = Conv2dSame(64,128,5, stride=2)
        #image size is now imagesize/16

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(self.size[-2] * self.size[-1]/(16*16))
        self.dimensionality = finishing_size * 128
        #reshaped = tf.reshape(self.conv6, [-1, dimensionality])
        
        self.fc1 = nn.Linear(self.dimensionality, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        
        x = x.type(torch.FloatTensor)
        x = x.to(self.device)
        x = self.conv1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x) #negative_slope = 0.2 to be comparable with tensorflow implementation
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
        x = torch.flatten(x, start_dim=1)    
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        output = self.fc2(x)

        return output        