import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import os, sys, pdb, tqdm, random, json, gzip, bz2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import importlib
import copy
import argparse
from torchvision import transforms, datasets





def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




class wide_resnet_t(nn.Module):

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)


    class wide_basic(nn.Module):
        def __init__(self, in_planes, planes, dropout_rate, stride=1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes, affine=False)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.bn2 = nn.BatchNorm2d(planes, affine=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

            self.shortcut = nn.Sequential()

            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size = 1, stride=stride, bias=True), )

        def forward(self, x):
            # out = self.bn1(x)
            out = self.dropout(self.conv1(F.relu(out)))
            # out = self.bn2(out)
            out = self.conv2(F.relu(out))
            out += self.shortcut(x)
            return out


    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_planes):
        super().__init__()
        self.in_planes = in_planes

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| wide_resnet_t %dx%d' %(depth, k))

        nStages = [in_planes, in_planes*k, 2*in_planes*k, 4*in_planes*k]

        self.conv1 = self.conv3x3(self.in_planes, nStages[0])
        self.layer1 = self._wide_layer(self.wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(self.wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(self.wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9, affine=False)
        self.linear = nn.Linear(nStages[3], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.bn1(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out,1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out