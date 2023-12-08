# File Name: cnn.py
# Author: Christopher Parker
# Created: Fri Dec 01, 2023 | 10:37P EST
# Last Modified: Fri Dec 01, 2023 | 10:53P EST

"""Implements a Torch Module for CNN"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1,
                 padding=0, batch_size=3):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride,
                      padding)
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self.readout = nn.Linear(32768, batch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.relu()
        x = self.maxpool(x)
        x = x.relu()
        x = torch.flatten(x)
        x = self.readout(x)
        # We don't use a final nonlinearity, since we will use the
        # binary cross entropy loss with logits function (which includes a
        # sigmoid nonlinearity)
        return x

class CNN_oldstyle(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1,
                 padding=0, batch_size=3):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride,
                      padding)
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self.readout = nn.Linear(batch_size*output_channels*121, batch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.relu()
        x = self.maxpool(x)
        x = x.relu()
        x = torch.flatten(x)
        x = self.readout(x)
        # We don't use a final nonlinearity, since we will use the
        # binary cross entropy loss with logits function (which includes a
        # sigmoid nonlinearity)
        return x

