# File Name: cnn.py
# Author: Christopher Parker
# Created: Fri Dec 01, 2023 | 10:37P EST
# Last Modified: Fri Dec 01, 2023 | 10:53P EST

"""Implements a Torch Module for CNN"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_channels2, output_channels, kernel_size, stride=1,
                 padding=0, batch_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size, stride,
                      padding)
        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels2, kernel_size, stride,
                      padding)
        self.batchnorm2 = nn.BatchNorm2d(hidden_channels2)
        self.conv3 = nn.Conv2d(hidden_channels2, output_channels, kernel_size, stride,
                      padding)
        self.batchnorm3 = nn.BatchNorm2d(output_channels)
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self.linear = nn.Linear(11, 1)
        self.linear2 = nn.Linear(11, 1)
        # self.readout = nn.Linear(hidden_channels, 1)
        self.readout = nn.Linear(1936, 1)

        self.batch_size = batch_size

    def forward(self, x):
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        # x = x.relu()
        # x = self.maxpool(x)
        x = x.relu()
        # x = self.conv2(x)
        # x = self.batchnorm2(x)
        # x = x.relu()
        # x = self.conv3(x)
        # x = self.batchnorm3(x)
        # x = x.relu()
        x = self.maxpool(x)
        x = x.relu()
        # x = self.linear(x)
        # x = x.relu().squeeze()
        # x = self.linear2(x)
        # x = x.relu().squeeze()
        x = x.reshape(x.shape[0], -1)
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

