# File Name: rnn.py
# Author: Christopher Parker
# Created: Tue Aug 22, 2023 | 02:37P EDT
# Last Modified: Fri Sep 29, 2023 | 01:14P EDT

"""Contains the class for using ANN instead of NODE/NCDE"""

import torch
import torch.nn as nn
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_channels, hdim, output_channels, n_layers,
                 device=torch.device('cpu')):
        super().__init__()

        self.net = nn.RNN(
            input_channels, hdim, n_layers,
        ).to(device)

        # for name, param in self.net.named_parameters():
        #     # We want to initialize the biases as 0
        #     if name[:4].lower() == 'bias':
        #         nn.init.constant_(param, 0)
        #         continue
        #     # And the weights with Xavier normal
        #     nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('tanh'))

        self.readout = nn.Linear(hdim, output_channels)

    def forward(self, y):
        (y, _) = self.net(y)
        y = self.readout(y)

        if len(y.shape) > 1 and y.shape[-1] >= 3:
            y[...,[1,2]] = y[...,[1,2]].relu()
            y[...,[0,3]] = y[...,[0,3]].tanh()
        return y


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#                                 MIT License                               #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#     Copyright (c) 2023 Christopher John Parker <parkecp@mail.uc.edu>      #
#                                                                           #
# Permission is hereby granted, free of charge, to any person obtaining a   #
# copy of this software and associated documentation files (the "Software"),#
# to deal in the Software without restriction, including without limitation #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,  #
# and/or sell copies of the Software, and to permit persons to whom the     #
# Software is furnished to do so, subject to the following conditions:      #
#                                                                           #
# The above copyright notice and this permission notice shall be included   #
# in all copies or substantial portions of the Software.                    #
#                                                                           #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS   #
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF                #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.    #
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY      #
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,      #
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE         #
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                    # 
#                                                                           #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

