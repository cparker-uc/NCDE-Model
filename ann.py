# File Name: ann.py
# Author: Christopher Parker
# Created: Tue Aug 22, 2023 | 10:07P EDT
# Last Modified: Tue Sep 19, 2023 | 02:26P EDT

"""Contains the class for using ANN instead of NODE/NCDE"""

import torch
import torch.nn as nn
import numpy as np


class ANN(nn.Module):
    def __init__(self, input_channels, hdim, output_channels, dropout=0.,
                 device=torch.device('cpu')):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_channels, hdim),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, output_channels),
        ).to(device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias, 0, np.sqrt(6/(hdim*2)))

    def forward(self, y):
        return self.net(y)


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

