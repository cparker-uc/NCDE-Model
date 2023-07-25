# File Name: test.py
# Author: Christopher Parker
# Created: Fri Jul 21, 2023 | 12:45P EDT
# Last Modified: Fri Jul 21, 2023 | 02:46P EDT

import torch
import torch.optim as optim
import torch.nn as nn

class mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 2)
        self.linear2 = nn.Linear(2, 1)
    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()

        return x


def nest(model, data, optimizer):
    for itr in range(10):
        optimizer.zero_grad()
        pred_y = model(data)
        print(f'{pred_y=}')
        output = nn.MSELoss()(pred_y, torch.ones(1))
        output.backward()
        optimizer.step()

def fun():
    model = mod()
    optimizer = optim.Adam(model.parameters())
    print(f'{model.state_dict()=}, {optimizer.state_dict()=}')
    nest(model, torch.zeros(1), optimizer)
    print(f'{model.state_dict()=}, {optimizer.state_dict()=}')


if __name__ == '__main__':
    fun()
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#                                 MIT License                                 #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#     Copyright (c) 2022 Christopher John Parker <parkecp@mail.uc.edu>        #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

