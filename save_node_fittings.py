# File Name: save_node_fittings.py
# Author: Christopher Parker
# Created: Wed Jan 03, 2024 | 12:30P EST
# Last Modified: Wed Jan 03, 2024 | 12:58P EST

"""Save the results of NODE fittings as CSV files for use in MATLAB plotting"""

DIRECTORY = 'Network States/Old Individual Fittings (11 nodes)'
RESULTS_DIRECTORY = 'NODE Raw Results/Old Individual Fittings (11 nodes)'
DATA_DIRECTORY = 'Nelson TSST Individual Patient Data'

import os
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

if not os.path.exists(RESULTS_DIRECTORY):
    os.makedirs(RESULTS_DIRECTORY)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        # Set up neural networks for each element of a 3 equation HPA axis
        #  model
        self.hpa_net = nn.Sequential(
            nn.Linear(2, 11, bias=True),
            nn.ReLU(),
            nn.Linear(11, 11, bias=True),
            nn.ReLU(),
            nn.Linear(11, 2, bias=True)
        )

        for m in self.hpa_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.5)

    def forward(self, t, y):
        "Compute the next step of the diff eq by iterating the neural network"
        # Do we need to make the NN take time as an input, also?
        # self.file.write(f"y: {y},\ntype of y: {type(y)}")
        return self.hpa_net(y)


def main():
    for file in os.listdir(DIRECTORY):
        if file[0] == '.':
            continue
        state = torch.load(os.path.join(DIRECTORY, file))
        data = np.loadtxt(os.path.join(DATA_DIRECTORY, file))

        model = ANN().double()
        model.load_state_dict(state)

        y0 = data[0,1:]
        y0 = torch.from_numpy(y0)

        pred_y = odeint(model, y0, torch.linspace(0,140,100))
        np.savetxt(os.path.join(RESULTS_DIRECTORY, file), torch.cat((torch.linspace(0, 140, 100).view(-1,1), pred_y), 1))

if __name__ == '__main__':
    with torch.no_grad():
        main()
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

