# File Name: patient_fitting_old.py
# Description: 
# Author: Christopher Parker
# Created: Fri Dec 08, 2023 | 10:58P EST
# Last Modified:
# File Name: model.py
# Author: Christopher Parker
# Created: Fri Mar 24, 2023 | 10:10P EDT
# Last Modified: Mon Apr 10, 2023 | 08:26P EDT

"First pass at an NODE model with PyTorch"

ITERS = 5000
LEARNING_RATE = 1e-3
OPT_RESET = 200
ATOL = 1e-9
RTOL = 1e-7
METHOD = 'dopri5'

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint

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

class NelsonData(Dataset):
    def __init__(self, data_dir, patient_group):
        self.data_dir = data_dir
        self.patient_group = patient_group

    def __len__(self):
        return 15

    def __getitem__(self, idx):
        """This function will be used by the DataLoader to iterate through the
        data files of the given patient group and load the data and labels.
        Due to the nature of the problem, we actually call the time points the
        data and the concentrations the labels because given the 'data' the
        ANN should try to match the 'label'. This is slightly different than
        what would normally be used for training on an image, or something
        because the data is a time series, as is the label."""
        ACTHdata_path = os.path.join(
            self.data_dir, f'{self.patient_group}Patient{idx+1}_ACTH.txt'
        )
        CORTdata_path = os.path.join(
            self.data_dir, f'{self.patient_group}Patient{idx+1}_CORT.txt'
        )

        ACTHdata = np.genfromtxt(ACTHdata_path)
        CORTdata = np.genfromtxt(CORTdata_path)

        data = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[0,2]]
        )
        label = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[1,3]]
        )
        return data, label

if __name__ == '__main__':
    for i in range(15):
        dataset = NelsonData(
            '/Users/christopher/Documents/PTSD/NODE Model.nosync/Nelson TSST'
            ' Individual Patient Data', 'Control'
        )
        # loader = DataLoader(
        #     dataset=dataset, batch_size=3, shuffle=True
        # )
        data, label = dataset[i]

        # Define the device to use for neural network computations
        device = torch.device('cpu')

        # We need to convert the model parameters to double precision because
        #  that is the format of the datasets and they must match
        func = ANN().double().to(device)

        # List of parameters to optimize
        opt_params = list(func.parameters())

        # Initialize the optimizer and the loss function
        optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
        loss = nn.MSELoss()

        # Initialize tensor to track change in loss over each iteration
        loss_over_time = torch.zeros(ITERS)

        start_time = time.time()
        # Start main optimization loop
        for itr in range(1, ITERS + 1):
            # Reset gradient for each training example
            optimizer.zero_grad()

            y0_tensor = label[0,:]

            pred_y = odeint(
                func,
                y0_tensor,
                data[:,0],
                rtol=RTOL,
                atol=ATOL,
                method=METHOD,
            )
            # Compute the loss for this iteration
            output = loss(pred_y, label)

            # Backpropagation to calculate the gradient from the loss
            output.backward()

            # Step the optimizer with the new gradient
            optimizer.step()

            # Save the loss value to the loss_over_time tensor
            loss_over_time[itr-1] = output.item()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}: loss = {output.item():.6f}")

            # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
            #  reset the learning rate
            if itr % OPT_RESET == 0:
                optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

        runtime = time.time() - start_time
        print(f"Runtime: {runtime:.6f} seconds")


        torch.save(
            func.state_dict(),
            f'NN_state_2HL_11nodes_controlPatient{i}_5kITER_200optreset.txt'
        )
        # torch.save(
        #     optimizer.state_dict(),
        #     f'optimizer_state_Adam_controlPatient{i}.txt'
        # )

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

