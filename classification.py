# File Name: galerkin_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Fri Jun 16, 2023 | 10:48P EDT

"Working on NCDE classification of augmented Nelson data"

# Network architecture parameters
INPUT_CHANNELS = 3
HDIM = 32
OUTPUT_CHANNELS = 1

# Training hyperparameters
ITERS = 500
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-8
RTOL = 1e-6

# Training data selection parameters
PATIENT_GROUPS = ['Control', 'Atypical']
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'Standardize'
NUM_PER_PATIENT = 100
NUM_PATIENTS = 10
POP_NUMBER = 1
BATCH_SIZE = 100


# from IPython.core.debugger import set_trace
import os
import time
import torch
import torchcde
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from torchdyn.core import NeuralODE
from typing import Tuple
from augment_data import NORMALIZE_STANDARDIZE, NUM_PATIENTS
from get_nelson_data import NelsonData, VirtualPopulation


# Not certain if this is necessary, but in the quickstart docs they have
#  done a wildcard import of torchdyn base library, and this is all that does
TTuple = Tuple[torch.Tensor, torch.Tensor]


class CDEFunc(torch.nn.Module):
    """CDEs are defined as: z_t = z_0 + \int_{0}^t f_{theta}(z_s) dX_s, where
    f_{theta} is a neural network (and X_s is a rough path controlling the
    diff eq. This class defines f_{theta}"""
    def __init__(self, input_channels, hidden_channels):
          super().__init__()
          self.input_channels = input_channels
          self.hidden_channels = hidden_channels

          # Define the layers of the NN, with 128 hidden nodes (arbitrary)
          self.linear1 = torch.nn.Linear(hidden_channels, 128)
          self.linear2 = torch.nn.Linear(128, hidden_channels*input_channels)

    def forward(self, t, z):
          """t is passed as an argument by the solver, but it is unused in most
          cases"""
          z = self.linear1(z)
          z = z.relu()
          z = self.linear2(z)

          # The first author of the NCDE paper (Kidger) suggests that using tanh
          #  for the final activation leads to better results
          z = z.tanh()

          # The output represents a linear map from R^input_channels to
          #  R^hidden_channels, so it takes the form of a
          #  (hidden_channels x input_channels) matrix
          z = z.view(z.size(0), self.hidden_channels, self.input_channels)
          return z


class NeuralCDE(torch.nn.Module):
    """This class packages the CDEFunc class with the torchcde NCDE solver,
    so that when we call the instance of NeuralCDE it solves the system"""
    def __init__(self, input_channels, hidden_channels, output_channels,
                 t_interval=None, interpolation='cubic'):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels)

        # self.initial represents l_{theta}^2 in the equation
        #  z_0 = l_{theta}^2 (x)
        # This is essentially augmenting the dimension with a linear map,
        #  something Massaroli et al warned against
        self.initial = torch.nn.Linear(input_channels, hidden_channels)

        # self.readout represents l_{theta}^1 in the equation
        #  y ~ l_{theta}^1 (z_T)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation
        self.t_interval = t_interval

    def forward(self, coeffs):
        """coeffs is the coefficients that describe the spline between the
        datapoints. In the case of cubic interpolation (the default), this
        is a, b, 2c and 3d because the derivative of the spline is used more
        often with cubic Hermite splines"""
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=self.t_interval)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only cubic and linear interpolation are implemented"
            )

        # The initial hidden state
        #   (a linear map from the first observation)
        X0 = X.evaluate(X.interval[0]) # evaluate the spline at its first point
        z0 = self.initial(X0)

        # Solve the CDE
        z_T = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=self.t_interval
        )
        # cdeint returns the initial value and terminal value from the
        #  integration, we only need the terminal
        z_T = z_T[:,1]

        # Convert z_T to y with the readout linear map
        pred_y = self.readout(z_T)
        return pred_y

class NDEOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tup):
        (t_eval, sol) = tup
        # The result returned from NeuralODE is (11, 1, 2) instead of
        #  (11, 2, 1) so we swap the last two axes
        # return torch.swapaxes(sol, 1, 2)
        return sol


def main(virtual=True):
    device = torch.device('cpu')

    if not virtual:
        dataset = NelsonData(
            'Nelson TSST Individual Patient Data',
            patient_groups=PATIENT_GROUPS,
            normalize_standardize='standardize'
        )
    else:
        dataset = VirtualPopulation(
            patient_groups=PATIENT_GROUPS,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            num_patients=NUM_PATIENTS,
            pop_number=POP_NUMBER
        )
    # Time points we need the solver to output
    t_eval = dataset[0][0][:,0].contiguous()

    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    model = NeuralCDE(
        INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS, t_interval=t_eval
    ).double()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

    # loss = nn.CrossEntropyLoss()
    loss_over_time = []

    start_time = time.time()
    print(f'Starting Training on {PATIENT_GROUP} Population Number {POP_NUMBER}')
    for itr in range(1, ITERS+1):
        for j, (data, labels) in enumerate(dataloader):
            coeffs = torchcde.\
                hermite_cubic_coefficients_with_backward_differences(
                    data, t=t_eval
                )

            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = model(coeffs).squeeze(-1)

            # Compute the loss based on the results
            output = binary_cross_entropy_with_logits(pred_y, labels)
            loss_over_time.append((j, output.item()))

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            output.backward()

            # Use the gradients calculated through backpropagation to adjust the
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")

        # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
        #  reset the learning rate and momentum
        if OPT_RESET is None:
            pass
        elif itr % OPT_RESET == 0:
            optimizer = optim.AdamW(
                model.parameters(), lr=LR, weight_decay=DECAY
            )

        if itr % 100 == 0:
            runtime = time.time() - start_time
            print(f"Runtime: {runtime:.6f} seconds")
            torch.save(
                model.state_dict(),
                f'Network States (VPOP Training)/NN_state_2HL_128nodes_NCDE_'
                f"{METHOD}{'Virtual' if virtual else 'Real'}{PATIENT_GROUP}"
                f'{POP_NUMBER}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr}ITER_{NORMALIZE_STANDARDIZE}.txt'
            )
            with open(f'Network States (VPOP Training)/NN_state_2HL_128nodes_'
                      f"NCDE_{METHOD}{'Virtual' if virtual else 'Real'}"
                      f'{PATIENT_GROUP}{POP_NUMBER}_'
                      f'{NUM_PER_PATIENT}perPatient_'
                      f'batchsize{BATCH_SIZE}_'
                      f'{itr}ITER_{NORMALIZE_STANDARDIZE}_setup.txt',
                      'w+') as file:
                file.write(
                    f'Model Setup for {METHOD} '
                    "{'virtual' if virtual else 'real'} "
                    '{PATIENT_GROUP} Trained Network:\n\n'
                )
                file.write(
                    'Network Architecture Parameters\n'
                    f'Input channels={INPUT_CHANNELS}\n'
                    f'Hidden channels={HDIM}\n'
                    f'Output channels={OUTPUT_CHANNELS}\n\n'
                    'Training hyperparameters\n'
                    f'Optimizer={optimizer}'
                    f'Training Iterations={itr}\n'
                    f'Learning rate={LR}\n'
                    f'Weight decay={DECAY}\n'
                    f'Optimizer reset frequency={OPT_RESET}\n\n'
                    'Training Data Selection Parameters\n'
                    '(If not virtual, the only important params are the groups'
                    ' and whether data was normalized/standardized)\n'
                    f'Patient groups={PATIENT_GROUPS}\n'
                    f'Augmentation strategy={METHOD}\n'
                    f'Normalized/standardized={NORMALIZE_STANDARDIZE}\n'
                    'Number of virtual patients'
                    f' per real patient={NUM_PER_PATIENT}\n'
                    'Number of real patients sampled from each group='
                    f'{NUM_PATIENTS}\n'
                    f'Virtual population number used={POP_NUMBER}\n'
                    f'Training batch size={BATCH_SIZE}\n'
                    'Training Results:\n'
                    f'Runtime={runtime}\n'
                    f'Loss over time={loss_over_time}'
                    # Currently not using:
                    # f'ATOL={ATOL}\nRTOL={RTOL}\n'
                )


if __name__ == "__main__":
    for POP_NUMBER in range(1,6):
        for PATIENT_GROUPS in [['Control', 'Atypical'], ['Control', 'Melancholic'],
                               ['Control', 'Neither']]:
            PATIENT_GROUP = PATIENT_GROUPS[1]
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
