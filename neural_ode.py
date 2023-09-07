# File Name: neural_ode.py
# Author: Christopher Parker
# Created: Mon Aug 14, 2023 | 11:02P EDT
# Last Modified: Wed Sep 06, 2023 | 01:58P EDT

"""Contains the class for running NODE training"""

INPUT_CHANNELS: int = 2
NOISE_MAGNITUDE: float = 0.50
DIRECTORY = (
    f'Network States (NODE)/'
    f'Toy Dataset/'
)

import os
# Make sure the directory exists before we try to write anything
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
from get_augmented_data import ToyDataset

DEVICE = torch.device('cpu')

class NeuralODE(nn.Module):
    def __init__(self, input_channels, hdim, output_channels,
                 device=torch.device('cpu')):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_channels, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, output_channels),
            nn.ReLU(),
        ).to(device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.bias, mean=0., std=np.sqrt(2/(hdim*2)))

    def forward(self, t, y):
        """t is unnecessary, but still passed by the NODE solver"""
        return self.net(y)


def main():
    dataset = ToyDataset(
        test=False,
        noise_magnitude=NOISE_MAGNITUDE,
        method='Uniform',
        normalize_standardize='Standardize',
        t_end=2.35
    )
    loader = DataLoader(dataset, batch_size=200, shuffle=True)
    func = NeuralODE(2, 11, 2).double().to(DEVICE)
    readout = nn.Linear(2, 1).double().to(DEVICE)

    opt_params = list(func.parameters())

    optimizer = torch.optim.AdamW(opt_params, lr=1e-3, weight_decay=1e-6)

    loss_over_time = []

    start_time = time.time()
    for itr in range(1, 201):
        for j, (data, labels) in enumerate(loader):
            y0 = data[:,0,1:]
            t_eval = data[0,:,0].view(-1)
            # Ensure we have assigned the data and labels to the correct
            #  processing device
            if INPUT_CHANNELS == 2:
                data = data[...,[2,3]]
                y0 = data[:,0,...]
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            print(f"{y0.shape=}")
            print(f"{t_eval=}")
            print(f"{data.shape=}")
            # Zero the gradient from the previous data
            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = odeint(
                func, y0, t_eval
            )
            pred_y = readout(pred_y)[-1].squeeze(-1)

            # Compute the loss based on the results
            output = binary_cross_entropy_with_logits(pred_y, labels)

            # This happens in place, so we don't need to return loss_over_time
            loss_over_time.append((j, output.item()))

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            output.backward()

            # Use the gradients calculated through backpropagation to adjust the
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if ((itr == 1) or (itr % 10 == 0)):
                print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")
        if itr % 100 == 0:
            runtime = time.time() - start_time
            filename = (
                f'NN_state_11nodes_NODE_'
                f'Uniform{NOISE_MAGNITUDE}Virtual_'
                f'1000perPatient_'
                f'batchsize200_'
                f'{itr}ITER_Standardize_'
                f'smoothing0_dropout0.txt'
            )
            # Add _setup to the filename before the .txt extension
            setup_filename = "".join([filename[:-4], "_setup", filename[-4:]])
            readout_filename = "".join([filename[:-4], "_readout", filename[-4:]])

            # Save the network state dictionary
            torch.save(
                func.state_dict(), os.path.join(DIRECTORY, filename)
            )
            torch.save(
                readout.state_dict(), os.path.join(DIRECTORY, readout_filename)
            )

            # Write the hyperparameters to the setup file
            with open(os.path.join(DIRECTORY, setup_filename), 'w+') as file:
                file.write(
                    f'Model Setup for Uniform 0.1 Toy Dataset Trained NODE\n\n'
                )
                file.write(
                    'Network Architecture Parameters\n'
                    f'Input channels={INPUT_CHANNELS}\n'
                    f'Hidden channels=11\n'
                    f'Output channels=1\n\n'
                    'Training hyperparameters\n'
                    f'Optimizer={optimizer}'
                    f'Training Iterations={itr}\n'
                    f'Learning rate=1e-3\n'
                    f'Weight decay=1e-6\n'
                    f'Optimizer reset frequency=None\n\n'
                    'Dropout probability '
                    f'(after initial linear layer before NCDE): 0.\n'
                    'Training Data Selection Parameters\n'
                    '(If not virtual, the only important params are the groups'
                    ' and whether data was normalized/standardized)\n'
                    f'Augmentation strategy=Uniform\n'
                    f'Noise Magnitude={NOISE_MAGNITUDE}\n'
                    f'Normalized/standardized=Standardize\n'
                    'Number of virtual patients'
                    f' per real patient=1000\n'
                    f'Label smoothing factor=0.\n'
                    'Test Patient Combinations:\n'
                    f'Training batch size=200\n'
                    'Training Results:\n'
                    f'Runtime={runtime}\n'
                    f'Loss over time={loss_over_time}'
                    # Currently not using:
                    # f'ATOL={ATOL}\nRTOL={RTOL}\n'
                )


def test():
    dataset = ToyDataset(
        test=True,
        noise_magnitude=NOISE_MAGNITUDE,
        method='Uniform',
        normalize_standardize='Standardize',
        t_end=2.35
    )
    loader = DataLoader(dataset, batch_size=2000, shuffle=False)
    func = NeuralODE().double().to(DEVICE)
    readout = nn.Linear(2, 1).double().to(DEVICE)
    # Create the Pandas DataFrame which we will write to an Excel sheet
    performance_df = pd.DataFrame(
        columns=(
            'Iterations',
            'Success',
            'Prediction',
            'Error',
            'Cross Entropy Loss',
            'Success %'
        )
    )

    # Set up the index numbers to match the patient numbers of the test
    #  patients (or just 1-5 for Control and the same for MDD if no test
    #  combinations)
    index = [i for i in range(2000)]
    tmp_index = []
    for i, entry in enumerate(index):
        if i < 1000:
            t = 'Control ' + str(entry)
        else:
            t = 'Atypical ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Loop over the state dictionaries based on the number of iterations, from
    #  100 to MAX_ITR*100
    for itr in range(1,3):
        # Set the filename for the network state_dict
        filename = (
            f'NN_state_11nodes_NODE_'
            f'Uniform{NOISE_MAGNITUDE}Virtual_'
            f'1000perPatient_'
            f'batchsize200_'
            f'{itr*100}ITER_Standardize_'
            f'smoothing0_dropout0.txt'
        )
        readout_filename = "".join([filename[:-4], "_readout", filename[-4:]])
        state_filepath = os.path.join(DIRECTORY, filename)
        readout_state_filepath = os.path.join(DIRECTORY, readout_filename)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        readout_state_dict = torch.load(readout_state_filepath, map_location=DEVICE)
        func.load_state_dict(state_dict)
        readout.load_state_dict(readout_state_dict)

        # Pandas Series to allow us to insert the number of iterations for each
        #  group of predicitons only in the first row of the group
        iterations = pd.Series((itr*100,), index=(index[0],))

        # Loop through the test patients
        for (data, labels) in loader:
            # Ensure the data is only CORT if CORT_ONLY, and that the data and
            #  labels are loaded into the proper device memory
            y0 = data[:,0,1:]
            t_eval = data[0,:,0].view(-1)
            # Ensure we have assigned the data and labels to the correct
            #  processing device
            if INPUT_CHANNELS == 2:
                data = data[...,[2,3]]
                y0 = data[:,0,...]
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            pred_y = odeint(
                func, y0, t_eval
            )
            pred_y = readout(pred_y)[-1].squeeze(-1)
            loss = binary_cross_entropy_with_logits(pred_y, labels)

            # We need to run pred_y through a sigmoid layer to check for
            #  success and error because when training we use
            #  binary_cross_entropy_with_logits, which combines a sigmoid
            #  layer with BCE (improved performance over running sigmoid then
            #  BCE with torch)
            pred_y = torch.sigmoid(pred_y)
            error = torch.abs(labels - pred_y)

            # Rounding the predicted y to see if it was successful
            rounded_y = torch.round(pred_y)
            success = [not y for y in torch.abs(labels - rounded_y)]

            # Create Pandas Series objects so that we can insert these only
            #  in the last row of each iteration prediction group
            cross_entropy_loss = pd.Series(
                (loss.item(),),
                index=(index[-1],)
            )
            success_pct = pd.Series(
                (sum(success)/len(pred_y),),
                index=(index[-1],)
            )
            tmp_df = pd.DataFrame(
                data={
                    'Iterations': iterations,
                    'Success': success,
                    'Prediction': pred_y,
                    'Error': error,
                    'Cross Entropy Loss': cross_entropy_loss,
                    'Success %': success_pct
                }, index=index
            )

            # Add the performance metrics to the DataFrame as a new row
            performance_df = pd.concat(
                (
                    performance_df,
                    tmp_df
                ),
            )
    results_directory = (
        f'Classification Results/'
        f'Toy Dataset/NODE/'
    )
    # Make sure the directory exists before we try to write anything
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # Set the filename for the network state_dict
    filename = (
        f'NODE_11nodes_'
        f'Uniform{NOISE_MAGNITUDE}Virtual_'
        f'1000perPatient_'
        f'batchsize200_'
        f'200maxITER_Standardize_'
        f'smoothing0_dropout0.xlsx'
    )

    with pd.ExcelWriter(os.path.join(results_directory, filename)) as writer:
        performance_df.to_excel(writer)


if __name__ == '__main__':
    # with torch.no_grad():
    #     test()
    main()

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

