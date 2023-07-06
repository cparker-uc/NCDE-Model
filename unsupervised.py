# File Name: unsupervised.py
# Author: Christopher Parker
# Created: Thu Jun 29, 2023 | 10:42P EDT
# Last Modified: Wed Jul 05, 2023 | 12:52P EDT

"""Use a Neural CDE Autoencoder to visualize the Nelson data patients. By
encoding the data into 2 dimensions and plotting, we will hopefully observe
clustering of groups of patients."""


# Network architecture parameters
INPUT_CHANNELS = 3
ENCODED_CHANNELS = 2
HDIM = 128
OUTPUT_CHANNELS = 3

# Training hyperparameters
ITERS = 5000
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-8
RTOL = 1e-6

# Training data selection parameters
PATIENT_GROUPS = ['Control', 'Atypical', 'Melancholic', 'Neither']
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'Standardize'
NUM_PER_PATIENT = 100
NUM_PATIENTS = 10
POP_NUMBER = 1
BATCH_SIZE = 1
DROPOUT = 0


# from IPython.core.debugger import set_trace
import os
import time
import torch
import torchcde
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from copy import copy
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
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
          self.linear1 = torch.nn.Linear(hidden_channels, 2)
          self.linear2 = torch.nn.Linear(2, hidden_channels*input_channels)

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


class NCDE_Encoder(torch.nn.Module):
    """This class packages the CDEFunc class with the torchcde NCDE solver,
    so that when we call the instance of NeuralCDE it solves the system"""
    def __init__(self, input_channels, hidden_channels, output_channels,
                 # adjoint_params,
                 t_interval=None, interpolation='cubic'):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels)

        # self.initial represents l_{theta}^2 in the equation
        #  z_0 = l_{theta}^2 (x)
        # This is essentially augmenting the dimension with a linear map,
        #  something Massaroli et al warned against
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=DROPOUT, inplace=True)

        # self.readout represents l_{theta}^1 in the equation
        #  y ~ l_{theta}^1 (z_T)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation
        self.t_interval = t_interval
        self.adjoint_params = self.parameters()

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
        self.dropout(z0)

        # Solve the CDE
        z_T = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=self.t_interval,
            adjoint_params=self.adjoint_params
        )
        z_T = z_T.tanh()
        # We actually need the entire time series back, so that we can have it
        #  reconstructed by the network
        # cdeint returns the initial value and terminal value from the
        #  integration, we only need the terminal
        # z_T = z_T[:,1]

        # Convert z_T to y with the readout linear map
        pred_y = self.readout(z_T)
        new_coeffs = torchcde.\
            hermite_cubic_coefficients_with_backward_differences(
                pred_y, t=self.t_interval
            )

        return (pred_y, new_coeffs)


class NCDE_Decoder(torch.nn.Module):
    """This class packages the CDEFunc class with the torchcde NCDE solver,
    so that when we call the instance of NeuralCDE it solves the system"""
    def __init__(self, input_channels, hidden_channels, output_channels,
                 # adjoint_params,
                 t_interval=None, interpolation='cubic'):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels)

        # self.initial represents l_{theta}^2 in the equation
        #  z_0 = l_{theta}^2 (x)
        # This is essentially augmenting the dimension with a linear map,
        #  something Massaroli et al warned against
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=DROPOUT, inplace=True)

        # self.readout represents l_{theta}^1 in the equation
        #  y ~ l_{theta}^1 (z_T)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation
        self.t_interval = t_interval
        self.adjoint_params = self.parameters()

    def forward(self, tuple):
        """coeffs is the coefficients that describe the spline between the
        datapoints. In the case of cubic interpolation (the default), this
        is a, b, 2c and 3d because the derivative of the spline is used more
        often with cubic Hermite splines"""
        (pred_y, coeffs) = tuple
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=self.t_interval)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only cubic and linear interpolation are implemented"
            )

        # self.adjoint_params = self.adjoint_params + (coeffs,)
        # The initial hidden state
        #   (a linear map from the first observation)
        X0 = X.evaluate(X.interval[0]) # evaluate the spline at its first point
        z0 = self.initial(X0)
        # self.dropout(z0)

        # Solve the CDE
        z_T = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=self.t_interval,
            adjoint_params=self.adjoint_params
        )
        # We actually need the entire time series back, so that we can have it
        #  reconstructed by the network
        # cdeint returns the initial value and terminal value from the
        #  integration, we only need the terminal
        # z_T = z_T[:,1]

        # Convert z_T to y with the readout linear map
        pred_y = self.readout(z_T)

        return pred_y

def train():
    device = torch.device('cpu')
    dataset = NelsonData(
        'Nelson TSST Individual Patient Data',
        patient_groups=PATIENT_GROUPS,
        normalize_standardize=NORMALIZE_STANDARDIZE
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    t_eval = dataset[0][0][:,0].contiguous()

    encoder = NCDE_Encoder(
        INPUT_CHANNELS, HDIM, ENCODED_CHANNELS, t_interval=t_eval
    ).double().to(device)
    # adjoint_params = tuple(encoder.parameters())
    decoder = NCDE_Decoder(
        ENCODED_CHANNELS, HDIM, OUTPUT_CHANNELS, #adjoint_params,
        t_interval=t_eval
    ).double().to(device)
    model = nn.Sequential(encoder, decoder)

    opt_params = list(decoder.parameters())
    optimizer = optim.AdamW(opt_params, lr=LR, weight_decay=DECAY)
    loss = nn.MSELoss()

    loss_over_time = []

    start_time = time.time()

    for itr in range(1, ITERS+1):
        # We don't need the labels, because we are doing unsupervised training
        for j, (data, _) in enumerate(loader):
            data = data.double().to(device)
            input_coeffs = torchcde.\
                hermite_cubic_coefficients_with_backward_differences(
                    data, t=t_eval
                )

            optimizer.zero_grad()

            pred_y = model(input_coeffs).squeeze(-1)
            # Encode the input data into ENCODED_CHANNELS dimensions
            # enc_y = encoder(input_coeffs).squeeze(-1)

            # encoded_coeffs = torchcde.\
            #     hermite_cubic_coefficients_with_backward_differences(
            #         enc_y, t=t_eval
            #     )

            # # Decode with the second NCDE to reconstruct the original data
            # dec_y = decoder(encoded_coeffs).squeeze(-1)

            # Compute the loss based on the reconstruction error
            # output = loss(dec_y, data)
            output = loss(pred_y, data)
            loss_over_time.append((j, output.item()))

            # Backpropagate through the adjoint of the NCDEs to compute
            #  gradients WRT each parameter
            output.backward(retain_graph=True)

            # Use the gradients calculated through backpropagation to adjust the
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")

        if OPT_RESET is None:
            pass
        elif itr % OPT_RESET == 0:
            optimizer = optim.AdamW(
                opt_params, lr=LR, weight_decay=DECAY
            )

        if itr % 100 == 0:
            runtime = time.time() - start_time
            print(f"Runtime: {runtime:.6f} seconds")
            torch.save(
                encoder.state_dict(),
                'Network States (Autoencoders)/encoder_state_2encoded-channels_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                f'dropout{DROPOUT}.txt'
            )
            torch.save(
                decoder.state_dict(),
                'Network States (Autoencoders)/decoder_state_2encoded-channels_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                f'dropout{DROPOUT}.txt'
            )
            with open('Network States (Autoencoders)/'
                      'autoencoder_state_2encoded-channels_'
                      f'batchsize{BATCH_SIZE}_'
                      f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                      f'dropout{DROPOUT}_setup.txt',
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
                    'Dropout probability '
                    f'(after initial linear layer before NCDE): {DROPOUT}\n'
                    'Training Data Selection Parameters\n'
                    '(If not virtual, the only important params are the groups'
                    ' and whether data was normalized/standardized)\n'
                    f'Patient groups={PATIENT_GROUPS}\n'
                    f'Normalized/standardized={NORMALIZE_STANDARDIZE}\n'
                    f'Training batch size={BATCH_SIZE}\n'
                    'Training Results:\n'
                    f'Runtime={runtime}\n'
                    f'Loss over time={loss_over_time}'
                )


def test_encodings(model_state):
    dataset = NelsonData(
        'Nelson TSST Individual Patient Data',
        patient_groups=PATIENT_GROUPS,
        normalize_standardize=NORMALIZE_STANDARDIZE
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    t_eval = dataset[0][0][:,0].contiguous()

    encoder = NCDE_Encoder(
        INPUT_CHANNELS, HDIM, ENCODED_CHANNELS, t_interval=t_eval
    ).double()
    encoder.load_state_dict(model_state)

    # Initialize a tensor to hold the encoded values (to save)
    encoded = torch.zeros((len(dataset), 11, 2))
    # Initialize the matplotlib figure for clustering analysis
    # fig, ax = plt.subplots(nrows=1, figsize=(10,10))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for idx, (data, _) in enumerate(loader):
        data = data.double().to(device)
        coeffs = torchcde.\
            hermite_cubic_coefficients_with_backward_differences(
                data, t=t_eval
            )
        (encoded[idx,...], _) = encoder(coeffs)


        ax.scatter(
            t_eval, encoded[idx,:,0], encoded[idx,:,1], 'o', label=f'Patient {idx}'
        )
    ax.set(title='Encoded Values All Patients')
    # ax.legend(shadow=True, fancybox=True, loc='upper right')
    plt.show()
    plt.savefig('encoded_values_all_patients_batchsize3training_3d.png', dpi=300)
    torch.save(encoded, 'encoded_values_all_patients_batchsize3training.txt')


def test_decodings(encoder_state, decoder_state):
    dataset = NelsonData(
        'Nelson TSST Individual Patient Data',
        patient_groups=PATIENT_GROUPS,
        normalize_standardize=NORMALIZE_STANDARDIZE
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    t_eval = dataset[0][0][:,0].contiguous()

    encoder = NCDE_Encoder(
        INPUT_CHANNELS, HDIM, ENCODED_CHANNELS, t_interval=t_eval
    ).double()
    encoder.load_state_dict(encoder_state)

    decoder = NCDE_Decoder(
        ENCODED_CHANNELS, HDIM, OUTPUT_CHANNELS, t_interval=t_eval
    ).double()
    decoder.load_state_dict(decoder_state)
    model = nn.Sequential(encoder, decoder)


    for idx, (data, _) in enumerate(loader):
        data = data.double().to(device)
        coeffs = torchcde.\
            hermite_cubic_coefficients_with_backward_differences(
                data, t=t_eval
            )
        reconstructed = model(coeffs).squeeze(-1)

        # print(f'{data[0][:,0]=}')
        # print(f'{reconstructed[0][:,0]=}')
        # Initialize the matplotlib figure for checking against real data
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
        ax1.plot(t_eval, data[0][:,1], label='data')
        ax1.plot(t_eval, reconstructed[0][:,1], label='reconstruction')
        ax1.set(title='ACTH')
        ax1.legend(fancybox=True, shadow=True, loc='upper right')

        ax2.plot(t_eval, data[0][:,2], label='data')
        ax2.plot(t_eval, reconstructed[0][:,2], label='reconstruction')
        ax2.set(title='CORT')
        ax2.legend(fancybox=True, shadow=True, loc='upper right')

        plt.savefig(f'Reconstruction Figures/patient{idx}.png', dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    device = torch.device('cpu')
    # train()

    encoder_state = torch.load('Network States (Autoencoders)/'
                       'encoder_state_2encoded-channels_batchsize3_1000ITER_'
                       'Standardize_dropout0.txt')
    decoder_state = torch.load('Network States (Autoencoders)/'
                       'decoder_state_2encoded-channels_batchsize3_1000ITER_'
                       'Standardize_dropout0.txt')
    with torch.no_grad():
        test_decodings(encoder_state, decoder_state)
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

