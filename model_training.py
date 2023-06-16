# File Name: model_training.py
# Author: Christopher Parker
# Created: Thu Apr 27, 2023 | 04:13P EDT
# Last Modified: Sun Jun 04, 2023 | 04:17P EDT

"Training an NCDE model for individual patients from Dr Nelson's TSST data"

import torch
import numpy as np
import torchcde
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from get_nelson_data import nelsonData

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
                 interpolation='cubic'):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels)

        # self.initial represents l_{theta}^2 in the equation
        #  z_0 = l_{theta}^2 (x)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)

        # self.readout represents l_{theta}^1 in the equation
        #  y ~ l_{theta}^1 (z_T)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation

    def forward(self, coeffs):
        """coeffs is the coefficients that describe the spline between the
        datapoints. In the case of cubic interpolation (the default), this
        is a, b, 2c and 3d because the derivative of the spline is used more
        often with cubic Hermite splines"""
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
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
            t=X.interval
        )

        # cdeint returns the initial value and terminal value from the
        #  integration, we only need the terminal
        z_T = z_T[:,1]

        # Convert z_T to y with the readout linear map
        pred_y = self.readout(z_T)
        return pred_y

def main(exclude_num, num_epochs=50):
    "Main function called when script is executed"

    # Load all patients from the Nelson data into a DataLoader in batches of 3
    #  along with labels classifying as control (0) or MDD (1)
    train_X, train_y = nelsonData()

    # Exclude one patient from each network, so we can test our network later
    train_X = torch.cat(
        [
            train_X[0:exclude_num,...],
            train_X[exclude_num+1:,...]
        ]
    )
    print(train_X)
    train_y = torch.cat(
        [
            train_y[0:exclude_num],
            train_y[exclude_num+1:]
        ]
    )

    device = torch.device('cpu')

    # 3 input channels: ACTH, CORT, time
    # 8 hidden channels: arbitrarily chosen
    # 1 output channel: binary classification of control/MDD
    model = NeuralCDE(
        input_channels=3, hidden_channels=11, output_channels=1
    )
    # We need to convert the model params to double precision because that
    #  is the format of the datasets
    # model = model.double().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_coeffs = torchcde.\
        hermite_cubic_coefficients_with_backward_differences(train_X)

    train_dataset = TensorDataset(train_coeffs, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=7)

    # Loop over the number of epochs, training on each batch in the dataloader
    #  for each epoch
    for epoch in range(num_epochs):
        batch_losses = []
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch

            # Why do we squeeze? This removes the last dimension if it has
            #  size 1, but that shouldn't be the case, anyway
            # I suppose it doesn't hurt to leave it
            pred_y = model(batch_coeffs).squeeze(-1)

            # Worth trying other loss functions, but this is the one
            #  recommended by Kidger
            loss = binary_cross_entropy_with_logits(
                pred_y, batch_y,
                pos_weight=torch.tensor(
                    15/42 if exclude_num in range(15) else 14/43
                )
            )
            batch_losses.append(loss.item())

            # Perform the backpropagation and then zero the gradient for the
            #  next batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Report progress at each epoch
        print(f'Patient {exclude_num} epoch {epoch}: {np.mean(batch_losses)}')
    torch.save(
        model.state_dict(),
        f'NCDE_state_128Hnodes_8Hchannels_control-vs-MDD_classification'
        f'_exclude{exclude_num}.txt'
    )

if __name__ == '__main__':
    for i in range(58):
        main(i)

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

