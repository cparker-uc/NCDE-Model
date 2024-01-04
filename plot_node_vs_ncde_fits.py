# File Name: plot_node_vs_ncde_fits.py
# Author: Christopher Parker
# Created: Tue Dec 19, 2023 | 02:58P EST
# Last Modified: Thu Dec 21, 2023 | 08:50P EST

"""This script plots the NODE and NCDE fits for a single patient on the same
axes"""

import torch
import torchcde
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from neural_cde import NeuralCDE
from neural_ode import NeuralODE
from get_data import NelsonData

def main():
    dense_t = torch.linspace(0, 140, 100)
    node_model = NeuralODE(
        2, 32, 2,
    )
    node_state = torch.load('/Users/christopher/Documents/PTSD/NCDE Model.nosync/Network States/Prediction/Control/NN_state_32nodes_NODE_Control1_batchsize1_5000ITER_None_smoothing0_dropout0.0.txt')
    node_model.load_state_dict(node_state)
    ncde_model = NeuralCDE(
        2, 2, 2, t_interval=dense_t, prediction=True,
    )
    ncde_state = torch.load('/Users/christopher/Documents/PTSD/NODE Model.nosync/Refitting/Individual Trained/2HL_128nodes_NCDE_Adam_ReLU+Tanh/NN_state_2HL_128nodes_NCDE_Control1_2000ITER_normed.txt')
    ncde_model.load_state_dict(ncde_state)

    dataset = NelsonData(['Control'], 'None', individual_number=1)
    loader = DataLoader(dataset)

    for (data, labels) in loader:
        node_pred_y = odeint(
            node_model, data[0,0,1:], data[...,0].squeeze(), method='rk4',
        )

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data[...,1:])
        ncde_pred_y = ncde_model(coeffs)

        print(f"{node_pred_y=}")
        print(f"{ncde_pred_y=}")


if __name__ == '__main__':
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

