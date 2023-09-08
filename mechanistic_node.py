# File Name: mechanistic_node.py
# Author: Christopher Parker
# Created: Wed Jun 07, 2023 | 12:13P EDT
# Last Modified: Fri Sep 08, 2023 | 10:15P EDT

"""Root file for vector field prediction with NODEs (specifically with
mechanistic components added)"""

ITERS = 1000
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
BATCH_SIZE = 1
ATOL = 1e-7
RTOL = 1e-5
METHOD = 'dopri5'
PATIENT_GROUP = 'Atypical'
N_POINTS = 240
N_DIM = 4 # Use to set the number of variables in the system (either ACTH&CORT or all 4)
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
HDIM = 20

SAVE_FREQ = 100

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
from get_augmented_data import ToyDataset, NelsonVirtualPopulation
from get_data import NelsonData, AblesonData
from neural_ode import NeuralODE


def main():
    # Define the system of equations
    device = torch.device('cpu')
    model = NeuralODE(INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS).to(device)

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=DECAY)
    loss = nn.MSELoss()
    loss_over_time = []

    dataset = NelsonData(patient_groups=['Control'], normalize_standardize='Standardize')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    start_time = time.time()
    for i, batch in enumerate(loader):
        (label, _) = batch
        t_interval = label[...,0].squeeze()
        label = label[...,1:].squeeze()
        data = label[0,:]

        for itr in range(1, ITERS+1):
            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = odeint(
                model, data, t_interval, rtol=RTOL, atol=ATOL, method=METHOD
            ).squeeze()

            # Compute the loss based on the results
            output = loss(pred_y, label)
            loss_over_time.append(output.item())

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            output.backward()

            # Use the gradients calculated through backpropagation to adjust the
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}: loss = {output.item():.6f}")

            # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
            #  reset the learning rate and momentum
            if OPT_RESET is None:
                pass
            elif itr % OPT_RESET == 0:
                optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

            if itr % SAVE_FREQ == 0:
                runtime = time.time() - start_time
                print(f"Runtime: {runtime:.6f} seconds")
                torch.save(
                    model.state_dict(),
                    f'Refitting/NN_state_1HL_10nodes_mechanisticFeedback_paramOpt_Ki1p5_n5_controlPatient{i+1}_'
                    f'{itr}ITER_{OPT_RESET}optreset_normed.txt'
                )
                with open(f'Refitting/NN_state_1HL_10nodes_mechanisticFeedback_paramOpt_Ki1p5_n5_controlPatient{i+1}'
                          f'_{itr}ITER_{OPT_RESET}optreset_normed_setup.txt',
                          'w+') as file:
                    file.write(f'Model Setup for {PATIENT_GROUP} Patient {i+1}:\n')
                    file.write(
                        f'ITERS={itr}\nLEARNING_RATE={LR}\n'
                        f'OPT_RESET={OPT_RESET}\nATOL={ATOL}\nRTOL={RTOL}\n'
                        f'METHOD={METHOD}\n'
                        f'Input channels={INPUT_CHANNELS}\n'
                        f'Hidden channels={HDIM}\n'
                        f'Output channels={OUTPUT_CHANNELS}\n'
                        f'Runtime={runtime}\n'
                        f'Optimizer={optimizer}'
                        f'Loss over time={loss_over_time}'
                    )


def test(state, patient_group, patient_num, classifier=''):
    device = torch.device('cpu')
    model = ANN(INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS).to(device)
    model.load_state_dict(state)

    dataset = NelsonData(patient_groups=['Control'], normalize_standardize='Standardize')
    label, _ = dataset[patient_num-1]
    t_tensor = label[...,0].squeeze()
    label = label[...,1:].squeeze()
    y0 = label[0,:]
    dense_t_tensor = torch.linspace(0, 1, 10000)

    pred_y = odeint(model, y0, dense_t_tensor, atol=ATOL, rtol=RTOL, method=METHOD)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, label[:,0], 'o', label=f'Nelson {patient_group} Mean')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, label[:,1], 'o', label=f'Nelson {patient_group} Mean')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)

    return


if __name__ == "__main__":
    main()
    # state_file = torch.load(
    #     f'Refitting/NN_state_1HL_10nodes_mechanisticFeedback_paramOpt_Ki1p5_n5_controlPatient1_'
    #     f'1000ITER_Noneoptreset_normed.txt'
    # )
    # with torch.no_grad():
    #     test(state_file, 'Control', 1, '_redone_NODE')


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

