# File Name: training.py
# Author: Christopher Parker
# Created: Fri Jul 21, 2023 | 12:49P EDT
# Last Modified: Mon Aug 14, 2023 | 11:11P EDT

"""This file defines the functions used for network training. These functions
are used in classification.py"""

import os
import time
import torch
import torch.optim as optim
import torchcde
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader

from neural_cde import NeuralCDE
from get_data import AblesonData, NelsonData
from get_augmented_data import (FullVirtualPopulation,
                                FullVirtualPopulation_ByLab,
                                NelsonVirtualPopulation, ToyDataset)

# The constants listed here are to be defined in classification.py. The train()
#  function will iterate through the parameter names passed in the
#  parameter_dict argument and set values to these global variables for use in
#  all of the functions in this namespace
# Network architecture parameters
INPUT_CHANNELS: int = 0
HDIM: int = 0
OUTPUT_CHANNELS: int = 0

# Training hyperparameters
ITERS: int = 0
LR: float = 0.
DECAY: float = 0.
OPT_RESET: int = 0
ATOL: float = 0.
RTOL: float = 0.

# Training data selection parameters
PATIENT_GROUPS: list = []
METHOD: str = ''
NORMALIZE_STANDARDIZE: str = ''
NOISE_MAGNITUDE: float = 0.
NUM_PER_PATIENT: int = 0
POP_NUMBER: int = 0
BATCH_SIZE: int = 0
LABEL_SMOOTHING: float = 0.
DROPOUT: float = 0.
CORT_ONLY: bool = False
T_END: int = 0

# Define the device with which to train networks
DEVICE = torch.device('cpu')


def train(hyperparameters: dict, virtual: bool=True,
          permutations: list=[], ctrl_range: list=[], mdd_range: list=[],
          ableson_pop: bool=False, plus_ableson_mdd: bool=False,
          toy_data: bool=False):
    """Main function (called when script is executed directly"""
    # Loop over the constants passed in hyperparameters and set the values to
    #  the corresponding global variables
    for (key, val) in hyperparameters.items():
        globals()[key] = val

    # Set a flag to indicate if we are labeling the data by lab or by diagnosis
    by_lab = True if 'Nelson' in PATIENT_GROUPS else False

    # If the user didn't pass lists with the order of test patient
    #  permutations, we only run training on one population
    if not permutations:
        train_single(virtual, ableson_pop, toy_data)
        return

    # Create the list of all combinations of Control and MDD (or Nelson and
    #  Ableson) populations in ctrl_range and mdd_range
    loop_combos = [(i,j) for i in ctrl_range for j in mdd_range]
    for (ctrl_num, mdd_num) in loop_combos:
        control_combination = tuple(
            permutations[0][ctrl_num*5:(ctrl_num+1)*5]
        )
        mdd_combination = tuple(
            permutations[1][mdd_num*5:((mdd_num+1)*5)
                            if (mdd_num+1)*5 < len(permutations[1])
                            else len(permutations[1])]
        )
        # Load the dataset for training
        loader = load_data(
            control_combination=control_combination,
            mdd_combination=mdd_combination, by_lab=by_lab,
            plus_ableson_mdd=plus_ableson_mdd
        )
        model = NeuralCDE(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
            device=DEVICE, dropout=DROPOUT
        ).double()

        info = {
            'virtual': virtual,
            'ctrl_num': ctrl_num,
            'control_combination': control_combination,
            'mdd_num': mdd_num,
            'mdd_combination': mdd_combination,
            'by_lab': by_lab,
        }
        run_training(model, loader, info)


def train_single(virtual: bool, ableson_pop: bool=False, toy_data: bool=False):
    loader = load_data(
        virtual, POP_NUMBER, ableson_pop=ableson_pop, toy_data=toy_data
    )

    model = NeuralCDE(
        INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
        device=DEVICE, dropout=DROPOUT
    ).double()

    info = {
        'virtual': virtual,
        'toy_data': toy_data
    }
    run_training(model, loader, info)


def load_data(virtual: bool=True, pop_number: int=0,
              control_combination: tuple=(),
              mdd_combination: tuple=(), patient_groups: list=[],
              by_lab: bool=False, ableson_pop: bool=False,
              plus_ableson_mdd: bool=False, test: bool=False,
              toy_data: bool=False):
    if not patient_groups:
        patient_groups = PATIENT_GROUPS
    if not virtual:
        dataset = NelsonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE
        ) if not ableson_pop else AblesonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE
        )
    elif toy_data:
        dataset = ToyDataset(
            test=test,
            noise_magnitude=NOISE_MAGNITUDE,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            t_end=T_END
        )
    elif pop_number:
        dataset = NelsonVirtualPopulation(
            patient_groups=patient_groups,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            pop_number=POP_NUMBER,
            test=test,
            label_smoothing=LABEL_SMOOTHING,
        )
    elif patient_groups[1] not in ['MDD', 'Ableson']:
        dataset = NelsonVirtualPopulation(
            patient_groups=patient_groups,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=test,
            label_smoothing=LABEL_SMOOTHING,
            noise_magnitude=NOISE_MAGNITUDE,
            no_test_patients=ableson_pop,
            plus_ableson_mdd=plus_ableson_mdd,
        )
    elif by_lab:
        dataset = FullVirtualPopulation_ByLab(
            method=METHOD,
            noise_magnitude=NOISE_MAGNITUDE,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            # Since Nelson is treated as control for the purposes of
            #  labeling, we use control_combination for the Nelson data
            #  and mdd_combination for the Ableson data
            nelson_combination=control_combination,
            ableson_combination=mdd_combination,
            test=test,
        )
    else:
        dataset = FullVirtualPopulation(
            method=METHOD,
            noise_magnitude=NOISE_MAGNITUDE,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=test,
            label_smoothing=LABEL_SMOOTHING
        )
    loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    return loader


def run_training(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the training procedure for the given model and DataLoader"""
    virtual = info.get('virtual')
    control_combination = info.get('control_combination')
    mdd_combination = info.get('mdd_combination')
    toy_data = info.get('toy_data', False)

    # Print which population we are using to train
    if not virtual:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]} '
              f'vs {PATIENT_GROUPS[1]}')
    elif POP_NUMBER:
        print(f'Starting Training w/ {PATIENT_GROUPS[1]} '
              f'Population Number {POP_NUMBER}')
    else:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]} {control_combination}'
              f' vs {PATIENT_GROUPS[1]} {mdd_combination}')

    start_time = time.time()

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=DECAY
    )

    loss_over_time = []

    for itr in range(1, ITERS+1):
        training_epoch(itr, loader, model, optimizer, loss_over_time, toy_data=toy_data)

        # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
        #  reset the learning rate and momentum
        if not OPT_RESET:
            pass
        elif itr % OPT_RESET == 0:
            optimizer = optim.AdamW(
                model.parameters(), lr=LR, weight_decay=DECAY
            )

        if itr % 100 == 0:
            runtime = time.time() - start_time
            print(f"Runtime: {runtime:.6f} seconds")

            # Add the iteration number and runtime to the info dictionary and
            #  pass to save_network
            save_info = {
                'itr': itr,
                'runtime': runtime,
                'loss_over_time': loss_over_time
            }
            save_info.update(info)
            save_network(model, optimizer, save_info)


def training_epoch(itr: int, loader: DataLoader, model: NeuralCDE,
                   optimizer: optim.AdamW, loss_over_time: list, toy_data: bool=False):

    for j, (data, labels) in enumerate(loader):
        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,[0,2]]
        if toy_data and INPUT_CHANNELS == 3:
            data = data[...,[0,2,3]]
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        coeffs = torchcde.\
            hermite_cubic_coefficients_with_backward_differences(
                data
            )

        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # Compute the forward direction of the NODE
        pred_y = model(coeffs).squeeze(-1)

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
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")


def save_network(model: NeuralCDE, optimizer: optim.AdamW, info: dict):
    """Save the network state_dict and the training hyperparameters in the
    relevant directory"""
    # Access the necessary variables from the info dictionary
    virtual = info.get("virtual")
    ctrl_num = info.get("ctrl_num")
    control_combination = info.get("control_combination")
    mdd_num = info.get("mdd_num")
    mdd_combination = info.get("mdd_combination")
    by_lab = info.get("by_lab")
    itr = info.get("itr")
    runtime = info.get("runtime")
    loss_over_time = info.get("loss_over_time")
    toy_data = info.get('toy_data', False)

    # Set the directory name based on which type of dataset was used for the
    #  training
    if not virtual:
        directory = (
            f'Network States/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"By Lab" if by_lab else "By Diagnosis"}/'
            f'{PATIENT_GROUPS[0]} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1]} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Network States (VPOP Training)/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Make sure the directory exists before we try to write anything
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set the filename for the network state_dict
    filename = (
        f'NN_state_{HDIM}nodes_NCDE_'
        f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
        f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
        f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
        f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
        f'{NUM_PER_PATIENT}perPatient_'
        f'batchsize{BATCH_SIZE}_'
        f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}'
        f'{"_byLab" if by_lab else ""}.txt'
    ) if virtual else (
        f'NN_state_{HDIM}nodes_NCDE_'
        f'Control_vs_{PATIENT_GROUPS[1]}_'
        f'batchsize{BATCH_SIZE}_'
        f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}.txt'
    )
    # Add _setup to the filename before the .txt extension
    setup_filename = "".join([filename[:-4], "_setup", filename[-4:]])

    # Save the network state dictionary
    torch.save(
        model.state_dict(), os.path.join(directory, filename)
    )

    # Write the hyperparameters to the setup file
    with open(os.path.join(directory, setup_filename), 'w+') as file:
        file.write(
            f'Model Setup for {METHOD+"virtual " if virtual else ""}'
            '{PATIENT_GROUPS} Trained Network:\n\n'
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
            f'Augmentation strategy={METHOD}\n'
            f'Noise Magnitude={NOISE_MAGNITUDE}\n'
            f'Normalized/standardized={NORMALIZE_STANDARDIZE}\n'
            'Number of virtual patients'
            f' per real patient={NUM_PER_PATIENT}\n'
            f'Label smoothing factor={LABEL_SMOOTHING}\n'
            'Test Patient Combinations:\n'
            f'Control: {control_combination}\n'
            f'MDD: {mdd_combination}\n'
            f'Training batch size={BATCH_SIZE}\n'
            'Training Results:\n'
            f'Runtime={runtime}\n'
            f'Loss over time={loss_over_time}'
            # Currently not using:
            # f'ATOL={ATOL}\nRTOL={RTOL}\n'
        )


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

