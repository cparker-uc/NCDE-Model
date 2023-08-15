# File Name: testing.py
# Author: Christopher Parker
# Created: Fri Jul 21, 2023 | 04:30P EDT
# Last Modified: Mon Aug 14, 2023 | 12:02P EDT

"""Code for testing trained networks and saving summaries of classification
success rates into Excel spreadsheets"""

import os
from sre_constants import IN
import torch
import torchcde
import pandas as pd
from copy import copy

from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

from neural_cde import NeuralCDE
from get_data import NelsonData, AblesonData
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
MAX_ITR: int = 1
T_END: int = 0

# Define the device with which to train networks
DEVICE:torch.device = torch.device('cpu')


def test(hyperparameters: dict, virtual: bool=True,
         permutations: list=[], ctrl_range: list=[], mdd_range: list=[],
         ableson_pop: bool=False, plus_ableson_mdd: bool=False,
         toy_data: bool=False):
    """Run the test procedure given the order of test patients withheld from
    the training datasets"""

    # Loop over the constants passed in hyperparameters and set the values to
    #  the corresponding global variables
    for (key, val) in hyperparameters.items():
        globals()[key] = val

    # Set a flag to indicate if we are labeling the data by lab or by diagnosis
    by_lab = True if 'Nelson' in PATIENT_GROUPS else False

    # If the user didn't pass lists with the order of test patient
    #  permutations, we only test on one population
    if not permutations:
        test_single(virtual, ableson_pop, toy_data)
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
            mdd_combination=mdd_combination,
            patient_groups=PATIENT_GROUPS, by_lab=by_lab,
            plus_ableson_mdd=plus_ableson_mdd,
            test=True
        )
        model = NeuralCDE(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
            device=DEVICE, dropout=DROPOUT
        ).double()

        info = {
            'ctrl_num': ctrl_num,
            'control_combination': control_combination,
            'mdd_num': mdd_num,
            'mdd_combination': mdd_combination,
            'by_lab': by_lab,
        }
        with torch.no_grad():
            run_testing(model, loader, info)

def test_single(virtual: bool, ableson_pop: bool=False, toy_data: bool=False):
    """When we do not have a list of test patient groups, run a test on a
    single test population"""
    loader = load_data(
        virtual, POP_NUMBER, patient_groups=PATIENT_GROUPS,
        ableson_pop=ableson_pop, toy_data=toy_data, test=True
    )

    model = NeuralCDE(
        INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
        device=DEVICE, dropout=DROPOUT
    ).double()

    info = {
        'virtual': virtual,
        'ableson_pop': ableson_pop,
        'toy_data': toy_data
    }
    with torch.no_grad():
        run_testing(model, loader, info)


def load_data(virtual: bool=True, pop_number: int=0,
              control_combination: tuple=(),
              mdd_combination: tuple=(), patient_groups: list=[],
              by_lab: bool=False, ableson_pop: bool=False,
              plus_ableson_mdd: bool=False,
              toy_data: bool=False, test: bool=False):
    if not patient_groups:
        patient_groups = PATIENT_GROUPS
    if not virtual:
        dataset = NelsonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE
        ) if not ableson_pop else AblesonData(
            patient_groups=copy(patient_groups),
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
            plus_ableson_mdd=plus_ableson_mdd
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
        dataset=dataset, batch_size=2000, shuffle=False
    )
    return loader


def run_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)

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
    if control_combination:
        index = control_combination+mdd_combination
    elif ableson_pop:
        index = [i for i in range(50)]
    elif toy_data:
        index = [i for i in range(2000)]
    else:
        index = [i for i in range(5)]
        index = index+index

    tmp_index = []
    for i, entry in enumerate(index):
        if ableson_pop:
            if i < 37:
                t = PATIENT_GROUPS[0] + ' ' +  str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif toy_data:
            if i < 1000:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        else:
            if i < 5:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Set the directory name where the model state dictionaries are located
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
            f'{"Control" if not by_lab else "Nelson"} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1] if not by_lab else "Ableson"} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Network States (VPOP Training)/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Loop over the state dictionaries based on the number of iterations, from
    #  100 to MAX_ITR*100
    for itr in range(1,MAX_ITR+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_NCDE_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*100}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_NCDE_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*100}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_NCDE_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*100}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        state_filepath = os.path.join(directory, state_file)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)

        # Pandas Series to allow us to insert the number of iterations for each
        #  group of predicitons only in the first row of the group
        iterations = pd.Series((itr*100,), index=(index[0],))

        # Loop through the test patients
        for batch in loader:
            # Ensure the data is only CORT if CORT_ONLY, and that the data and
            #  labels are loaded into the proper device memory
            (data, label) = batch
            if CORT_ONLY:
                data = data[...,[0,2]]
            if toy_data and INPUT_CHANNELS == 3:
                data = data[...,[0,2,3]]
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            coeffs = torchcde.\
                hermite_cubic_coefficients_with_backward_differences(data)

            pred_y = model(coeffs).squeeze(-1)
            loss = binary_cross_entropy_with_logits(pred_y, label)

            # We need to run pred_y through a sigmoid layer to check for
            #  success and error because when training we use
            #  binary_cross_entropy_with_logits, which combines a sigmoid
            #  layer with BCE (improved performance over running sigmoid then
            #  BCE with torch)
            pred_y = torch.sigmoid(pred_y)
            error = torch.abs(label - pred_y)

            # Rounding the predicted y to see if it was successful
            rounded_y = torch.round(pred_y)
            success = [not y for y in torch.abs(label - rounded_y)]

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
    # Save the DataFrame to a file
    save_performance(performance_df, info)


def save_performance(performance_df: pd.DataFrame, info: dict):
    """Save the performance characteristics to an Excel spreadsheet"""
    # DataFrame to track performance (with a row for each network state at
    #  multiples of 100 iterations)
    # Access the necessary variables from the info dictionary
    virtual = info.get('virtual', True)
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    toy_data = info.get('toy_data', False)

    # Set the directory name based on which type of dataset was used for the
    #  training
    if not virtual:
        directory = (
            f'Classification Results/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Classification Results/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Classification Results/Augmented Data/'
            f'{"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Classification/'
            f'{"By Lab" if by_lab else "By Diagnosis"}/'
            f'{"Control" if not by_lab else "Nelson"} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1] if not by_lab else "Ableson"} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Classification Results/Augmented Data/'
            f'VPOP Classification/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Make sure the directory exists before we try to write anything
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set the filename for the network state_dict
    filename = (
        f'NCDE_{HDIM}nodes_'
        f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
        f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
        f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
        f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
        f'{NUM_PER_PATIENT}perPatient_'
        f'batchsize{BATCH_SIZE}_'
        f'{MAX_ITR}maxITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}'
        f'{"_byLab" if by_lab else ""}.xlsx'
    ) if virtual else (
        f'NN_state_{HDIM}nodes_NCDE_'
        f'Control_vs_{PATIENT_GROUPS[1]}_'
        f'batchsize{BATCH_SIZE}_'
        f'{MAX_ITR}maxITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}.xlsx'
    )

    with pd.ExcelWriter(os.path.join(directory, filename)) as writer:
        performance_df.to_excel(writer)


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

