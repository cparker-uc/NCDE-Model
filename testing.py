# File Name: testing.py
# Author: Christopher Parker
# Created: Fri Jul 21, 2023 | 04:30P EDT
# Last Modified: Mon Sep 18, 2023 | 06:38P EDT

"""Code for testing trained networks and saving summaries of classification
success rates into Excel spreadsheets"""

import os
import torch
import torch.nn as nn
import torchcde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchdiffeq import odeint_adjoint as odeint

from neural_cde import NeuralCDE
from neural_ode import NeuralODE
from ann import ANN
from rnn import RNN
from get_data import NelsonData, AblesonData, SriramSimulation
from get_augmented_data import (FullVirtualPopulation,
                                FullVirtualPopulation_ByLab,
                                NelsonVirtualPopulation, ToyDataset)

# The constants listed here are to be defined in classification.py. The train()
#  function will iterate through the parameter names passed in the
#  parameter_dict argument and set values to these global variables for use in
#  all of the functions in this namespace
# Network architecture parameters
NETWORK_TYPE: str = ''
INPUT_CHANNELS: int = 0
HDIM: int = 0
OUTPUT_CHANNELS: int = 0
# Only necessary for RNN
N_RECURS: int = 0
CLASSIFY: bool = True # For use in choosing between classification/prediction
MECHANISTIC: bool = False # Should the mechanistic components be included?

# Training hyperparameters
ITERS: int = 0
SAVE_FREQ: int = 0
LR: float = 0.
DECAY: float = 0.
OPT_RESET: int = 0
ATOL: float = 0.
RTOL: float = 0.

# Training data selection parameters
PATIENT_GROUPS: list = []
INDIVIDUAL_NUMBER: int = 0
METHOD: str = ''
NORMALIZE_STANDARDIZE: str = ''
NOISE_MAGNITUDE: float = 0.
IRREGULAR_T_SAMPLES = True
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
        loader, (t_steps, t_start, t_end) = load_data(
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            patient_groups=PATIENT_GROUPS, by_lab=by_lab,
            plus_ableson_mdd=plus_ableson_mdd,
            test=True
        )
        info = {
            'ctrl_num': ctrl_num,
            'control_combination': control_combination,
            'mdd_num': mdd_num,
            'mdd_combination': mdd_combination,
            'by_lab': by_lab,
            't_steps': t_steps,
            't_start': t_start,
            't_end': t_end,
        }
        model = model_init(info)
        with torch.no_grad():
            if NETWORK_TYPE in ('NCDE', 'NCDE_LBFGS'):
                classification_ncde_testing(model, loader, info) if CLASSIFY else prediction_ncde_testing(model, loader, info)
            elif NETWORK_TYPE == 'NODE':
                classification_node_testing(model, loader, info) if CLASSIFY else prediction_node_testing(model, loader, info)
            elif NETWORK_TYPE == 'ANN':
                classification_ann_testing(model, loader, info) if CLASSIFY else prediction_ann_testing(model, loader, info)
            elif NETWORK_TYPE == 'RNN':
                classification_rnn_testing(model, loader, info) if CLASSIFY else prediction_rnn_testing(model, loader, info)
            else:
                raise ValueError(
                    "NETWORK_TYPE must be one of: NCDE, NODE or ANN"
                )

def test_single(virtual: bool, ableson_pop: bool=False, toy_data: bool=False):
    """When we do not have a list of test patient groups, run a test on a
    single test population"""
    loader, (t_steps, t_start, t_end) = load_data(
        virtual, POP_NUMBER, patient_groups=PATIENT_GROUPS,
        ableson_pop=ableson_pop, toy_data=toy_data, test=True
    )

    info = {
        'virtual': virtual,
        'ableson_pop': ableson_pop,
        'toy_data': toy_data,
        't_steps': t_steps,
        't_start': t_start,
        't_end': t_end,
    }
    model = model_init(info)
    with torch.no_grad():
        if NETWORK_TYPE in ('NCDE', 'NCDE_LBFGS'):
            classification_ncde_testing(model, loader, info) if CLASSIFY else prediction_ncde_testing(model, loader, info)
        elif NETWORK_TYPE == 'NODE':
            classification_node_testing(model, loader, info) if CLASSIFY else prediction_node_testing(model, loader, info)
        elif NETWORK_TYPE == 'ANN':
            if MECHANISTIC:
                prediction_ann_testing_mechanistic(model, loader, info)
            else:
                classification_ann_testing(model, loader, info) if CLASSIFY else prediction_ann_testing(model, loader, info)
        elif NETWORK_TYPE == 'RNN':
            classification_rnn_testing(model, loader, info) if CLASSIFY else prediction_rnn_testing(model, loader, info)
        else:
            raise ValueError(
                "NETWORK_TYPE must be one of: NCDE, NODE or ANN"
            )


def load_data(virtual: bool=True, pop_number: int=0,
              control_combination: tuple=(),
              mdd_combination: tuple=(), patient_groups: list=[],
              by_lab: bool=False, ableson_pop: bool=False,
              plus_ableson_mdd: bool=False,
              toy_data: bool=False, test: bool=True):
    if not patient_groups:
        patient_groups = PATIENT_GROUPS
    if toy_data and not virtual:
        dataset = SriramSimulation(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE
        )
    elif not virtual:
        dataset = NelsonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            individual_number=INDIVIDUAL_NUMBER,
        ) if not ableson_pop else AblesonData(
            patient_groups=copy(patient_groups),
            normalize_standardize=NORMALIZE_STANDARDIZE
        )
    elif toy_data:
        dataset = ToyDataset(
            test=test,
            patient_groups=PATIENT_GROUPS,
            noise_magnitude=NOISE_MAGNITUDE,
            irregular_t_samples=IRREGULAR_T_SAMPLES,
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

    t_steps = len(dataset[0][0][...,0].squeeze())
    t = dataset[0][0][...,0]
    t_start = t[0]
    t_end = t[-1]
    loader = DataLoader(
        dataset=dataset, batch_size=2000, shuffle=False
    )
    return loader, (t_steps, t_start, t_end)


def model_init(info: dict):
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)
    t_start = info.get('t_start', 0)
    t_end = info.get('t_end', 1)

    if NETWORK_TYPE in ('NCDE', 'NCDE_LBFGS'):
        return NeuralCDE(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
            t_interval=torch.linspace(t_start, t_end, 1000),
            device=DEVICE, dropout=DROPOUT, prediction=not CLASSIFY
        ).double()
    if NETWORK_TYPE == 'NODE':
        return NeuralODE(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
            device=DEVICE,
        ).double()
    if NETWORK_TYPE == 'ANN':
        if MECHANISTIC:
            return ANN(
                INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS, dropout=DROPOUT,
                device=DEVICE
            )
        return ANN(
            INPUT_CHANNELS*t_steps, HDIM,
            OUTPUT_CHANNELS*t_steps,
            device=DEVICE,
        ).double()
    if NETWORK_TYPE == 'RNN':
        return RNN(
            INPUT_CHANNELS*t_steps, HDIM,
            N_RECURS, device=DEVICE,
        ).double()
    raise ValueError("NETWORK_TYPE unsupported")


def classification_ncde_testing(model: NeuralCDE, loader: DataLoader, info: dict):
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
        elif control_combination:
            if i < len(control_combination):
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
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{"By Lab" if by_lab else "By Diagnosis"}/'
            f'{"Control" if not by_lab else "Nelson"} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1] if not by_lab else "Ableson"} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Network States (VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Loop over the state dictionaries based on the number of iterations, from
    #  100 to MAX_ITR*100
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    print(f"{n_saves=}")
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_{"mechanistic_" if MECHANISTIC else ""}'
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
        iterations = pd.Series((itr*SAVE_FREQ,), index=(index[0],))

        # Loop through the test patients
        for batch in loader:
            # Ensure the data is only CORT if CORT_ONLY, and that the data and
            #  labels are loaded into the proper device memory
            (data, labels) = batch
            if CORT_ONLY:
                data = data[...,[0,2]]
            if toy_data and INPUT_CHANNELS == 3:
                data = data[...,[0,2,3]]
            data = data.to(DEVICE)
            labels = labels.to(DEVICE) if CLASSIFY else data
            coeffs = torchcde.\
                hermite_cubic_coefficients_with_backward_differences(data)

            pred_y = model(coeffs).squeeze(-1)
            output = loss(pred_y, labels)

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
                (output.item(),),
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


def classification_node_testing(model: NeuralCDE, loader: DataLoader, info: dict):
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
        elif control_combination:
            if i < len(control_combination):
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
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_{"mechanistic_" if MECHANISTIC else ""}'

                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        readout_filename = "".join(
            [state_file[:-4], '_readout', state_file[-4:]]
        )
        state_filepath = os.path.join(directory, state_file)
        readout_filepath = os.path.join(directory, readout_filename)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        readout_state_dict = torch.load(readout_filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)
        readout = nn.Linear(OUTPUT_CHANNELS, 1)
        readout.load_state_dict(readout_state_dict)

        # Pandas Series to allow us to insert the number of iterations for each
        #  group of predicitons only in the first row of the group
        iterations = pd.Series((itr*100,), index=(index[0],))

        # Loop through the test patients
        for (data, labels) in loader:
            t_eval = data[0,:,0].view(-1)
            # Ensure we have assigned the data and labels to the correct
            #  processing device
            if CORT_ONLY:
                # If we are only using CORT, we can discard the middle column as it
                #  contains the ACTH concentrations
                data = data[...,2]
            if toy_data and INPUT_CHANNELS == 2:
                data = data[...,[2,3]]
            data = data.to(DEVICE)
            labels = labels.to(DEVICE) if CLASSIFY else data
            y0 = data[:,0,1:]

            # Compute the forward direction of the NODE
            pred_y = odeint(
                model, y0, t_eval
            )
            # We need to take the output_channels down to a single output, then
            #  we only need the last value (so the value after the entire depth of
            #  the network)
            # We squeeze to remove the extraneous dimension of the output so that
            #  it matches the shape of the labels
            pred_y = readout(pred_y)[-1].squeeze(-1)
            output = loss(pred_y, labels)

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
                (output.item(),),
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


def classification_ann_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)

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
        elif control_combination:
            if i < len(control_combination):
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
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_{"mechanistic_" if MECHANISTIC else ""}'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
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
        for (data, labels) in loader:
            # Check how many patients are in the batch (as it may be less than
            #  BATCH_SIZE if it's the last batch)
            batch_size_ = data.size(0)
            # Ensure the data is only CORT if CORT_ONLY, and that the data and
            #  labels are loaded into the proper device memory
            if CORT_ONLY:
                data = data[...,2]
            elif toy_data and INPUT_CHANNELS == 2:
                data = data[...,[2,3]]
            else:
                data = data[...,1:]
            data = data.to(DEVICE)
            labels = labels.to(DEVICE) if CLASSIFY else data

            # We need to reshape the data to fit the ANN properly
            pred_y = model(
                data.reshape(
                    batch_size_,
                    INPUT_CHANNELS*t_steps)
            ).squeeze(-1).reshape(-1, labels.size(1), labels.size(2))
            output = loss(pred_y, labels)

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
                (output.item(),),
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


def classification_rnn_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)

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
        elif control_combination:
            if i < len(control_combination):
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
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_{"mechanistic_" if MECHANISTIC else ""}'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        readout_filename = "".join(
            [state_file[:-4], '_readout', state_file[-4:]]
        )
        state_filepath = os.path.join(directory, state_file)
        readout_filepath = os.path.join(directory, readout_filename)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        readout_state_dict = torch.load(readout_filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)
        readout = nn.Linear(HDIM, 1).double()
        readout.load_state_dict(readout_state_dict)

        # Pandas Series to allow us to insert the number of iterations for each
        #  group of predicitons only in the first row of the group
        iterations = pd.Series((itr*100,), index=(index[0],))

        # Loop through the test patients
        for (data, labels) in loader:
            # Check how many patients are in the batch (as it may be less than
            #  BATCH_SIZE if it's the last batch)
            batch_size_ = data.size(0)
            # Ensure the data is only CORT if CORT_ONLY, and that the data and
            #  labels are loaded into the proper device memory
            if CORT_ONLY:
                data = data[...,2]
            elif toy_data and INPUT_CHANNELS == 2:
                data = data[...,[2,3]]
            else:
                data = data[...,1:]
            data = data.to(DEVICE)
            labels = labels.to(DEVICE) if CLASSIFY else data

            # We need to reshape the data to fit the ANN properly
            pred_y = model(
                data.reshape(
                    batch_size_,
                    INPUT_CHANNELS*t_steps)
            )
            pred_y = readout(pred_y).squeeze(-1).reshape(-1, labels.size(1), labels.size(2))

            output = loss(pred_y, labels)

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
                (output.item(),),
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


def prediction_ncde_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)

    # Set up the index numbers to match the patient numbers of the test
    #  patients (or just 1-5 for Control and the same for MDD if no test
    #  combinations)
    if control_combination:
        index = control_combination+mdd_combination
    elif ableson_pop:
        index = [i for i in range(50)]
    elif toy_data:
        index = [i for i in range(2000)]
    elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
        index = [INDIVIDUAL_NUMBER]
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
        elif control_combination:
            if i < len(control_combination):
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            t = PATIENT_GROUPS[0] + ' ' + str(entry)
        else:
            if i < 5:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Set the directory name where the model state dictionaries are located
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0]}/'
        )
    elif not virtual:
        directory = (
            f'Network States/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{"By Lab" if by_lab else "By Diagnosis"}/'
            f'{"Control" if not by_lab else "Nelson"} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1] if not by_lab else "Ableson"} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Network States (VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Loop over the state dictionaries based on the number of iterations, from
    #  100 to MAX_ITR*100
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_{"mechanistic_" if MECHANISTIC else ""}'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
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
        iterations = pd.Series((itr*SAVE_FREQ,), index=(index[0],))

        # Loop through the test patients
        for (data, labels) in loader:
            for i, pt in enumerate(data):
                if INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
                    pt = pt.double().view(1,t_steps,-1)
                # Ensure the data is only CORT if CORT_ONLY, and that the data and
                #  labels are loaded into the proper device memory
                data_orig = pt
                if CORT_ONLY:
                    pt = pt[...,[0,2]]
                if toy_data and INPUT_CHANNELS == 3:
                    pt = pt[...,[0,2,3]]
                pt = pt.to(DEVICE)
                labels = labels.to(DEVICE) if CLASSIFY else pt
                coeffs = torchcde.\
                    hermite_cubic_coefficients_with_backward_differences(data)

                pred_y = model(coeffs).squeeze(-1)

                if i < len(index)/2:
                    patient_id = f'Control Patient {index[i]}'
                else:
                    patient_id = f'MDD Patient {index[i]}'

                # Graph the results
                graph_results(pred_y, data_orig, patient_id, itr, info)


def prediction_node_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)

    # Initialize the parameters if we are doing mechanistic loss
    if MECHANISTIC:
        param_init(model)

    # Set up the index numbers to match the patient numbers of the test
    #  patients (or just 1-5 for Control and the same for MDD if no test
    #  combinations)
    if control_combination:
        index = control_combination+mdd_combination
    elif ableson_pop:
        index = [i for i in range(50)]
    elif toy_data and len(PATIENT_GROUPS) == 1:
        index = [i for i in range(1000)]
    elif toy_data:
        index = [i for i in range(2000)]
    elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
        index = [INDIVIDUAL_NUMBER]
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
        elif toy_data and len(PATIENT_GROUPS) == 1:
            t = PATIENT_GROUPS[0] + ' ' + str(entry)
        elif toy_data:
            if i < 1000:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif control_combination:
            if i < len(control_combination):
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            t = PATIENT_GROUPS[0] + ' ' + str(entry)
        else:
            if i < 5:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Set the directory name where the model state dictionaries are located
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0]}/'
        )
    elif not virtual:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_{"mechanistic_" if MECHANISTIC else ""}'

                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        elif len(PATIENT_GROUPS) == 1:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        readout_filename = "".join(
            [state_file[:-4], '_readout', state_file[-4:]]
        )
        state_filepath = os.path.join(directory, state_file)
        readout_filepath = os.path.join(directory, readout_filename)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)

        # Loop through the test patients
        for (data, labels) in loader:
            for i, pt in enumerate(data):
                if INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
                    pt = pt.double().view(1,t_steps,INPUT_CHANNELS+1)
                elif toy_data and not virtual:
                    pt = pt.double().view(1, t_steps, INPUT_CHANNELS+1)
                data_orig = pt
                t_eval = pt[0,:,0].view(-1)
                dense_t_eval = torch.linspace(t_eval[0], t_eval[-1], 1000, dtype=torch.float64)
                # Ensure we have assigned the data and labels to the correct
                #  processing device
                if CORT_ONLY:
                    # If we are only using CORT, we can discard the middle column as it
                    #  contains the ACTH concentrations
                    pt = pt[...,2]
                if toy_data and INPUT_CHANNELS == 2:
                    pt = pt[...,[2,3]]
                pt = pt.to(DEVICE)
                labels = labels.to(DEVICE) if CLASSIFY else pt
                y0 = pt[:,0,1:]

                # Compute the forward direction of the NODE
                pred_y = odeint(
                    model, y0, dense_t_eval
                ).squeeze(-1)

                if i < len(index)/2:
                    patient_id = f'Control Patient {index[i]}'
                else:
                    patient_id = f'MDD Patient {index[i]}'

                # Graph the results
                graph_results(pred_y, dense_t_eval, data_orig, patient_id,
                              itr, info)


def prediction_ann_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)

    # Set up the index numbers to match the patient numbers of the test
    #  patients (or just 1-5 for Control and the same for MDD if no test
    #  combinations)
    if control_combination:
        index = control_combination+mdd_combination
    elif ableson_pop:
        index = [i for i in range(50)]
    elif toy_data:
        index = [i for i in range(2000)]
    elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
        index = [INDIVIDUAL_NUMBER]
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
        elif control_combination:
            if i < len(control_combination):
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            t = PATIENT_GROUPS[0] + ' ' + str(entry)
        else:
            if i < 5:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Set the directory name where the model state dictionaries are located
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0]}/'
        )
    elif not virtual:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if toy_data:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}_'
                f'{PATIENT_GROUPS[1]+"_" if len(PATIENT_GROUPS) > 1 else ""}'
                f'batchsize{BATCH_SIZE}_'
                f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}_{T_END}hrs'
                f'{"_irregularSamples" if IRREGULAR_T_SAMPLES else ""}.txt'
            )
        elif virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_{"mechanistic_" if MECHANISTIC else ""}'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
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

        # Loop through the test patients
        for i, (data, labels) in enumerate(loader):
            for i, pt in enumerate(data):
                if INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
                    pt = pt.double().view(1,t_steps,-1)
                data_orig = pt
                # Here, we only ever have a batch size of 1 for plotting
                batch_size_ = 1
                # Ensure the data is only CORT if CORT_ONLY, and that the data and
                #  labels are loaded into the proper device memory
                if CORT_ONLY:
                    pt = pt[...,2]
                elif toy_data and INPUT_CHANNELS == 2:
                    pt = pt[...,[2,3]]
                else:
                    pt = pt[...,1:]
                pt = pt.to(DEVICE)
                labels = labels.to(DEVICE) if CLASSIFY else pt

                # We need to reshape the data to fit the ANN properly
                pred_y = model(
                    pt.reshape(
                        batch_size_,
                        INPUT_CHANNELS*t_steps)
                ).squeeze(-1).reshape(-1, labels.size(1))

                if i < len(index)/2:
                    patient_id = f'Control Patient {index[i]}'
                else:
                    patient_id = f'MDD Patient {index[i]}'

                # Graph the results
                graph_results(pred_y, dense_t_eval, data_orig, patient_id,
                              itr, info)


def prediction_ann_testing_mechanistic(model: NeuralCDE, loader: DataLoader,
                                       info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)

    # Initialize the parameters if we are doing mechanistic loss
    param_init(model)

    # Set up the index numbers to match the patient numbers of the test
    #  patients (or just 1-5 for Control and the same for MDD if no test
    #  combinations)
    if control_combination:
        index = control_combination+mdd_combination
    elif ableson_pop:
        index = [i for i in range(50)]
    elif toy_data and len(PATIENT_GROUPS) == 1:
        index = [i for i in range(1000)]
    elif toy_data:
        index = [i for i in range(2000)]
    elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
        index = [INDIVIDUAL_NUMBER]
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
        elif control_combination:
            if i < len(control_combination):
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            t = PATIENT_GROUPS[0] + ' ' + str(entry)
        else:
            if i < 5:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Set the directory name where the model state dictionaries are located
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0] if not toy_data else "Toy Dataset"}/'
        )
    elif not virtual:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if toy_data:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}_'
                f'{PATIENT_GROUPS[1]+"_" if len(PATIENT_GROUPS) > 1 else ""}'
                f'batchsize{BATCH_SIZE}_'
                f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}_{T_END}hrs'
                f'{"_irregularSamples" if IRREGULAR_T_SAMPLES else ""}.txt'
            )
        elif virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_{"mechanistic_" if MECHANISTIC else ""}'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
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
        domain = torch.linspace(0, T_END, 1000).view(-1,1)

        # Loop through the test patients
        for i, (data, labels) in enumerate(loader):
            for i, pt in enumerate(data):
                if INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
                    pt = pt.double().view(1,t_steps,-1)

                # We need to reshape the data to fit the ANN properly
                pred_y = model(domain).squeeze(-1).reshape(domain.size(0), OUTPUT_CHANNELS)

                if i < len(index)/2:
                    patient_id = f'Control Patient {index[i]}'
                else:
                    patient_id = f'MDD Patient {index[i]}'

                # Graph the results
                graph_results(pred_y, domain, pt, patient_id,
                              itr, info)


def prediction_rnn_testing(model: NeuralCDE, loader: DataLoader, info: dict):
    """Run the testing procedure for the given model and DataLoader"""
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    virtual = info.get('virtual', True)
    ableson_pop = info.get('ableson_pop', False)
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)

    # Set up the index numbers to match the patient numbers of the test
    #  patients (or just 1-5 for Control and the same for MDD if no test
    #  combinations)
    if control_combination:
        index = control_combination+mdd_combination
    elif ableson_pop:
        index = [i for i in range(50)]
    elif toy_data:
        index = [i for i in range(2000)]
    elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
        index = [INDIVIDUAL_NUMBER]
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
        elif control_combination:
            if i < len(control_combination):
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            t = PATIENT_GROUPS[0] + ' ' + str(entry)
        else:
            if i < 5:
                t = PATIENT_GROUPS[0] + ' ' + str(entry)
            else:
                t = PATIENT_GROUPS[1] + ' ' + str(entry)
        tmp_index.append(t)
    index = tuple(tmp_index)

    # Set the directory name where the model state dictionaries are located
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0]}/'
        )
    elif not virtual:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Network States ({"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
    n_saves = int(np.floor(MAX_ITR/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        if virtual:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_{"mechanistic_" if MECHANISTIC else ""}'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
                f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}'
                f'{"_byLab" if by_lab else ""}.txt'
            )
        elif ableson_pop:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
                f'{PATIENT_GROUPS[0]+str(control_combination)}_'
                f'{PATIENT_GROUPS[1]+str(mdd_combination)}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        else:
            state_file = (
                f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
                f'Control_vs_{PATIENT_GROUPS[1]}_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
        readout_filename = "".join(
            [state_file[:-4], '_readout', state_file[-4:]]
        )
        state_filepath = os.path.join(directory, state_file)
        readout_filepath = os.path.join(directory, readout_filename)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        readout_state_dict = torch.load(readout_filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)
        readout = nn.Linear(HDIM, 1).double()
        readout.load_state_dict(readout_state_dict)

        # Loop through the test patients
        for (data, labels) in loader:
            for i, pt in enumerate(data):
                if INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
                    pt = pt.double().view(1,t_steps,-1)
                data_orig = pt
                # Here, we only ever have a batch size of 1 for plotting
                batch_size_ = 1
                # Ensure the data is only CORT if CORT_ONLY, and that the data and
                #  labels are loaded into the proper device memory
                if CORT_ONLY:
                    pt = pt[...,2]
                elif toy_data and INPUT_CHANNELS == 2:
                    pt = pt[...,[2,3]]
                else:
                    pt = pt[...,1:]
                pt = pt.to(DEVICE)
                labels = labels.to(DEVICE) if CLASSIFY else pt

                # We need to reshape the data to fit the ANN properly
                pred_y = model(
                    pt.reshape(
                        batch_size_,
                        INPUT_CHANNELS*t_steps)
                )
                pred_y = readout(pred_y).squeeze(-1).reshape(
                    -1, labels.size(1),
                )

                if i < len(index)/2:
                    patient_id = f'Control Patient {index[i]}'
                else:
                    patient_id = f'MDD Patient {index[i]}'

                # Graph the results
                graph_results(pred_y, dense_t_eval, data_orig, patient_id,
                              itr, info)


def param_init(model: nn.Module):
    """Initialize the parameters for the mechanistic loss, and set them to
    require gradient"""
    k_stress = torch.nn.Parameter(torch.tensor(13.7), requires_grad=True)
    Ki = torch.nn.Parameter(torch.tensor(1.6), requires_grad=True)
    VS3 = torch.nn.Parameter(torch.tensor(3.25), requires_grad=True)
    Km1 = torch.nn.Parameter(torch.tensor(1.74), requires_grad=True)
    KP2 = torch.nn.Parameter(torch.tensor(8.3), requires_grad=True)
    VS4 = torch.nn.Parameter(torch.tensor(0.907), requires_grad=False)
    Km2 = torch.nn.Parameter(torch.tensor(0.112), requires_grad=False)
    KP3 = torch.nn.Parameter(torch.tensor(0.945), requires_grad=False)
    VS5 = torch.nn.Parameter(torch.tensor(0.00535), requires_grad=False)
    Km3 = torch.nn.Parameter(torch.tensor(0.0768), requires_grad=False)
    Kd1 = torch.nn.Parameter(torch.tensor(0.00379), requires_grad=False)
    Kd2 = torch.nn.Parameter(torch.tensor(0.00916), requires_grad=False)
    Kd3 = torch.nn.Parameter(torch.tensor(0.356), requires_grad=False)
    n1 = torch.nn.Parameter(torch.tensor(5.43), requires_grad=True)
    n2 = torch.nn.Parameter(torch.tensor(5.1), requires_grad=True)
    Kb = torch.nn.Parameter(torch.tensor(0.0202), requires_grad=True)
    Gtot = torch.nn.Parameter(torch.tensor(3.28), requires_grad=True)
    VS2 = torch.nn.Parameter(torch.tensor(0.0509), requires_grad=True)
    K1 = torch.nn.Parameter(torch.tensor(0.645), requires_grad=True)
    Kd5 = torch.nn.Parameter(torch.tensor(0.0854), requires_grad=True)

    params = {
        'k_stress': k_stress,
        'Ki': Ki,
        'VS3': VS3,
        'Km1': Km1,
        'KP2': KP2,
        'VS4': VS4,
        'Km2': Km2,
        'KP3': KP3,
        'VS5': VS5,
        'Km3': Km3,
        'Kd1': Kd1,
        'Kd2': Kd2,
        'Kd3': Kd3,
        'n1': n1,
        'n2': n2,
        'Kb': Kb,
        'Gtot': Gtot,
        'VS2': VS2,
        'K1': K1,
        'Kd5': Kd5
    }

    for key, val in params.items():
        model.register_parameter(key, val)

    return params


def loss(pred_y, labels):
    """To compute the loss with potential mechanistic components"""
    if CLASSIFY:
        return binary_cross_entropy_with_logits(pred_y, labels)

    if MECHANISTIC:
        pass

    return mse_loss(pred_y, labels)


def save_performance(performance_df: pd.DataFrame, info: dict):
    """Save the performance characteristics to an Excel spreadsheet"""
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
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/Augmented Data/'
            f'{"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Classification/'
            f'{"By Lab" if by_lab else "By Diagnosis"}/'
            f'{"Control" if not by_lab else "Nelson"} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1] if not by_lab else "Ableson"} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/Augmented Data/'
            f'VPOP Classification/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Make sure the directory exists before we try to write anything
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set the filename for the network state_dict
    filename = (
        f'{NETWORK_TYPE}_{HDIM}nodes_'
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
        f'{NETWORK_TYPE}_{HDIM}nodes_'
        f'Control_vs_{PATIENT_GROUPS[1]}_'
        f'batchsize{BATCH_SIZE}_'
        f'{MAX_ITR}maxITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}.xlsx'
    )

    with pd.ExcelWriter(os.path.join(directory, filename)) as writer:
        performance_df.to_excel(writer)


def graph_results(pred_y: torch.Tensor, pred_t: torch.Tensor,
                  data: torch.Tensor, patient_id: str,
                  save_num: int, info: dict):
    # Access the necessary variables from the info dictionary
    virtual = info.get('virtual', True)
    ctrl_num = info.get('ctrl_num')
    control_combination = info.get('control_combination')
    mdd_num = info.get('mdd_num')
    mdd_combination = info.get('mdd_combination')
    by_lab = info.get('by_lab')
    toy_data = info.get('toy_data', False)

    # Convert the save number to the number of iterations
    itr = save_num*SAVE_FREQ

    # Set the directory name based on which type of dataset was used for the
    #  training
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0] if not toy_data else "Toy Dataset"}/'
        )
    elif not virtual:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
        )
    elif toy_data:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Toy Dataset/'
        )
    elif not POP_NUMBER:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/Augmented Data/'
            f'{"Full" if not CORT_ONLY else "CORT ONLY"} VPOP Classification/'
            f'{"By Lab" if by_lab else "By Diagnosis"}/'
            f'{"Control" if not by_lab else "Nelson"} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1] if not by_lab else "Ableson"} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Results/'
            f'{"Classification" if CLASSIFY else "Prediction"}/Augmented Data/'
            f'VPOP Classification/'
            f'Control vs {PATIENT_GROUPS[1]} Population {POP_NUMBER}/'
        )

    # Make sure the directory exists before we try to write anything
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set the filename for the network state_dict
    if toy_data:
        filename = (
            f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
            f'{PATIENT_GROUPS[0]}_'
            f'{PATIENT_GROUPS[1]+"_" if len(PATIENT_GROUPS) > 1 else ""}'
            f'batchsize{BATCH_SIZE}_'
            f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}_{T_END}hrs'
            f'{"_irregularSamples" if IRREGULAR_T_SAMPLES else ""}_'
        )
    elif not virtual and len(PATIENT_GROUPS) == 1:
        filename = (
            f'{NETWORK_TYPE}_{HDIM}nodes_'
            f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
            f'batchsize{BATCH_SIZE}_'
            f'{MAX_ITR}maxITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}_'
        )
    elif not virtual:
        filename = (
            f'{NETWORK_TYPE}_{HDIM}nodes_'
            f'Control_vs_{PATIENT_GROUPS[1]}_'
            f'batchsize{BATCH_SIZE}_'
            f'{MAX_ITR}maxITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}_'
        )
    else:
        filename = (
            f'{NETWORK_TYPE}_{HDIM}nodes_'
            f'{METHOD}{NOISE_MAGNITUDE}Virtual_'
            f'{PATIENT_GROUPS[0]+str(control_combination) if not POP_NUMBER else ""}_'
            f'{PATIENT_GROUPS[1]+str(mdd_combination) if not POP_NUMBER else ""}_'
            f'{"Control vs "+PATIENT_GROUPS[1]+" Population"+str(POP_NUMBER) if POP_NUMBER else ""}'
            f'{NUM_PER_PATIENT}perPatient_'
            f'batchsize{BATCH_SIZE}_'
            f'{MAX_ITR}maxITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}_'
            f'{"_byLab" if by_lab else ""}_'
        )

    # pred_y = pred_y.view(data.size(1), data.size(2)-1)
    pred_y = pred_y.squeeze()
    fig, axes = plt.subplots(nrows=OUTPUT_CHANNELS, figsize=(10,10))
    for idx,ax in enumerate(range(OUTPUT_CHANNELS)):
        axes[ax].plot(data[...,0].squeeze(), data[...,idx+1].squeeze(), 'o', label=patient_id)
        axes[ax].plot(
            pred_t, pred_y[:,idx], label=f'Predicted y ({itr} iterations)'
        )
        axes[ax].set(title=patient_id, xlabel='Time (normalized)', ylabel='Concentration')
        axes[ax].legend(fancybox=True, shadow=True, loc='upper right')

    plt.savefig(os.path.join(directory, filename+patient_id+'.png'), dpi=300)
    plt.close(fig)


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

