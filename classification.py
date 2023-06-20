# File Name: galerkin_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Tue Jun 20, 2023 | 06:29P EDT

"Working on NCDE classification of augmented Nelson data"

# Network architecture parameters
INPUT_CHANNELS = 3
HDIM = 32
OUTPUT_CHANNELS = 1

# Training hyperparameters
ITERS = 300
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
LABEL_SMOOTHING = 0
DROPOUT = 0.2


# from IPython.core.debugger import set_trace
import os
import time
import torch
import torchcde
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from itertools import combinations, combinations_with_replacement
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
        self.dropout = torch.nn.Dropout(p=DROPOUT, inplace=True)

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
        self.dropout(z0)

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


def main(virtual=True, use_combinations=False):
    device = torch.device('cpu')

    # Ensure that these variables are not unbound, so that we can reference them
    #  when writing the setup file
    control_combination = None
    mdd_combination = None
    if not virtual:
        dataset = NelsonData(
            'Nelson TSST Individual Patient Data',
            patient_groups=PATIENT_GROUPS,
            normalize_standardize='standardize'
        )
    elif use_combinations:
        all_combos_control = combinations(range(15), 5)
        all_combos_melancholic = copy(all_combos_control)
        all_combos_atypical_neither = combinations(range(14), 4)

        random_control = torch.randint(0, 3002, (1,))
        random_melancholic = torch.randint(0, 3002, (1,))
        random_atypical_neither = torch.randint(0, 1000, (1,))

        control_combination = [combo for combo in all_combos_control][random_control]
        mdd_combination = [combo for combo in all_combos_melancholic][random_melancholic] if PATIENT_GROUP == 'Melancholic' else [combo for combo in all_combos_atypical_neither][random_atypical_neither]

        dataset = VirtualPopulation(
            patient_groups=PATIENT_GROUPS,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            num_patients=NUM_PATIENTS,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=False,
            label_smoothing=LABEL_SMOOTHING
        )
    else:
        dataset = VirtualPopulation(
            patient_groups=PATIENT_GROUPS,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            num_patients=NUM_PATIENTS,
            pop_number=POP_NUMBER,
            test=False,
            label_smoothing=LABEL_SMOOTHING
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
    if use_combinations:
        print(f'Starting Training on {PATIENT_GROUP} Test Groups: Control {control_combination} {PATIENT_GROUP} {mdd_combination}')
    else:
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
                f'{POP_NUMBER if not use_combinations else [control_combination, mdd_combination]}_'
                f'{NUM_PER_PATIENT}perPatient_'
                f'batchsize{BATCH_SIZE}_'
                f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                f'smoothing{LABEL_SMOOTHING}_'
                f'dropout{DROPOUT}.txt'
            )
            with open(f'Network States (VPOP Training)/NN_state_2HL_128nodes_'
                      f"NCDE_{METHOD}{'Virtual' if virtual else 'Real'}"
                      f'{PATIENT_GROUP}{POP_NUMBER if not use_combinations else [control_combination, mdd_combination]}_'
                      f'{NUM_PER_PATIENT}perPatient_'
                      f'batchsize{BATCH_SIZE}_'
                      f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                      f'smoothing{LABEL_SMOOTHING}_setup_'
                      f'dropout{DROPOUT}.txt',
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
                    f'Augmentation strategy={METHOD}\n'
                    f'Normalized/standardized={NORMALIZE_STANDARDIZE}\n'
                    'Number of virtual patients'
                    f' per real patient={NUM_PER_PATIENT}\n'
                    'Number of real patients sampled from each group='
                    f'{NUM_PATIENTS}\n'
                    f'Label smoothing factor={LABEL_SMOOTHING}\n'
                    f'Virtual population number used={POP_NUMBER}\n'
                    'Test Patient Combinations:\n'
                    f'Control: {control_combination}\n'
                    f'{PATIENT_GROUP}: {mdd_combination}\n'
                    f'Training batch size={BATCH_SIZE}\n'
                    'Training Results:\n'
                    f'Runtime={runtime}\n'
                    f'Loss over time={loss_over_time}'
                    # Currently not using:
                    # f'ATOL={ATOL}\nRTOL={RTOL}\n'
                )


def main_given_perms(permutations):
    device = torch.device('cpu')

    # Ensure that these variables are not unbound, so that we can reference them
    #  when writing the setup file
    for combo in combinations_with_replacement([0,1,2], 2):
        control_combination = tuple(permutations[0][combo[0]*5:(combo[0]+1)*5])
        mdd_combination = tuple(permutations[1][combo[1]*5:((combo[1]+1)*5) if (combo[1]+1)*5 < len(permutations[1]) else len(permutations[1])-1])

        dataset = VirtualPopulation(
            patient_groups=PATIENT_GROUPS,
            method=METHOD,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            num_patients=NUM_PATIENTS,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            fixed_perms=True,
            test=False,
            label_smoothing=LABEL_SMOOTHING
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
        print(f'Starting Training on {PATIENT_GROUP} Test Groups: Control {control_combination} {PATIENT_GROUP} {mdd_combination}')
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
                    f"{METHOD}Virtual{PATIENT_GROUP}"
                    f'Control{control_combination}_{PATIENT_GROUP}{mdd_combination}_'
                    f'{NUM_PER_PATIENT}perPatient_'
                    f'batchsize{BATCH_SIZE}_'
                    f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                    f'smoothing{LABEL_SMOOTHING}_'
                    f'dropout{DROPOUT}.txt'
                )
                with open(f'Network States (VPOP Training)/NN_state_2HL_128nodes_'
                          f"NCDE_{METHOD}Virtual"
                          f'{PATIENT_GROUP}_'
                          f'Control{control_combination}_'
                          f'{PATIENT_GROUP}{mdd_combination}_'
                          f'{NUM_PER_PATIENT}perPatient_'
                          f'batchsize{BATCH_SIZE}_'
                          f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
                          f'smoothing{LABEL_SMOOTHING}_setup_'
                          f'dropout{DROPOUT}.txt',
                          'w+') as file:
                    file.write(
                        f'Model Setup for {METHOD} '
                        "virtual "
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
                        f'Augmentation strategy={METHOD}\n'
                        f'Normalized/standardized={NORMALIZE_STANDARDIZE}\n'
                        'Number of virtual patients'
                        f' per real patient={NUM_PER_PATIENT}\n'
                        'Number of real patients sampled from each group='
                        f'{NUM_PATIENTS}\n'
                        f'Label smoothing factor={LABEL_SMOOTHING}\n'
                        f'Virtual population number used={POP_NUMBER}\n'
                        'Test Patient Combinations:\n'
                        f'Control: {control_combination}\n'
                        f'{PATIENT_GROUP}: {mdd_combination}\n'
                        f'Training batch size={BATCH_SIZE}\n'
                        'Training Results:\n'
                        f'Runtime={runtime}\n'
                        f'Loss over time={loss_over_time}'
                        # Currently not using:
                        # f'ATOL={ATOL}\nRTOL={RTOL}\n'
                    )


def test(method, patient_groups, num_per_patient, batch_size,
         normalize_standardize, num_patients, input_channels, hdim,
         output_channels, max_itr, control_combination=None,
         mdd_combination=None, pop_number=None, label_smoothing=0,
         state_dir='Network States (VPOP Training)'):
    patient_group = patient_groups[1]
    if control_combination and mdd_combination:
        dataset = VirtualPopulation(
            patient_groups=patient_groups,
            method=method,
            normalize_standardize=normalize_standardize,
            num_per_patient=num_per_patient,
            num_patients=num_patients,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=True,
            label_smoothing=label_smoothing
        )
    elif pop_number:
        dataset = VirtualPopulation(
            patient_groups=patient_groups,
            method=method,
            normalize_standardize=normalize_standardize,
            num_per_patient=num_per_patient,
            num_patients=num_patients,
            pop_number=pop_number,
            test=True,
        )
    else:
        print('Pop number or combination of patients required')
        return
    # Time points we need the solver to output
    t_eval = dataset[0][0][:,0].contiguous()
    loader = DataLoader(dataset, batch_size=len(dataset))

    # DataFrame to track performance (with a row for each network state at
    #  multiples of 100 iterations)
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

    for itr in range(1,max_itr+1):
        index = control_combination+mdd_combination
        tmp_index = []
        for i, entry in enumerate(index):
            if i < 5:
                t = 'Control ' + str(entry)
            else:
                t = patient_groups[1] + ' ' + str(entry)
            tmp_index.append(t)
        index = tuple(tmp_index)
        # Pandas Series to allow us to insert the number of iterations for each
        #  group of predicitons only in the first row of the group
        iterations = pd.Series((itr*100,), index=(index[0],))

        model = NeuralCDE(
            input_channels, hdim, output_channels, t_interval=t_eval
        ).double()
        try:
            state_dict = torch.load(
                os.path.join(state_dir, f'NN_state_2HL_128nodes_NCDE_'
                             f'{method}Virtual{patient_group}'
                             f'{pop_number if not (control_combination and mdd_combination) else [control_combination, mdd_combination]}_'
                             f'{num_per_patient}perPatient_'
                             f'batchsize{batch_size}_'
                             f'{itr*100}ITER_{normalize_standardize}_'
                             f'smoothing{label_smoothing}_'
                             f'dropout{DROPOUT}.txt')
            )
        except FileNotFoundError as e:
            print(f'Caught error: {e}')
            break

        model.load_state_dict(state_dict)

        for batch in loader:
            (data, label) = batch
            coeffs = torchcde.\
                hermite_cubic_coefficients_with_backward_differences(
                    data, t=t_eval
                )

            pred_y = model(coeffs).squeeze(-1)
            pred_y = torch.sigmoid(pred_y)

            loss = binary_cross_entropy_with_logits(pred_y, label)
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

    else:
        with pd.ExcelWriter(
            f'Classification Results/Augmented Data/NCDE_{method}Virtual'
            f'{patient_group}{pop_number if not mdd_combination else mdd_combination}'
            f"vsControl{'' if not control_combination else control_combination}"
            f'_classification_{num_per_patient}perPatient_batchsize{batch_size}_'
            f'{normalize_standardize}_smoothing{label_smoothing}.xlsx'
        ) as writer:
            performance_df.to_excel(writer)


def run_multiple_tests(patient_groups, pop_numbers, max_itr):
    with torch.no_grad():
        for groups in patient_groups:
            for pop_number in pop_numbers:
                test(
                    method='Uniform',
                    patient_groups=groups,
                    pop_number=pop_number,
                    num_per_patient=100,
                    batch_size=100,
                    normalize_standardize='Standardize',
                    num_patients=10,
                    input_channels=3,
                    hdim=32,
                    output_channels=1,
                    max_itr=max_itr,
                    label_smoothing=0.1,
                    state_dir='Network States (VPOP Training)/Uniform - 100 per Patient - 100 Batchsize - Standardized Data - 0.1 Smoothing'
                )


if __name__ == "__main__":
    # TRAIN WITH COMBINATIONS OF PATIENTS (which we record the patients
    #  left out for testing)
    # for PATIENT_GROUPS in [['Control', 'Atypical'], ['Control', 'Melancholic'],
    #                        ['Control', 'Neither']]:
    #     PATIENT_GROUP = PATIENT_GROUPS[1]
    #     main(use_combinations=True)

    # TRAIN WITH GROUP OF 3 POPULATIONS WHERE EACH PATIENT IS LEFT OUT IN ONE
    #  GROUP
    PATIENT_GROUPS = ['Control', 'Atypical']
    PATIENT_GROUP = PATIENT_GROUPS[1]
    perms = [
        # Control
        [13, 11,  8, 14,  6,  2, 12, 10,  5,  1,  0,  9,  3,  7,  4],
        # Atypical
        [ 0,  7, 12,  8, 11,  2,  6,  9,  3,  5,  4,  1, 13, 10]
    ]
    main_given_perms(perms)
    # TESTING WITH COMBINATIONS OF TEST PATIENTS
    # with torch.no_grad():
    #     test(
    #         method='Uniform',
    #         patient_groups=['Control', 'Melancholic'],
    #         num_per_patient=100,
    #         batch_size=100,
    #         normalize_standardize='Standardize',
    #         num_patients=10,
    #         input_channels=3,
    #         hdim=32,
    #         output_channels=1,
    #         max_itr=3,
    #         control_combination=(2, 3, 6, 11, 13),
    #         mdd_combination=(1, 9, 10, 12, 14),
    #     )

    # TEST WITH RANDOM TEST CASES
    # for POP_NUMBER in range(1,3):
    #     for PATIENT_GROUPS in [['Control', 'Atypical'], ['Control', 'Melancholic'],
    #                            ['Control', 'Neither']]:
    #         PATIENT_GROUP = PATIENT_GROUPS[1]
    #         main()

    # TESTING WITH RANDOM TEST CASES
    # run_multiple_tests(
    #     patient_groups=[['Control', 'Atypical'], ['Control', 'Melancholic'],
    #                     ['Control', 'Neither']],
    #     pop_numbers=[1,2,3],
    #     max_itr=3 # Pass the max number of 100 iteration steps that were run
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
