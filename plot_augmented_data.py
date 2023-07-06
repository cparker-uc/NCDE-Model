# File Name: plot_augmented_data.py
# Author: Christopher Parker
# Created: Thu Jun 22, 2023 | 12:34P EDT
# Last Modified: Mon Jun 26, 2023 | 12:29P EDT

"""Uses Matplotlib to plot augmented datasets for examination"""

# Training data selection parameters
PATIENT_GROUPS = ['Control', 'Melancholic']
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'Standardize'
NUM_PER_PATIENT = 100
NUM_PATIENTS = 10
BATCH_SIZE = 1
LABEL_SMOOTHING = 0

import torch
import numpy as np
import matplotlib.pyplot as plt
from get_nelson_data import VirtualPopulation
from torch.utils.data import DataLoader

PERMUTATIONS = [
    # Control
    [13, 11,  8, 14,  6,  2, 12, 10,  5,  1,  0,  9,  3,  7,  4],
    # Atypical
    [ 0,  7, 12,  8, 11,  2,  6,  9,  3,  5,  4,  1, 13, 10],
    # Melancholic
    [10, 13,  4,  2,  3, 12, 14,  6,  8,  9,  5,  7,  0, 11,  1],
    # Neither
    [ 5, 12,  7, 13, 11,  9,  4,  1,  0,  2,  3, 10,  6,  8]
]


def plot_test_patients_group_comparison_mean_lines(patient_groups, control_combination, mdd_combination):
    dataset = VirtualPopulation(
        patient_groups=patient_groups,
        method=METHOD,
        normalize_standardize=NORMALIZE_STANDARDIZE,
        num_per_patient=NUM_PER_PATIENT,
        num_patients=NUM_PATIENTS,
        control_combination=control_combination,
        mdd_combination=mdd_combination,
        fixed_perms=True,
        test=True,
        label_smoothing=LABEL_SMOOTHING
    )
    loader = DataLoader(dataset=dataset)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
    control_mean = torch.mean(dataset[:5][0], 0)
    mdd_mean = torch.mean(dataset[5:][0], 0)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
    ax1.plot(control_mean[:,0], control_mean[:,1], color='orange', label=f'Control Set {combo[0]}')
    ax1.plot(mdd_mean[:,0], mdd_mean[:,1], color='blue', label=f'{PATIENT_GROUPS[1]} Set {combo[1]}')
    ax1.legend(shadow=True, fancybox=True, loc='upper right')
    ax2.plot(control_mean[:,0], control_mean[:,2], color='orange', label=f'Control Set {combo[0]}')
    ax2.plot(mdd_mean[:,0], mdd_mean[:,2], color='blue', label=f'{PATIENT_GROUPS[1]} Set {combo[1]}')
    ax2.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Group-Comparison_Mean_Lines_Control_{PATIENT_GROUPS[1]}_augmented_combination{combo}_comparison-graph.png', dpi=300)
    plt.close(fig)


def plot_test_patients_group_comparison_indiv_lines(patient_groups, control_combination, mdd_combination):
    dataset = VirtualPopulation(
        patient_groups=patient_groups,
        method=METHOD,
        normalize_standardize=NORMALIZE_STANDARDIZE,
        num_per_patient=NUM_PER_PATIENT,
        num_patients=NUM_PATIENTS,
        control_combination=control_combination,
        mdd_combination=mdd_combination,
        fixed_perms=True,
        test=True,
        label_smoothing=LABEL_SMOOTHING
    )
    loader = DataLoader(dataset=dataset)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
    control_count = 0
    mdd_count = 0
    for (data, label) in loader:
        data = data[0]
        control = (label.item() == 0)
        control_count = (control_count + 1 if control else control_count)
        mdd_count = (mdd_count + 1 if not control else mdd_count)
        ax1.plot(data[...,0], data[...,1], color=f'{"orange" if control else "blue"}',
                 label=f'{"Control" if control else patient_groups[1]} Set {control_combination[control_count-1] if control else mdd_combination[mdd_count-1]}')
        ax1.set(xlabel='Time (normalized)', ylabel='CORT Concentration (standardized)', title='ACTH Comparison')
        ax1.legend(shadow=True, fancybox=True, loc='upper right')

        ax2.plot(data[...,0], data[...,2], color=f'{"orange" if control else "blue"}',
                 label=f'{"Control" if control else patient_groups[1]} Set {control_combination[control_count-1] if control else mdd_combination[mdd_count-1]}')
        ax2.set(xlabel='Time (normalized)', ylabel='ACTH Concentration (standardized)', title='Cortisol Comparison')
        ax2.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Group-Comparison_Individual_Lines_Control_{PATIENT_GROUPS[1]}_augmented_combination{combo}_comparison-graph.png', dpi=300)
    plt.close(fig)


def plot_test_patients_by_group(patient_groups, control_combination,
                                mdd_combination, plot_control=True):
    dataset = VirtualPopulation(
        patient_groups=patient_groups,
        method=METHOD,
        normalize_standardize=NORMALIZE_STANDARDIZE,
        num_per_patient=NUM_PER_PATIENT,
        num_patients=NUM_PATIENTS,
        control_combination=control_combination,
        mdd_combination=mdd_combination,
        fixed_perms=True,
        test=True,
        label_smoothing=LABEL_SMOOTHING
    )
    loader = DataLoader(dataset=dataset)

    if plot_control:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
        control_count = 0
        mdd_count = 0
        for (data, label) in loader:
            data = data[0]
            control = (label.item() == 0)
            control_count = (control_count + 1 if control else control_count)
            mdd_count = (mdd_count + 1 if not control else mdd_count)
            if control:
                ax1.plot(data[...,0], data[...,1],
                         label=f'{"Control" if control else patient_groups[1]} Set {control_combination[control_count-1] if control else mdd_combination[mdd_count-1]}')
                ax1.set(xlabel='Time (normalized)', ylabel='CORT Concentration (standardized)', title='Control ACTH')
                ax1.legend(shadow=True, fancybox=True, loc='upper right')

                ax2.plot(data[...,0], data[...,2],
                         label=f'{"Control" if control else patient_groups[1]} Set {control_combination[control_count-1] if control else mdd_combination[mdd_count-1]}')
                ax2.set(xlabel='Time (normalized)', ylabel='ACTH Concentration (standardized)', title='Control Cortisol')
                ax2.legend(shadow=True, fancybox=True, loc='upper right')
        plt.savefig(f'Augmented Data Comparison Graphs/Individuals_Control_test_combination{control_combination}_comparison-graph.png', dpi=300)
        plt.close(fig)
    fig2, (ax3, ax4) = plt.subplots(nrows=2, figsize=(10,10))
    control_count = 0
    mdd_count = 0
    for (data, label) in loader:
        data = data[0]
        control = (label.item() == 0)
        control_count = (control_count + 1 if control else control_count)
        mdd_count = (mdd_count + 1 if not control else mdd_count)
        if not control:
            ax3.plot(data[...,0], data[...,1],
                     label=f'{patient_groups[1]} Set {mdd_combination[mdd_count-1]}')
            ax3.set(xlabel='Time (normalized)', ylabel='CORT Concentration (standardized)', title=f'{patient_groups[1]} ACTH')
            ax3.legend(shadow=True, fancybox=True, loc='upper right')

            ax4.plot(data[...,0], data[...,2],
                     label=f'{patient_groups[1]} Set {mdd_combination[mdd_count-1]}')
            ax4.set(xlabel='Time (normalized)', ylabel='ACTH Concentration (standardized)', title=f'{patient_groups[1]} Cortisol')
            ax4.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Individuals_{PATIENT_GROUPS[1]}_test_combination{mdd_combination}_comparison-graph.png', dpi=300)
    plt.close(fig2)


if __name__ == '__main__':
    match PATIENT_GROUPS[1]:
        case 'Atypical':
            per_idx = 1
        case 'Melancholic':
            per_idx = 2
        case 'Neither':
            per_idx = 3
        case _:
            raise ValueError('Invalid MDD Group Selected')

    for combo in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]:

        control_combination = tuple(PERMUTATIONS[0][combo[0]*5:(combo[0]+1)*5])
        mdd_combination = tuple(PERMUTATIONS[per_idx][combo[1]*5:((combo[1]+1)*5) if (combo[1]+1)*5 < len(PERMUTATIONS[per_idx]) else len(PERMUTATIONS[per_idx])])

        # plot_test_patients_by_group(PATIENT_GROUPS, control_combination, mdd_combination, plot_control=False)
        plot_test_patients_group_comparison_indiv_lines(PATIENT_GROUPS, control_combination, mdd_combination)
        plot_test_patients_group_comparison_mean_lines(PATIENT_GROUPS, control_combination, mdd_combination)
        # dataset = VirtualPopulation(
        #     patient_groups=PATIENT_GROUPS,
        #     method=METHOD,
        #     normalize_standardize=NORMALIZE_STANDARDIZE,
        #     num_per_patient=NUM_PER_PATIENT,
        #     num_patients=NUM_PATIENTS,
        #     control_combination=control_combination,
        #     mdd_combination=mdd_combination,
        #     fixed_perms=True,
        #     test=True,
        #     label_smoothing=LABEL_SMOOTHING
        # )
        # loader = DataLoader(dataset=dataset)



        # for batch in loader:
        #     data, label = batch

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

