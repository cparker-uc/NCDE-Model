# File Name: plot_augmented_data.py
# Author: Christopher Parker
# Created: Thu Jun 22, 2023 | 12:34P EDT
# Last Modified: Wed Jul 19, 2023 | 08:46P EDT

"""Uses Matplotlib to plot augmented datasets for examination"""

# Training data selection parameters
PATIENT_GROUPS = ['Control', 'Melancholic']
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'StandardizeAll'
NUM_PER_PATIENT = 100
NUM_PATIENTS = 10
BATCH_SIZE = 1
LABEL_SMOOTHING = 0

import torch
import numpy as np
import matplotlib.pyplot as plt
from get_nelson_data import (VirtualPopulation, FullVirtualPopulation,
                             FullVirtualPopulation_ByLab)
from torch.utils.data import DataLoader

# PERMUTATIONS = [
#     # Control
#     [13, 11,  8, 14,  6,  2, 12, 10,  5,  1,  0,  9,  3,  7,  4],
#     # Atypical
#     [ 0,  7, 12,  8, 11,  2,  6,  9,  3,  5,  4,  1, 13, 10],
#     # Melancholic
#     [10, 13,  4,  2,  3, 12, 14,  6,  8,  9,  5,  7,  0, 11,  1],
#     # Neither
#     [ 5, 12,  7, 13, 11,  9,  4,  1,  0,  2,  3, 10,  6,  8]
# ]
# PERMUTATIONS = [
#     # Control
#     [30, 11, 3, 38, 29, 35, 1, 31, 14, 19, 39, 17, 23, 27, 8, 16, 22, 47,
#      15, 7, 26, 33, 36, 49, 2, 37, 4, 45, 48, 20, 12, 18, 34, 42, 21, 46,
#      28, 13, 50, 51, 25, 44, 40, 41, 43, 0, 6, 9, 24, 32, 10, 5],
#     # MDD
#     [41, 8, 15, 16, 33, 43, 3, 19, 7, 1, 11, 12, 53, 29, 55, 37, 24, 6, 54,
#      21, 27, 47, 13, 25, 5, 0, 30, 46, 17, 23, 36, 10, 39, 14, 18, 35, 22,
#      50, 45, 28, 38, 9, 49, 26, 34, 4, 32, 48, 44, 31, 42, 52, 20, 51, 40,
#      2]
# ]
PERMUTATIONS = [
    # Nelson
    [22, 10, 50, 39, 48, 40, 15,  6, 37, 25, 34,  0, 26, 12, 41, 24, 30, 57,
     49, 53, 46, 56,  4, 38,  5, 43, 19, 11, 17, 31, 29, 20, 35,  8, 52, 21,
     13, 18, 32, 54, 47, 28, 36, 14,  1, 45,  9, 44,  3,  2, 16, 27, 42, 51,
     33, 23, 55,  7],
    # Ableson
    [8, 45,  1,  2, 24,  7, 32, 40, 15, 34, 26, 13, 27, 25, 16, 18, 29, 14,
     48, 38, 36, 30, 39, 35, 43, 20, 47,  6, 28,  3, 33, 49, 46, 37, 41,  0,
     12, 11, 17, 44,  9, 23, 10,  4, 42, 19, 31, 22,  5, 21]
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
    ax1.plot(control_mean[:,0], control_mean[:,1], color='orange', label=f'Control Set {ctrl_num}')
    ax1.plot(mdd_mean[:,0], mdd_mean[:,1], color='blue', label=f'{PATIENT_GROUPS[1]} Set {mdd_num}')
    ax1.legend(shadow=True, fancybox=True, loc='upper right')
    ax2.plot(control_mean[:,0], control_mean[:,2], color='orange', label=f'Control Set {ctrl_num}')
    ax2.plot(mdd_mean[:,0], mdd_mean[:,2], color='blue', label=f'{PATIENT_GROUPS[1]} Set {mdd_num}')
    ax2.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Group-Comparison_Mean_Lines_Control_{PATIENT_GROUPS[1]}_augmented_combination{(ctrl_num, mdd_num)}_comparison-graph.png', dpi=300)
    plt.close(fig)


def plot_test_patients_full_group_comparison_mean_lines(ctrl_num,
                                                        control_combination,
                                                        mdd_num,
                                                        mdd_combination,
                                                        by_lab=False):
    if not by_lab:
        dataset = FullVirtualPopulation(
            method=METHOD,
            noise_magnitude=0.1,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=True,
            label_smoothing=LABEL_SMOOTHING,
        )
    else:
        dataset = FullVirtualPopulation_ByLab(
            method=METHOD,
            noise_magnitude=0.1,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            nelson_combination=control_combination,
            ableson_combination=mdd_combination,
            test=True,
        )
    loader = DataLoader(dataset=dataset)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
    control_mean = torch.mean(dataset[:5][0], 0)
    mdd_mean = torch.mean(dataset[5:][0], 0)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
    ax1.plot(control_mean[:,0], control_mean[:,1], color='orange', label=f'Control Set {ctrl_num}')
    ax1.plot(mdd_mean[:,0], mdd_mean[:,1], color='blue', label=f'MDD Set {mdd_num}')
    ax1.legend(shadow=True, fancybox=True, loc='upper right')
    ax2.plot(control_mean[:,0], control_mean[:,2], color='orange', label=f'Control Set {ctrl_num}')
    ax2.plot(mdd_mean[:,0], mdd_mean[:,2], color='blue', label=f'MDD Set {mdd_num}')
    ax2.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Full Virtual Population{" (By Lab)" if by_lab else ""}/Group-Comparison_Mean_Lines_Control_{PATIENT_GROUPS[1]}_augmented_combination{(ctrl_num, mdd_num)}_comparison-graph.png', dpi=300)
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


def plot_test_patients_full_group_comparison_indiv_lines(ctrl_num,
                                                         control_combination,
                                                         mdd_num,
                                                         mdd_combination,
                                                         by_lab=False):
    if not by_lab:
        dataset = FullVirtualPopulation(
            method=METHOD,
            noise_magnitude=0.1,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=True,
            label_smoothing=LABEL_SMOOTHING
        )
    else:
        dataset = FullVirtualPopulation_ByLab(
            method=METHOD,
            noise_magnitude=0.1,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            nelson_combination=control_combination,
            ableson_combination=mdd_combination,
            test=True,
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
                 label=f'{"Control" if control else "MDD"} Test Patient {control_combination[control_count-1] if control else mdd_combination[mdd_count-1]}')
        ax1.set(xlabel='Time (normalized)', ylabel='CORT Concentration (standardized)', title='ACTH Comparison')
        ax1.legend(shadow=True, fancybox=True, loc='upper right')

        ax2.plot(data[...,0], data[...,2], color=f'{"orange" if control else "blue"}',
                 label=f'{"Control" if control else "MDD"} Test Patient {control_combination[control_count-1] if control else mdd_combination[mdd_count-1]}')
        ax2.set(xlabel='Time (normalized)', ylabel='ACTH Concentration (standardized)', title='Cortisol Comparison')
        ax2.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Full Virtual Population{" (By Lab)" if by_lab else ""}/Group-Comparison_Individual_Lines_Control_{PATIENT_GROUPS[1]}_augmented_combination{(ctrl_num, mdd_num)}_comparison-graph.png', dpi=300)
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


def plot_test_patients_by_group_fullvpop(ctrl_num, control_combination,
                                mdd_num, mdd_combination, by_lab=False):
    if not by_lab:
        dataset = FullVirtualPopulation(
            method=METHOD,
            noise_magnitude=0.1,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=True,
            label_smoothing=LABEL_SMOOTHING
        )
    else:
        dataset = FullVirtualPopulation_ByLab(
            method=METHOD,
            noise_magnitude=0.1,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            nelson_combination=control_combination,
            ableson_combination=mdd_combination,
            test=True,
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
        if control:
            ax1.plot(data[...,0], data[...,1],
                     label=f'Control Test Patient {control_combination[control_count-1]}')
            ax1.set(xlabel='Time (normalized)', ylabel='CORT Concentration (standardized)', title='Control ACTH')
            ax1.legend(shadow=True, fancybox=True, loc='upper right')

            ax2.plot(data[...,0], data[...,2],
                     label=f'Control Test Patient {control_combination[control_count-1]}')
            ax2.set(xlabel='Time (normalized)', ylabel='ACTH Concentration (standardized)', title='Control Cortisol')
            ax2.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Full Virtual Population{" (By Lab)" if by_lab else ""}/Individuals_Control_test_combination{control_combination}_comparison-graph.png', dpi=300)
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
                     label=f'MDD Test Patient {mdd_combination[mdd_count-1]}')
            ax3.set(xlabel='Time (normalized)', ylabel='CORT Concentration (standardized)', title=f'MDD ACTH')
            ax3.legend(shadow=True, fancybox=True, loc='upper right')

            ax4.plot(data[...,0], data[...,2],
                     label=f'MDD Test Patient {mdd_combination[mdd_count-1]}')
            ax4.set(xlabel='Time (normalized)', ylabel='ACTH Concentration (standardized)', title=f'MDD Cortisol')
            ax4.legend(shadow=True, fancybox=True, loc='upper right')
    plt.savefig(f'Augmented Data Comparison Graphs/Full Virtual Population{" (By Lab)" if by_lab else ""}/Individuals_{PATIENT_GROUPS[1]}_test_combination{mdd_combination}_comparison-graph.png', dpi=300)
    plt.close(fig2)


if __name__ == '__main__':
    # match PATIENT_GROUPS[1]:
    #     case 'Atypical':
    #         per_idx = 1
    #     case 'Melancholic':
    #         per_idx = 2
    #     case 'Neither':
    #         per_idx = 3
    #     case _:
    #         raise ValueError('Invalid MDD Group Selected')

    ctrl_range = [i for i in range(12)]
    mdd_range = [i for i in range(10)]
    for ctrl_num in ctrl_range:
        for mdd_num in mdd_range:
            control_combination = tuple(PERMUTATIONS[0][ctrl_num*5:(ctrl_num+1)*5])
            mdd_combination = tuple(PERMUTATIONS[1][mdd_num*5:((mdd_num+1)*5) if (mdd_num+1)*5 < len(PERMUTATIONS[1]) else len(PERMUTATIONS[1])])

            plot_test_patients_by_group_fullvpop(ctrl_num, control_combination, mdd_num, mdd_combination, by_lab=True)
            plot_test_patients_full_group_comparison_indiv_lines(ctrl_num, control_combination, mdd_num, mdd_combination, by_lab=True)
            plot_test_patients_full_group_comparison_mean_lines(ctrl_num, control_combination, mdd_num, mdd_combination, by_lab=True)

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

