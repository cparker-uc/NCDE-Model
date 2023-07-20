# File Name: augment_data.py
# Author: Christopher Parker
# Created: Thu Jun 15, 2023 | 06:08P EDT
# Last Modified: Thu Jul 20, 2023 | 03:16P EDT

"""This script contains methods for augmenting a given tensor of time-series
data with various strategies, such as Gaussian noise."""

PATIENT_GROUP = 'Control'
NUM_PER_PATIENT = 100
NUM_PATIENTS = 10
NUM_POPS = 5
METHOD = 'Uniform'
NOISE_MAGNITUDE = 0.1
NORMALIZE_STANDARDIZE = 'StandardizeAll'

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from get_data import NelsonData, AblesonData
from itertools import combinations

def uniform_noise(input_tensor, noise_magnitude):
    output_tensor = torch.zeros_like(input_tensor)
    for idx, pt in enumerate(input_tensor):
        scaled = pt*noise_magnitude
        noise_range = 2*scaled
        lower_bound = pt - scaled
        output_tensor[idx] = torch.rand((1,))*noise_range + lower_bound
    return output_tensor

def generate_augmented_dataset(input_data, number, method, noise_magnitude=NOISE_MAGNITUDE):
    vpop = torch.zeros((number, 11, 3), dtype=float)
    match method:
        case 'Uniform':
            for vpt in vpop:
                vpt[...,0] = input_data[...,0]
                vpt[...,1] = uniform_noise(input_data[...,1], noise_magnitude)
                vpt[...,2] = uniform_noise(input_data[...,2], noise_magnitude)
        case _:
            print("Unsupported augmentation strategy")
            return

    return vpop

def generate_virtual_population(patient_group, num_per_patient, test_pop_nums,
                                method, shuffle=True):
    dataset = NelsonData(patient_groups=[patient_group], normalize_standardize=NORMALIZE_STANDARDIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    vpop = torch.zeros((0, 11, 3), dtype=float)
    test_pop = torch.zeros((0, 11, 3))
    for idx, batch in enumerate(loader):
        batch = batch[0]
        if idx in test_pop_nums:
            test_pop = torch.cat((test_pop, batch), 0)
            continue
        vpop = torch.cat(
            (
                vpop,
                generate_augmented_dataset(batch, num_per_patient, method)
            ), 0
        )

    return torch.cat((vpop, test_pop), 0)

def generate_full_virtual_population(patient_group, num_per_patient, test_pop_nums,
                                method, shuffle=True):
    if patient_group == 'MDD':
        nelson_dataset = NelsonData(patient_groups=['Atypical', 'Melancholic', 'Neither'], normalize_standardize=NORMALIZE_STANDARDIZE)
    else:
        nelson_dataset = NelsonData(patient_groups=[patient_group], normalize_standardize=NORMALIZE_STANDARDIZE)
    ableson_dataset = AblesonData(patient_groups=[patient_group], normalize_standardize=NORMALIZE_STANDARDIZE)
    loader = DataLoader(ConcatDataset((nelson_dataset, ableson_dataset)), batch_size=1, shuffle=shuffle)
    vpop = torch.zeros((0, 11, 3), dtype=float)
    test_pop = torch.zeros((0, 11, 3))
    for idx, batch in enumerate(loader):
        batch = batch[0]
        if idx in test_pop_nums:
            test_pop = torch.cat((test_pop, batch), 0)
            continue
        vpop = torch.cat(
            (
                vpop,
                generate_augmented_dataset(batch, num_per_patient, method)
            ), 0
        )

    return torch.cat((vpop, test_pop), 0)


def generate_nelson_virtual_population(num_per_patient, test_pop_nums, method):
    nelson_dataset = NelsonData(patient_groups=['Control', 'Atypical', 'Melancholic', 'Neither'], normalize_standardize=NORMALIZE_STANDARDIZE)
    loader = DataLoader(nelson_dataset, batch_size=1, shuffle=False)
    vpop = torch.zeros((0, 11, 3), dtype=float)
    test_pop = torch.zeros((0, 11, 3))
    for idx, batch in enumerate(loader):
        batch = batch[0]
        if idx in test_pop_nums:
            test_pop = torch.cat((test_pop, batch), 0)
            continue
        vpop = torch.cat(
            (
                vpop,
                generate_augmented_dataset(batch, num_per_patient, method)
            ), 0
        )

    return torch.cat((vpop, test_pop), 0)


def generate_ableson_virtual_population(num_per_patient, test_pop_nums, method):
    ableson_dataset = AblesonData(patient_groups=['Control', 'MDD'], normalize_standardize=NORMALIZE_STANDARDIZE)
    loader = DataLoader(ableson_dataset, batch_size=1, shuffle=False)
    vpop = torch.zeros((0, 11, 3), dtype=float)
    test_pop = torch.zeros((0, 11, 3))
    for idx, batch in enumerate(loader):
        batch = batch[0]
        if idx in test_pop_nums:
            test_pop = torch.cat((test_pop, batch), 0)
            continue
        vpop = torch.cat(
            (
                vpop,
                generate_augmented_dataset(batch, num_per_patient, method)
            ), 0
        )

    return torch.cat((vpop, test_pop), 0)


def generate_multiple_populations(num_pops):
    for i in range(num_pops):
        vpop_and_test = generate_virtual_population(PATIENT_GROUP, NUM_PER_PATIENT, NUM_PATIENTS, METHOD)
        torch.save(vpop_and_test, f'Virtual Populations/{PATIENT_GROUP}_{METHOD}{NOISE_MAGNITUDE}_{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_{NUM_PATIENTS}_{i+1}.txt')

def generate_all_pop_combinations():
    for combination in combinations(
            range(
                15 if PATIENT_GROUP in ['Control', 'Melancholic'] else 14
            ), 5 if PATIENT_GROUP in ['Control', 'Melancholic'] else 4
    ):
        vpop_and_test = generate_virtual_population(
            PATIENT_GROUP, NUM_PER_PATIENT, combination, METHOD, shuffle=False
        )
        torch.save(
            vpop_and_test,
            f'Virtual Populations/{PATIENT_GROUP}_{METHOD}{NOISE_MAGNITUDE}_'
            f'{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_'
            f'testPatients{combination}.txt'
        )


def generate_3combinations():
    perm_len = 5
    permutations = [
        # Control
        [13, 11,  8, 14,  6,  2, 12, 10,  5,  1,  0,  9,  3,  7,  4],
        # Atypical
        [ 0,  7, 12,  8, 11,  2,  6,  9,  3,  5,  4,  1, 13, 10],
    #     # Melancholic
        [10, 13,  4,  2,  3, 12, 14,  6,  8,  9,  5,  7,  0, 11,  1],
    #     # Neither
        [ 5, 12,  7, 13, 11,  9,  4,  1,  0,  2,  3, 10,  6,  8]
    ]
    match PATIENT_GROUP:
        case 'Control':
            random_permutation = permutations[0]
        case 'Atypical':
            random_permutation = permutations[1]
        case 'Melancholic':
            random_permutation = permutations[2]
        case 'Neither':
            random_permutation = permutations[3]

    # random_permutation = torch.randperm(15 if PATIENT_GROUP in ['Control', 'Melancholic'] else 14)

    random_combo1 = random_permutation[:perm_len]
    vpop_and_test1 = generate_virtual_population(
        PATIENT_GROUP, NUM_PER_PATIENT, random_combo1, METHOD, shuffle=False
    )
    torch.save(
        vpop_and_test1,
        f'Virtual Populations/{PATIENT_GROUP}_{METHOD}{NOISE_MAGNITUDE}_'
        f'{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_'
        f'testPatients{tuple(random_combo1)}_fixedperms.txt'
    )

    random_combo2 = random_permutation[perm_len:(2*perm_len)]
    vpop_and_test2 = generate_virtual_population(
        PATIENT_GROUP, NUM_PER_PATIENT, random_combo2, METHOD, shuffle=False
    )
    torch.save(
        vpop_and_test2,
        f'Virtual Populations/{PATIENT_GROUP}_{METHOD}{NOISE_MAGNITUDE}_'
        f'{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_'
        f'testPatients{tuple(random_combo2)}_fixedperms.txt'
    )

    random_combo3 = random_permutation[(2*perm_len):]
    vpop_and_test3 = generate_virtual_population(
        PATIENT_GROUP, NUM_PER_PATIENT, random_combo3, METHOD, shuffle=False
    )
    torch.save(
        vpop_and_test3,
        f'Virtual Populations/{PATIENT_GROUP}_{METHOD}{NOISE_MAGNITUDE}_'
        f'{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_'
        f'testPatients{tuple(random_combo3)}_fixedperms.txt'
    )


def generate_full_combinations(test_len):
    # random_permutation = torch.randperm(52 if PATIENT_GROUP == 'Control' else 56)
    # Using the same fixed permutation as with 0.1 noise to generate other noise
    #  amplitudes
    random_permutation = (
        [30, 11, 3, 38, 29, 35, 1, 31, 14, 19, 39, 17, 23, 27, 8, 16, 22, 47,
         15, 7, 26, 33, 36, 49, 2, 37, 4, 45, 48, 20, 12, 18, 34, 42, 21, 46,
         28, 13, 50, 51, 25, 44, 40, 41, 43, 0, 6, 9, 24, 32, 10, 5] if PATIENT_GROUP == 'Control' else
        [41, 8, 15, 16, 33, 43, 3, 19, 7, 1, 11, 12, 53, 29, 55, 37, 24, 6, 54,
         21, 27, 47, 13, 25, 5, 0, 30, 46, 17, 23, 36, 10, 39, 14, 18, 35, 22,
         50, 45, 28, 38, 9, 49, 26, 34, 4, 32, 48, 44, 31, 42, 52, 20, 51, 40,
         2]
    )


    # Loop through the necessary number of permutations to cover all patients
    #  with the given number of test patients per population
    n_pops = np.ceil(len(random_permutation)/test_len)
    for i in range(int(n_pops)):
        try:
            random_combo = random_permutation[i*test_len:(i+1)*test_len]
        except IndexError:
            random_combo = random_permutation[i*test_len:]
        vpop_and_test = generate_full_virtual_population(
            PATIENT_GROUP, NUM_PER_PATIENT, random_combo, METHOD, shuffle=False
        )
        torch.save(
            vpop_and_test,
            f'Virtual Populations/{PATIENT_GROUP}_{METHOD}{NOISE_MAGNITUDE}_'
            f'{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_'
            f'testPatients{tuple(random_combo)}_fixedperms.txt'
        )


def generate_full_combinations_by_lab(lab, test_len):
    # random_permutation = torch.randperm(58 if lab == 'Nelson' else 50)
    # Using the same fixed permutation as with 0.1 noise to generate other noise
    #  amplitudes
    random_permutation = [ 8, 45,  1,  2, 24,  7, 32, 40, 15, 34, 26, 13, 27, 25, 16, 18, 29, 14,
        48, 38, 36, 30, 39, 35, 43, 20, 47,  6, 28,  3, 33, 49, 46, 37, 41,  0,
        12, 11, 17, 44,  9, 23, 10,  4, 42, 19, 31, 22,  5, 21]

    # Loop through the necessary number of permutations to cover all patients
    #  with the given number of test patients per population
    n_pops = np.ceil(len(random_permutation)/test_len)
    for i in range(int(n_pops)):
        try:
            random_combo = random_permutation[i*test_len:(i+1)*test_len]
        except IndexError:
            random_combo = random_permutation[i*test_len:]
        if lab == 'Nelson':
            vpop_and_test = generate_nelson_virtual_population(
                NUM_PER_PATIENT, random_combo, METHOD
            )
        else:
            vpop_and_test = generate_ableson_virtual_population(
                NUM_PER_PATIENT, random_combo, METHOD
            )
        torch.save(
            vpop_and_test,
            f'Virtual Populations/{lab if lab=="Nelson" else "Ableson"}_{METHOD}{NOISE_MAGNITUDE}_'
            f'{NORMALIZE_STANDARDIZE}_{NUM_PER_PATIENT}_'
            f'testPatients{tuple(random_combo)}_fixedperms.txt'
        )
    return random_permutation


if __name__ == '__main__':
    # Generate a virtual population based on the parameters given
    # vpop_and_test = generate_virtual_population(PATIENT_GROUP, NUM_PER_PATIENT, NUM_PATIENTS, METHOD)

    # Save the virtual population and respective training population to files
    # torch.save(vpop_and_test, f'Virtual Populations/{PATIENT_GROUP}_{METHOD}_{NUM_PER_PATIENT}_{NUM_PATIENTS}_{POP_NUMBER}.txt')

    # for NORMALIZE_STANDARDIZE in ['Normalize', 'Standardize', 'None']:
    # for PATIENT_GROUP in ['Control', 'Atypical', 'Melancholic', 'Neither']:
    #     generate_all_pop_combinations()

    for PATIENT_GROUP in ['Control', 'Atypical', 'Melancholic', 'Neither']:
        generate_3combinations()

#     for PATIENT_GROUP in ['MDD', 'Control']:
#         generate_full_combinations(5)

    # x = generate_full_combinations_by_lab('Ableson', 5)

    # with open('nelson_combo.txt', 'w+') as f:
    #     f.write(str(x))
    # testpop = generate_full_virtual_population('MDD', 1, (0,), 'Uniform', shuffle=False)
    # print(testpop[:42,:,0])

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

