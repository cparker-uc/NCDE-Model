# File Name: galerkin_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Fri Jul 28, 2023 | 03:47P EDT

"Working on NCDE classification of augmented Nelson data"

# Network architecture parameters
INPUT_CHANNELS = 3
HDIM = 32
OUTPUT_CHANNELS = 1

# Training hyperparameters
ITERS = 200
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-8
RTOL = 1e-6

# Training data selection parameters
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'StandardizeAll'
NOISE_MAGNITUDE = 0.05
NUM_PER_PATIENT = 100
POP_NUMBER = 0
BATCH_SIZE = 200
LABEL_SMOOTHING = 0
DROPOUT = 0
CORT_ONLY = False
# These variables determine which population groups to train/test using
#  Should be 3 for both if using NelsonOnly data, 11 for control and 12 for MDD
#  if using FullVPOP or 12 for control and 10 for MDD if using FullVPOPByLab
CTRL_RANGE = list(range(12))
MDD_RANGE = list(range(10))
# CTRL_RANGE = [0]
# MDD_RANGE = [1]


import sys
import torch

# from typing import Tuple
from training import train
from testing import test


# Define the device with which to train networks
DEVICE = torch.device('cpu')

# Not certain if this is necessary, but in the quickstart docs they have
#  done a wildcard import of torchdyn base library, and this is all that does
# TTuple = Tuple[torch.Tensor, torch.Tensor]

PERMUTATIONS = {
    'nelsononly': [
        # Control
        [13, 11,  8, 14,  6,  2, 12, 10,  5,  1,  0,  9,  3,  7,  4],
        # Atypical
        [ 0,  7, 12,  8, 11,  2,  6,  9,  3,  5,  4,  1, 13, 10],
        # Melancholic
        [10, 13,  4,  2,  3, 12, 14,  6,  8,  9,  5,  7,  0, 11,  1],
        # Neither
        [ 5, 12,  7, 13, 11,  9,  4,  1,  0,  2,  3, 10,  6,  8]
    ],
    'plusablesonmdd0and12': [
        # Control
        [13, 11,  8, 14,  6,  2, 12, 10,  5,  1,  0,  9,  3,  7,  4],
        # Atypical
        [11, 12,  6, 10, 13,  2,  9,  3,  8, 14,  1,  4,  5,  7, 15,  0]
    ],
    'fullvpop': [
        # Control
        [30, 11, 3, 38, 29, 35, 1, 31, 14, 19, 39, 17, 23, 27, 8, 16, 22, 47,
         15, 7, 26, 33, 36, 49, 2, 37, 4, 45, 48, 20, 12, 18, 34, 42, 21, 46,
         28, 13, 50, 51, 25, 44, 40, 41, 43, 0, 6, 9, 24, 32, 10, 5],
        # MDD
        [41, 8, 15, 16, 33, 43, 3, 19, 7, 1, 11, 12, 53, 29, 55, 37, 24, 6, 54,
         21, 27, 47, 13, 25, 5, 0, 30, 46, 17, 23, 36, 10, 39, 14, 18, 35, 22,
         50, 45, 28, 38, 9, 49, 26, 34, 4, 32, 48, 44, 31, 42, 52, 20, 51, 40,
         2]
    ],
    'fullvpopbylab': [
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
}


def set_patient_groups(population):
    if population == 'nelsononly':
        return ['Control', sys.argv[3]]
    elif population == 'fullvpopbylab':
        return ['Nelson', 'Ableson']
    elif population == 'plusablesonmdd0and12':
        return ['Control', 'Atypical']

    return ['Control', 'MDD']


def usage_hint():
    raise(ValueError('Try again with the proper CLI arguments.\n\n'
                     'Usage:\n\tpython classification.py train/test population'
                     ' (Atypical, Melancholic, or Neither)'
                     '\n\nWhere train initiates training and test initiates '
                     'testing,\nand population is one of: "NelsonOnly",'
                     ' "FullVPOP", "FullVPOPByLab".\nIf using Nelson only '
                     'populations, include patient subtype after NelsonOnly.'
                     '\nTo change network '
                     'training hyperparameters, edit the constant variables '
                     'at the top of the file.\n'))

if __name__ == "__main__":
    try:
        pop = sys.argv[2].lower()
        patient_groups = set_patient_groups(pop)
        perms = PERMUTATIONS[pop]
        if pop == 'nelsononly':
            match patient_groups[1]:
                case 'Atypical':
                    perms = [perms[0], perms[1]]
                case 'Melancholic':
                    perms = [perms[0], perms[2]]
                case 'Neither':
                    perms = [perms[0], perms[3]]
    except IndexError:
        usage_hint()

    if sys.argv[1].lower() == 'train':
        train(
            hyperparameters={
                'INPUT_CHANNELS': INPUT_CHANNELS,
                'HDIM': HDIM,
                'OUTPUT_CHANNELS': OUTPUT_CHANNELS,
                'ITERS': ITERS,
                'LR': LR,
                'DECAY': DECAY,
                'OPT_RESET': OPT_RESET,
                'ATOL': ATOL,
                'RTOL': RTOL,
                'PATIENT_GROUPS': patient_groups,
                'METHOD': METHOD,
                'NORMALIZE_STANDARDIZE': NORMALIZE_STANDARDIZE,
                'NOISE_MAGNITUDE': NOISE_MAGNITUDE,
                'NUM_PER_PATIENT': NUM_PER_PATIENT,
                'POP_NUMBER': POP_NUMBER,
                'BATCH_SIZE': BATCH_SIZE,
                'LABEL_SMOOTHING': LABEL_SMOOTHING,
                'DROPOUT': DROPOUT,
                'CORT_ONLY': CORT_ONLY
            },
            virtual=True,
            permutations=perms,
            ctrl_range=CTRL_RANGE,
            mdd_range=MDD_RANGE,
            # ableson_pop=True,
            # plus_ableson_mdd=True
        )
    elif sys.argv[1].lower() == 'test':
        test(
            hyperparameters={
                'INPUT_CHANNELS': INPUT_CHANNELS,
                'HDIM': HDIM,
                'OUTPUT_CHANNELS': OUTPUT_CHANNELS,
                'ITERS': ITERS,
                'LR': LR,
                'DECAY': DECAY,
                'OPT_RESET': OPT_RESET,
                'ATOL': ATOL,
                'RTOL': RTOL,
                'PATIENT_GROUPS': patient_groups,
                'METHOD': METHOD,
                'NORMALIZE_STANDARDIZE': NORMALIZE_STANDARDIZE,
                'NOISE_MAGNITUDE': NOISE_MAGNITUDE,
                'NUM_PER_PATIENT': NUM_PER_PATIENT,
                'POP_NUMBER': POP_NUMBER,
                'BATCH_SIZE': BATCH_SIZE,
                'LABEL_SMOOTHING': LABEL_SMOOTHING,
                'DROPOUT': DROPOUT,
                'CORT_ONLY': CORT_ONLY,
                'MAX_ITR': int(ITERS/100)
            },
            virtual=True,
            permutations=perms,
            ctrl_range=CTRL_RANGE,
            mdd_range=MDD_RANGE,
            # ableson_pop=True,
            # plus_ableson_mdd=True
        )
    else:
        usage_hint()


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
