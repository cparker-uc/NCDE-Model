# File Name: galerkin_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Tue Dec 19, 2023 | 10:01P EST

"Root file for classification of augmented TSST or simulation data"

# Network architecture parameters
NETWORK_TYPE = 'RNN' # NCDE, NODE, ANN or RNN

# INPUT_CHANNELS Should be 40 for Toy dataset or 22 for others if using ANN or
#  RNN
# If using NCDE, should be the number of vars plus 1, since we include time
# If using NODE, should just be the number of vars
INPUT_CHANNELS = 22
HDIM = 32
OUTPUT_CHANNELS = 1 # 1 for non-NODE classification, else same as INPUT_CHANNELS

# Only necessary for RNN
N_LAYERS = 1
SEQ_LENGTH = 11

CLASSIFY = True
MECHANISTIC = False

# Training hyperparameters
ITERS = 200
SAVE_FREQ = 100
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-9
RTOL = 1e-6
ADJOINT_ATOL = 1e-9
ADJOINT_RTOL = 1e-6

# Training data selection parameters
POP = 'NelsonOnly'
PATIENT_GROUPS = ['Control', 'Atypical'] # Only necessary for POP='NelsonOnly' or 'AblesonOnly'
INDIVIDUAL_NUMBER = 0 # Same
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'None'
NOISE_MAGNITUDE = 0.05
IRREGULAR_T_SAMPLES = False
NUM_PER_PATIENT = 100
POP_NUMBER = 0
BATCH_SIZE = 200
LABEL_SMOOTHING = 0
DROPOUT = 0.
CORT_ONLY = False
# These variables determine which population groups to train/test using
#  Should be 3 for both if using NelsonOnly data, 11 for control and 12 for MDD
#  if using FullVPOP or 12 for control and 10 for MDD if using FullVPOPByLab
CTRL_RANGE = list(range(3))
MDD_RANGE = list(range(3))

# End time for use with toy dataset (2.35 hours, 10 hours or 24 hours)
T_END = 140


import sys
import torch

from training import train
from testing import test


# Define the device with which to train networks
DEVICE = torch.device('cpu')

# These are the permutations of test patients selected from each group 
#  (hard-coded for reproducibility and easy reference)

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
        return PATIENT_GROUPS
    elif population == 'ablesononly':
        return PATIENT_GROUPS
    elif population == 'fullvpopbylab':
        return ['Nelson', 'Ableson']
    elif population == 'plusablesonmdd0and12':
        return ['Control', 'Atypical']
    elif population == 'toydata':
        return PATIENT_GROUPS

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
    patient_groups = set_patient_groups(POP.lower())
    if POP.lower() not in ['toydata', 'ablesononly']:
        perms = PERMUTATIONS[POP.lower()]
    if POP.lower() == 'nelsononly' and not INDIVIDUAL_NUMBER:
        match patient_groups[1]:
            case 'Atypical':
                perms = [perms[0], perms[1]]
            case 'Melancholic':
                perms = [perms[0], perms[2]]
            case 'Neither':
                perms = [perms[0], perms[3]]
    elif INDIVIDUAL_NUMBER:
        perms = None
    elif POP.lower() not in ['toydata', 'ablesononly', 'nelsononly', 'fullvpop', 'fullvpopbylab']:
        perms = None
        # If not using Python 3.11:
        # if patient_groups[1] == "Atypical":
        #     perms = [perms[0], perms[1]]
        # elif patient_groups[1] == "Melancholic":
        #     perms = [perms[0], perms[2]]
        # elif patient_groups[1] == "Neither":
        #     perms = [perms[0], perms[3]]

    try:
        if sys.argv[1].lower() == 'train':
            train(
                hyperparameters={
                    'NETWORK_TYPE': NETWORK_TYPE,
                    'INPUT_CHANNELS': INPUT_CHANNELS,
                    'HDIM': HDIM,
                    'OUTPUT_CHANNELS': OUTPUT_CHANNELS,
                    'N_LAYERS': N_LAYERS,
                    'SEQ_LENGTH': SEQ_LENGTH,
                    'CLASSIFY': CLASSIFY,
                    'MECHANISTIC': MECHANISTIC,
                    'ITERS': ITERS,
                    'SAVE_FREQ': SAVE_FREQ,
                    'LR': LR,
                    'DECAY': DECAY,
                    'OPT_RESET': OPT_RESET,
                    'ATOL': ATOL,
                    'RTOL': RTOL,
                    'ADJOINT_ATOL': ADJOINT_ATOL,
                    'ADJOINT_RTOL': ADJOINT_RTOL,
                    'PATIENT_GROUPS': patient_groups,
                    'INDIVIDUAL_NUMBER': INDIVIDUAL_NUMBER,
                    'METHOD': METHOD,
                    'NORMALIZE_STANDARDIZE': NORMALIZE_STANDARDIZE,
                    'NOISE_MAGNITUDE': NOISE_MAGNITUDE,
                    'IRREGULAR_T_SAMPLES': IRREGULAR_T_SAMPLES,
                    'NUM_PER_PATIENT': NUM_PER_PATIENT,
                    'POP_NUMBER': POP_NUMBER,
                    'BATCH_SIZE': BATCH_SIZE,
                    'LABEL_SMOOTHING': LABEL_SMOOTHING,
                    'DROPOUT': DROPOUT,
                    'CORT_ONLY': CORT_ONLY,
                    'T_END': T_END,
                    'DEVICE': DEVICE,
                    'POP': POP,
                },
                virtual=True,
                ableson_pop=False,
                permutations=perms,
                ctrl_range=CTRL_RANGE,
                mdd_range=MDD_RANGE,
                plus_ableson_mdd=True if POP.lower()=='plusablesonmdd0and12' else False,
                toy_data=True if POP.lower()=='toydata' else False
            )
        elif sys.argv[1].lower() == 'test':
            test(
                hyperparameters={
                    'NETWORK_TYPE': NETWORK_TYPE,
                    'INPUT_CHANNELS': INPUT_CHANNELS,
                    'HDIM': HDIM,
                    'OUTPUT_CHANNELS': OUTPUT_CHANNELS,
                    'N_LAYERS': N_LAYERS,
                    'SEQ_LENGTH': SEQ_LENGTH,
                    'CLASSIFY': CLASSIFY,
                    'MECHANISTIC': MECHANISTIC,
                    'ITERS': ITERS,
                    'SAVE_FREQ': SAVE_FREQ,
                    'LR': LR,
                    'DECAY': DECAY,
                    'OPT_RESET': OPT_RESET,
                    'ATOL': ATOL,
                    'RTOL': RTOL,
                    'ADJOINT_ATOL': ADJOINT_ATOL,
                    'ADJOINT_RTOL': ADJOINT_RTOL,
                    'PATIENT_GROUPS': patient_groups,
                    'INDIVIDUAL_NUMBER': INDIVIDUAL_NUMBER,
                    'METHOD': METHOD,
                    'NORMALIZE_STANDARDIZE': NORMALIZE_STANDARDIZE,
                    'NOISE_MAGNITUDE': NOISE_MAGNITUDE,
                    'IRREGULAR_T_SAMPLES': IRREGULAR_T_SAMPLES,
                    'NUM_PER_PATIENT': NUM_PER_PATIENT,
                    'POP_NUMBER': POP_NUMBER,
                    'BATCH_SIZE': BATCH_SIZE,
                    'LABEL_SMOOTHING': LABEL_SMOOTHING,
                    'DROPOUT': DROPOUT,
                    'CORT_ONLY': CORT_ONLY,
                    'MAX_ITR': ITERS,
                    'T_END': T_END,
                    'DEVICE': DEVICE,
                    'POP': POP,
                },
                virtual=True,
                ableson_pop=False,
                permutations=perms,
                ctrl_range=CTRL_RANGE,
                mdd_range=MDD_RANGE,
                plus_ableson_mdd=True if POP.lower()=='plusablesonmdd0and12' else False,
                toy_data=True if POP.lower()=='toydata' else False
            )
        else:
            usage_hint()
    except IndexError:
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
