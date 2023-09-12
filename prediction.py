# File Name: prediction.py
# Author: Christopher Parker
# Created: Mon Sep 11, 2023 | 10:34P EDT
# Last Modified: Mon Sep 11, 2023 | 10:39P EDT

"""This is the root file for use in prediction (rather than classification).
The NN learns to approximate the underlying data through the loss function
(either with or without mechanistic components added)."""


# Network architecture parameters
NETWORK_TYPE = 'ANN' # NCDE, NODE, ANN or RNN
# Should be 40 for Toy dataset or 22 for others if using ANN or RNN
# If using NCDE, should be the number of vars plus 1, since we include time
# If using NODE, should just be the number of vars
INPUT_CHANNELS = 22
HDIM = 128
# Needs to be 2 for NODE (even though it will be run through readout to combine down to 1)
OUTPUT_CHANNELS = 1
# Only necessary for RNN
N_RECURS = 3
CLASSIFY = False
MECHANISTIC = False

# Training hyperparameters
ITERS = 20
SAVE_FREQ = 20
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-8
RTOL = 1e-6

# Training data selection parameters
POP = 'FullVPOP'
PATIENT_GROUPS = ['Control', 'Atypical'] # Only necessary for POP='NelsonOnly'
METHOD = 'Uniform'
NORMALIZE_STANDARDIZE = 'StandardizeAll'
NOISE_MAGNITUDE = 0.05
NUM_PER_PATIENT = 100
POP_NUMBER = 0
BATCH_SIZE = 10
LABEL_SMOOTHING = 0
DROPOUT = 0.5
CORT_ONLY = False
# These variables determine which population groups to train/test using
#  Should be 3 for both if using NelsonOnly data, 11 for control and 12 for MDD
#  if using FullVPOP or 12 for control and 10 for MDD if using FullVPOPByLab
CTRL_RANGE = list(range(11))
MDD_RANGE = list(range(12))

# End time for use with toy dataset (2.35 hours, 10 hours or 24 hours)
T_END = 2.35


import sys
import torch
from training import train
from testing import test


if __name__ == '__main__':
    try:
        if sys.argv[1].lower() == 'train':
            train(
                hyperparameters={
                    'NETWORK_TYPE': NETWORK_TYPE,
                    'INPUT_CHANNELS': INPUT_CHANNELS,
                    'HDIM': HDIM,
                    'OUTPUT_CHANNELS': OUTPUT_CHANNELS,
                    'N_RECURS': N_RECURS,
                    'ITERS': ITERS,
                    'SAVE_FREQ': SAVE_FREQ,
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
                    'T_END': T_END
                },
                virtual=True,
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
                    'N_RECURS': N_RECURS,
                    'ITERS': ITERS,
                    'SAVE_FREQ': SAVE_FREQ,
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
                    'MAX_ITR': ITERS,
                    'T_END': T_END
                },
                virtual=True,
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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#                                 MIT License                               #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#     Copyright (c) 2023 Christopher John Parker <parkecp@mail.uc.edu>      #
#                                                                           #
# Permission is hereby granted, free of charge, to any person obtaining a   #
# copy of this software and associated documentation files (the "Software"),#
# to deal in the Software without restriction, including without limitation #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,  #
# and/or sell copies of the Software, and to permit persons to whom the     #
# Software is furnished to do so, subject to the following conditions:      #
#                                                                           #
# The above copyright notice and this permission notice shall be included   #
# in all copies or substantial portions of the Software.                    #
#                                                                           #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS   #
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF                #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.    #
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY      #
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,      #
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE         #
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                    # 
#                                                                           #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

