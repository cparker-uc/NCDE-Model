# File Name: training.py
# Author: Christopher Parker
# Created: Fri Jul 21, 2023 | 12:49P EDT
# Last Modified: Fri Dec 15, 2023 | 11:49P EST

"""This file defines the functions used for network training. These functions
are used in classification.py"""

from IPython.core.debugger import set_trace
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchcde
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from torchviz import make_dot
from torch.nn.functional import (binary_cross_entropy_with_logits, mse_loss,
                                 l1_loss)
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchdiffeq import odeint_adjoint
from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)

import matplotlib.pyplot as plt
from neural_cde import NeuralCDE
from neural_ode import NeuralODE
from ann import ANN
from mech_ann import MechanisticANN
from rnn import RNN
from get_data import AblesonData, NelsonData, SriramSimulation
from get_augmented_data import (FullVirtualPopulation,
                                FullVirtualPopulation_ByLab,
                                NelsonVirtualPopulation, ToyDataset)

# The constants listed here are to be defined in classification.py. The train()
#  function will iterate through the parameter names passed in the
#  parameter_dict argument and set values to these global variables for use in
#  all of the functions in this namespace
# I just set them here so that linters don't throw warnings
# Network architecture parameters
NETWORK_TYPE: str = ''
INPUT_CHANNELS: int = 0
HDIM: int = 0
OUTPUT_CHANNELS: int = 0
N_LAYERS: int = 0 # Only for RNN
SEQ_LENGTH: int = 0
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
ADJOINT_ATOL: float = 0.
ADJOINT_RTOL: float = 0.

# Training data selection parameters
PATIENT_GROUPS: list = []
INDIVIDUAL_NUMBER: int = 0
METHOD: str = ''
NORMALIZE_STANDARDIZE: str = ''
NOISE_MAGNITUDE: float = 0.
IRREGULAR_T_SAMPLES: bool = False
NUM_PER_PATIENT: int = 0
POP_NUMBER: int = 0
BATCH_SIZE: int = 0
LABEL_SMOOTHING: float = 0.
DROPOUT: float = 0.
CORT_ONLY: bool = False
T_END: int = 0

# Define the device with which to train networks
DEVICE = torch.device('cpu')


# Flag to track whether we should be graphing the mechanistic and predicted dy
#  on the current iteration
GRAPH_FLAG:bool = False


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
        loader, (t_steps, t_start, t_end, t_eval) = load_data(
            virtual=virtual,
            control_combination=control_combination,
            mdd_combination=mdd_combination, by_lab=by_lab,
            plus_ableson_mdd=plus_ableson_mdd
        )
        info = {
            'virtual': virtual,
            'ctrl_num': ctrl_num,
            'control_combination': control_combination,
            'mdd_num': mdd_num,
            'mdd_combination': mdd_combination,
            'by_lab': by_lab,
            't_steps': t_steps,
            't_start': t_start,
            't_end': t_end,
            't_eval': t_eval,
        }

        model = model_init(info)
        run_training(model, loader, info)


def train_single(virtual: bool, ableson_pop: bool=False, toy_data: bool=False,
                 domain: torch.Tensor | None=None):
    loader, (t_steps, t_start, t_end, t_eval) = load_data(
        virtual, POP_NUMBER, ableson_pop=ableson_pop, toy_data=toy_data
    )

    info = {
        'virtual': virtual,
        'toy_data': toy_data,
        't_steps': t_steps,
        't_start': t_start,
        't_end': t_end,
        't_eval': t_eval,
    }
    model = model_init(info)

    if MECHANISTIC:
        domain = torch.linspace(0, T_END, SEQ_LENGTH, requires_grad=True, dtype=torch.double).to(DEVICE)
    else:
        domain = None

    if MECHANISTIC and not toy_data:
        params = param_init_tsst(model)
        for key, val in params.items():
            model.register_parameter(key, val)
        bounds = []
        for key, val in params.items():
            model.register_parameter(key, val)
            val = val.item()
            d = np.abs(val)*0.5
            bounds.append((val-d, val+d))

        for (data, _) in loader:
            y0 = torch.cat([torch.tensor([0]), data[0,0,1:], torch.tensor([0])])
            res = differential_evolution(de_loss, bounds, args=(data,y0), disp=True, maxiter=100)

        for idx, val in enumerate(params.values()):
            val = res.x[idx]

    elif MECHANISTIC and toy_data:
        params = param_init(model)
    else:
        params = {}
    run_training(model, loader, info, params, domain=domain)


def load_data(virtual: bool=True, pop_number: int=0,
              control_combination: tuple=(),
              mdd_combination: tuple=(), patient_groups: list=[],
              by_lab: bool=False, ableson_pop: bool=False,
              plus_ableson_mdd: bool=False, test: bool=False,
              toy_data: bool=False):
    if not patient_groups:
        patient_groups = PATIENT_GROUPS
    if toy_data and not virtual:
        dataset = SriramSimulation(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE
        )
    elif not virtual and control_combination and len(patient_groups) > 1 and patient_groups[1]=='MDD':
        # Pretty annoying training with non-augmented data from both labs,
        #  I'm just going to load the raw data then mask it to remove the
        #  test patients (and I'll do the opposite mask when testing)
        nelson_control = NelsonData(
            patient_groups=[patient_groups[0]],
            normalize_standardize=NORMALIZE_STANDARDIZE,
        )
        nelson_mdd = NelsonData(
            patient_groups=['Atypical', 'Melancholic', 'Neither'],
            normalize_standardize=NORMALIZE_STANDARDIZE,
        )
        ableson_control = AblesonData(
            patient_groups=[patient_groups[0]],
            normalize_standardize=NORMALIZE_STANDARDIZE,
        )
        ableson_mdd = AblesonData(
            patient_groups=[patient_groups[1]],
            normalize_standardize=NORMALIZE_STANDARDIZE,
        )

        control_dataset_tmp = ConcatDataset((nelson_control, ableson_control))
        mdd_dataset_tmp = ConcatDataset((nelson_mdd, ableson_mdd))

        control_mask = [i not in control_combination for i in range(len(control_dataset_tmp))]
        mdd_mask = [i not in mdd_combination for i in range(len(mdd_dataset_tmp))]

        # Pretty frustrating that Datasets can't just handle masks as indices
        control_dataset = []
        for i in range(len(control_dataset_tmp)):
            if control_mask[i]:
                control_dataset.append(control_dataset_tmp[i])
        mdd_dataset = []
        for i in range(len(mdd_dataset_tmp)):
            if mdd_mask[i]:
                mdd_dataset.append(mdd_dataset_tmp[i])

        dataset = ConcatDataset((control_dataset, mdd_dataset))

    elif not virtual and control_combination:
        dataset = NelsonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
        )
    elif not virtual:
        dataset = NelsonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            individual_number=INDIVIDUAL_NUMBER,
        ) if not ableson_pop else AblesonData(
            patient_groups=patient_groups,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            individual_number=INDIVIDUAL_NUMBER,
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
        # We set the last 3 time points to 0 if they are from the Nelson data
        for idx,(p,_) in enumerate(dataset):
            if p[-1,0] == 140:
                dataset[idx][0][-3:,:] = -10
    else:
        dataset = FullVirtualPopulation(
            method=METHOD,
            noise_magnitude=NOISE_MAGNITUDE,
            normalize_standardize=NORMALIZE_STANDARDIZE,
            num_per_patient=NUM_PER_PATIENT,
            control_combination=control_combination,
            mdd_combination=mdd_combination,
            test=test,
            label_smoothing=LABEL_SMOOTHING,
        )
        # We set the last 3 time points to 0 if they are from the Nelson data
        for idx,(p,_) in enumerate(dataset):
            if p[-1,0] == 140:
                dataset[idx][0][-3:,:] = -10

    if not virtual and INDIVIDUAL_NUMBER:
        t_steps = len(dataset[0][0][...,0])
        t = dataset[0][0][...,0].double()
        t_start = t[0]
        t_end = t[-1]
    else:
        t_steps = len(dataset[0][0][...,0])
        t = dataset[0][0][...,0]
        t_start = t[0]
        t_end = t[-1]
    loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True,
        # num_workers=8
    )
    return loader, (t_steps, t_start, t_end, t)


def model_init(info: dict):
    toy_data = info.get('toy_data', False)
    t_steps = info.get('t_steps', 11)
    t_eval = info.get('t_eval', [0,1])
    t_eval = torch.linspace(0, 140, 1000)

    if NETWORK_TYPE in ('NCDE', 'NCDE_LBFGS'):
        return NeuralCDE(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
            t_interval=t_eval.contiguous() if not CLASSIFY else torch.tensor((0,1)),
            device=DEVICE, dropout=DROPOUT, prediction=not CLASSIFY, atol=ATOL,
            rtol=RTOL, adjoint_atol=ADJOINT_ATOL, adjoint_rtol=ADJOINT_RTOL,
        ).double()
    elif NETWORK_TYPE == 'NODE':
        return NeuralODE(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
            device=DEVICE
        )
    elif NETWORK_TYPE == 'ANN':
        return ANN(
            INPUT_CHANNELS*t_steps if not MECHANISTIC else INPUT_CHANNELS, HDIM,
            OUTPUT_CHANNELS*t_steps if not MECHANISTIC else OUTPUT_CHANNELS,
            device=DEVICE
        ).double()
    elif NETWORK_TYPE == 'RNN':
        if MECHANISTIC:
            return RNN(
                INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS, N_LAYERS,
                device=DEVICE,
            ).double()
        return RNN(
            INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS, N_LAYERS,
            device=DEVICE,
        ).double()
    else:
        raise ValueError("NETWORK_TYPE must be one of NCDE, NCDE_LBFGS, NODE, ANN or RNN")


def param_init(model: nn.Module):
    """Initialize the parameters for the mechanistic loss, and set them to
    require gradient"""
    k_stress = torch.nn.Parameter(torch.tensor(10.1, device=DEVICE, dtype=torch.float32), requires_grad=False)
    Ki = torch.nn.Parameter(torch.tensor(1.51, device=DEVICE, dtype=torch.float32), requires_grad=True)
    VS3 = torch.nn.Parameter(torch.tensor(3.25, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Km1 = torch.nn.Parameter(torch.tensor(1.74, device=DEVICE, dtype=torch.float32), requires_grad=True)
    KP2 = torch.nn.Parameter(torch.tensor(8.3, device=DEVICE, dtype=torch.float32), requires_grad=True)
    VS4 = torch.nn.Parameter(torch.tensor(0.907, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Km2 = torch.nn.Parameter(torch.tensor(0.112, device=DEVICE, dtype=torch.float32), requires_grad=True)
    KP3 = torch.nn.Parameter(torch.tensor(0.945, device=DEVICE, dtype=torch.float32), requires_grad=True)
    VS5 = torch.nn.Parameter(torch.tensor(0.00535, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Km3 = torch.nn.Parameter(torch.tensor(0.0768, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Kd1 = torch.nn.Parameter(torch.tensor(0.00379, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Kd2 = torch.nn.Parameter(torch.tensor(0.00916, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Kd3 = torch.nn.Parameter(torch.tensor(0.356, device=DEVICE, dtype=torch.float32), requires_grad=True)
    n1 = torch.nn.Parameter(torch.tensor(5.43, device=DEVICE, dtype=torch.float32), requires_grad=True)
    n2 = torch.nn.Parameter(torch.tensor(5.1, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Kb = torch.nn.Parameter(torch.tensor(0.0202, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Gtot = torch.nn.Parameter(torch.tensor(3.28, device=DEVICE, dtype=torch.float32), requires_grad=True)
    VS2 = torch.nn.Parameter(torch.tensor(0.0509, device=DEVICE, dtype=torch.float32), requires_grad=True)
    K1 = torch.nn.Parameter(torch.tensor(0.645, device=DEVICE, dtype=torch.float32), requires_grad=True)
    Kd5 = torch.nn.Parameter(torch.tensor(0.0854, device=DEVICE, dtype=torch.float32), requires_grad=True)
    kdStress = torch.nn.Parameter(torch.tensor(0.19604, device=DEVICE), requires_grad=True)
    stressStr = torch.nn.Parameter(torch.tensor(1., device=DEVICE), requires_grad=True)

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
        'Kd5': Kd5,
        'kdStress': kdStress,
        'stressStr': stressStr,
    }
    for key, val in params.items():
        model.register_parameter(key, val)

    return params


def param_init_tsst(model: nn.Module):
    """Initialize the parameters for the mechanistic loss, and set them to
    require gradient"""
    R0CRH = torch.nn.Parameter(torch.tensor(-0.52239, device=DEVICE), requires_grad=False)
    RCRH_CRH = torch.nn.Parameter(torch.tensor(0.97555, device=DEVICE), requires_grad=False)
    RGR_CRH = torch.nn.Parameter(torch.tensor(-20.0241, device=DEVICE), requires_grad=False)
    RSS_CRH = torch.nn.Parameter(torch.tensor(9.8594, device=DEVICE), requires_grad=False)
    sigma = torch.nn.Parameter(torch.tensor(4.974, device=DEVICE), requires_grad=False)
    tsCRH = torch.nn.Parameter(torch.tensor(0.10008, device=DEVICE), requires_grad=False)
    R0ACTH = torch.nn.Parameter(torch.tensor(-0.29065, device=DEVICE), requires_grad=False)
    RCRH_ACTH = torch.nn.Parameter(torch.tensor(16.006, device=DEVICE), requires_grad=False)
    RGR_ACTH = torch.nn.Parameter(torch.tensor(-20.004, device=DEVICE), requires_grad=False)
    tsACTH = torch.nn.Parameter(torch.tensor(0.046655, device=DEVICE), requires_grad=False)
    R0CORT = torch.nn.Parameter(torch.tensor(-0.95265, device=DEVICE), requires_grad=False)
    RACTH_CORT = torch.nn.Parameter(torch.tensor(0.022487, device=DEVICE), requires_grad=False)
    tsCORT = torch.nn.Parameter(torch.tensor(0.048451, device=DEVICE), requires_grad=False)
    R0GR = torch.nn.Parameter(torch.tensor(-0.49428, device=DEVICE), requires_grad=False)
    RCORT_GR = torch.nn.Parameter(torch.tensor(0.02745, device=DEVICE), requires_grad=False)
    RGR_GR = torch.nn.Parameter(torch.tensor(0.10572, device=DEVICE), requires_grad=False)
    kdStress = torch.nn.Parameter(torch.tensor(0.19604, device=DEVICE), requires_grad=False)
    MaxCRH = torch.nn.Parameter(torch.tensor(30., device=DEVICE), requires_grad=False)
    MaxACTH = torch.nn.Parameter(torch.tensor(140.2386, device=DEVICE), requires_grad=False)
    MaxCORT = torch.nn.Parameter(torch.tensor(30.3072, device=DEVICE), requires_grad=False)
    BasalACTH = torch.nn.Parameter(torch.tensor(0.84733, device=DEVICE), requires_grad=False)
    BasalCORT = torch.nn.Parameter(torch.tensor(0.29757, device=DEVICE), requires_grad=False)
    ksGR = torch.nn.Parameter(torch.tensor(0.40732, device=DEVICE), requires_grad=False)
    kdGR = torch.nn.Parameter(torch.tensor(0.39307, device=DEVICE), requires_grad=False)

    params = {
        'R0CRH': R0CRH,
        'RCRH_CRH': RCRH_CRH,
        'RGR_CRH': RGR_CRH,
        'RSS_CRH': RSS_CRH,
        'sigma': sigma,
        'tsCRH': tsCRH,
        'R0ACTH': R0ACTH,
        'RCRH_ACTH': RCRH_ACTH,
        'RGR_ACTH': RGR_ACTH,
        'tsACTH': tsACTH,
        'R0CORT': R0CORT,
        'RACTH_CORT': RACTH_CORT,
        'tsCORT': tsCORT,
        'R0GR': R0GR,
        'RCORT_GR': RCORT_GR,
        'RGR_GR': RGR_GR,
        'kdStress': kdStress,
        'MaxCRH': MaxCRH,
        'MaxACTH': MaxACTH,
        'MaxCORT': MaxCORT,
        'BasalACTH': BasalACTH,
        'BasalCORT': BasalCORT,
        'ksGR': ksGR,
        'kdGR': kdGR,
    }
    for key, val in params.items():
        model.register_parameter(key, val)

    return params


def run_training(model: NeuralODE | NeuralCDE | ANN | RNN, loader: DataLoader,
                 info: dict, params: dict={}, domain: torch.Tensor | None=None):
    """Run the training procedure for the given model and DataLoader"""
    virtual = info.get('virtual')
    control_combination = info.get('control_combination')
    mdd_combination = info.get('mdd_combination')
    toy_data = info.get('toy_data', False)

    # Print which population we are using to train
    if not virtual and INDIVIDUAL_NUMBER:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]} #{INDIVIDUAL_NUMBER}')
    elif not virtual and len(PATIENT_GROUPS) == 1:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]}')
    elif not virtual:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]} '
              f'vs {PATIENT_GROUPS[1]}')
    elif POP_NUMBER:
        print(f'Starting Training w/ {PATIENT_GROUPS[1]} '
              f'Population Number {POP_NUMBER}')
    elif len(PATIENT_GROUPS) == 1:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]}')
    else:
        print(f'Starting Training w/ {PATIENT_GROUPS[0]} {control_combination}'
              f' vs {PATIENT_GROUPS[1]} {mdd_combination}')

    start_time = time.time()

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=DECAY
    )

    loss_over_time = []


    # We need to define the readout layer outside of the loop so that it
    #  isn't reset each iteration
    # Only necessary for NODE and RNN
    readout = None
    if NETWORK_TYPE == 'NODE':
        readout = nn.Linear(OUTPUT_CHANNELS, 1).to(DEVICE)
    # elif NETWORK_TYPE == 'RNN':
    #     readout = nn.Linear(HDIM, 1).double().to(DEVICE)

    for itr in range(1, ITERS+1):
        match NETWORK_TYPE:
            case 'NCDE':
                ncde_training_epoch(itr, loader, model, optimizer, loss_over_time, domain, info,
                                    toy_data=toy_data)
            case 'NCDE_LBFGS':
                optimizer = optim.LBFGS(
                    model.parameters()
                )
                ncde_training_epoch_lbfgs(itr, loader, model, optimizer, loss_over_time, domain, info,
                                    toy_data=toy_data)
            case 'NODE':
                node_training_epoch(itr, loader, model, readout,
                                    optimizer, loss_over_time, domain, params, info, toy_data=toy_data)
            case 'ANN':
                if MECHANISTIC:
                    ann_training_epoch_mech(itr, loader, model, optimizer, loss_over_time,
                                   domain, params, info, toy_data=toy_data)
                else:
                    ann_training_epoch(itr, loader, model, optimizer, loss_over_time,
                                       domain, info, toy_data=toy_data)
            case 'RNN':
                if MECHANISTIC:
                    rnn_training_epoch_mech(itr, loader, model, readout, optimizer, loss_over_time,
                                   domain, params, info, toy_data=toy_data)
                else:
                    rnn_training_epoch(itr, loader, model, readout,
                                       optimizer, loss_over_time, domain, info, toy_data=toy_data)
            case _:
                raise ValueError("NETWORK_TYPE must be one of NCDE, NODE, ANN, or RNN")

        # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
        #  reset the learning rate and momentum
        if not OPT_RESET:
            pass
        elif itr % OPT_RESET == 0:
            optimizer = optim.AdamW(
                model.parameters(), lr=LR, weight_decay=DECAY
            )

        if itr % SAVE_FREQ == 0:
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
            save_network(model, readout, optimizer, save_info)

        global GRAPH_FLAG
        global ITR
        if (itr+1) % 1000 == 0:
            GRAPH_FLAG = True
            ITR = itr+1
        else:
            GRAPH_FLAG = False
            ITR = itr+1


class SplitBatchDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.X = data
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx]



def ncde_training_epoch(itr: int, loader: DataLoader, model: NeuralCDE,
                        optimizer: optim.AdamW, loss_over_time: list, domain: torch.Tensor,  info: dict,
                        toy_data: bool=False):
    t_eval = info.get('t_eval', [0,1])
    global BATCH_SIZE
    for j, (data, labels) in enumerate(loader):
        i = 0
        nelson_idx = []
        # Iterate through the patients in the current data and check whether
        #  they have 0 in the last 3 time points (in which case they are
        #  truncated Nelson data
        for idx, pt in enumerate(data):
            if -10 in pt[-3:,0]:
                i += 1
                nelson_idx.append(idx)
        # If we have the same number of Nelson patients as patients in the
        #  batch, we just cut the last 3 time points off all patients
        if i == BATCH_SIZE:
            data = data[:,:-3,:]
        # Otherwise, split the dataset into Nelson and Ableson, then
        #  recursively call ncde_training_epoch on the Nelson data with
        #  BATCH_SIZE set i
        elif i != 0:
            batch_size_old = BATCH_SIZE
            data_nelson = data[nelson_idx,:-3,:]
            labels_nelson = labels[nelson_idx]
            dataset_nelson = SplitBatchDataset(data_nelson, labels_nelson)
            loader_nelson = DataLoader(dataset_nelson, batch_size=i)
            BATCH_SIZE = i
            ncde_training_epoch(
                itr, loader_nelson, model, optimizer, loss_over_time, domain, info
            )
            BATCH_SIZE = batch_size_old
            # Now that we've run the Nelson data, we just cut it out of the
            #  currently loaded data and continue as usual
            data = data[[i not in nelson_idx for i in range(BATCH_SIZE)],...]
            labels = labels[[i not in nelson_idx for i in range(BATCH_SIZE)]]

        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,[0,2]]
        if toy_data and INPUT_CHANNELS == 3:
            data = data[...,[0,2,3]]
        # If we are using prediction mode instead of classification, we need
        #  the data and labels to be the same
        t_eval = data[0,:,0]
        data = data.double().to(DEVICE) if CLASSIFY else data[...,1:].double().to(DEVICE)
        labels = labels.to(DEVICE) if CLASSIFY else data
        coeffs = torchcde.\
            hermite_cubic_coefficients_with_backward_differences(
                data, t=t_eval
            )

        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # Compute the forward direction of the NCDE
        pred_y = model(coeffs).squeeze(-1)

        # for using only the datapoints for the data loss, rather than a spline
        def find_nearest(array, value):
            idx = (torch.abs(array-value)).argmin()
            return idx

        pred_y_datapoints = pred_y[:,[find_nearest(torch.linspace(0,140,1000), t) for t in t_eval.squeeze()],...]
        # Compute the loss based on the results
        output = loss(pred_y_datapoints, labels, domain=domain)

        loss_over_time.append((j, output.item()))

        # Backpropagate through the adjoint of the NODE to compute gradients
        #  WRT each parameter
        output.backward()

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step()

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")


def ncde_training_epoch_lbfgs(itr: int, loader: DataLoader, model: NeuralCDE,
                              optimizer: optim.AdamW | optim.LBFGS, loss_over_time: list, domain: torch.Tensor, info: dict,
                              toy_data: bool=False):

    for j, (data, labels) in enumerate(loader):
        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,[0,2]]
        if toy_data and INPUT_CHANNELS == 3:
            data = data[...,[0,2,3]]
        data = data.double().to(DEVICE)
        labels = labels.to(DEVICE) if CLASSIFY else data
        coeffs = torchcde.\
            hermite_cubic_coefficients_with_backward_differences(
                data
            )

        def closure():
            # Zero the gradient from the previous data
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            # Compute the forward direction of the NCDE
            pred_y = model(coeffs).squeeze(-1)

            # Compute the loss based on the results
            output = loss(pred_y, labels, domain=domain)

            # This happens in place, so we don't need to return loss_over_time
            loss_over_time.append((j, output.item()))

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            if output.requires_grad:
                output.backward()

            return output

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step(closure)

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {closure().item():.6f}")


def node_training_epoch(itr: int, loader: DataLoader, model: NeuralODE,
                        readout: nn.Linear, optimizer: optim.AdamW,
                        loss_over_time: list, domain: torch.Tensor, params: dict,
                        info: dict, toy_data: bool=False):
    virtual = info.get('virtual', True)
    for j, (data, labels) in enumerate(loader):
        if toy_data and not virtual:
            data = data.squeeze(0)
        t_eval = data[0,:,0].view(-1).to(DEVICE)
        t_eval.requires_grad = True

        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,2]
        if toy_data and INPUT_CHANNELS == 2:
            data = data[...,[2,3]]
        else:
            data = data[...,1:]
        # data = data.double().to(DEVICE)
        data = data.to(DEVICE)
        labels = labels.to(DEVICE) if CLASSIFY else data
        y0 = data[:,0,:].to(DEVICE)
        # model = model.double()
        # readout = readout.double()
        # data = data.double()
        # labels = labels.double()
        # t_eval = t_eval.double()
        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # Compute the forward direction of the NODE
        pred_y = odeint_adjoint(
            model, y0, t_eval, atol=ATOL, rtol=RTOL, method='dopri5',
            adjoint_atol=ADJOINT_ATOL, adjoint_rtol=ADJOINT_RTOL,
        )
        # Compute the forward direction of the NODE with the dense domain for
        #  use in the mechanistic loss
        # dense_pred_y = None
        # if MECHANISTIC:
        #     dense_pred_y = odeint_adjoint(
        #         model, y0, domain
        #     ).squeeze().to(DEVICE)
        # We need to take the output_channels down to a single output, then
        #  we only need the last value (so the value after the entire depth of
        #  the network)
        # We squeeze to remove the extraneous dimension of the output so that
        #  it matches the shape of the labels
        pred_y = readout(pred_y)[-1].squeeze(-1) if CLASSIFY else pred_y.squeeze(-1)

        # Compute the loss based on the results
        output = loss(pred_y, labels, dense_pred_y=pred_y, domain=t_eval, params=params)

        # This happens in place, so we don't need to return loss_over_time
        loss_over_time.append((j, output.item()))

        # Backpropagate through the adjoint of the NODE to compute gradients
        #  WRT each parameter
        output.backward()

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step()

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 100 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")


def ann_training_epoch_mech(itr: int, loader: DataLoader, model: MechanisticANN,
                            optimizer: optim.AdamW, loss_over_time: list,
                            domain: torch.Tensor, params: dict, info: dict,
                            toy_data: bool=False):

    for j, (data, labels) in enumerate(loader):
        data_domain = data[...,0].double()
        # data_domain.requires_grad = True
        # Check how many patients are in the batch (as it may be less than
        #  BATCH_SIZE if it's the last batch)
        batch_size_ = data.size(0)
        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,2]
        elif toy_data and INPUT_CHANNELS == 2:
            data = data[...,[2,3]]
        else:
            data = data[...,1:]
        data = data.double().to(DEVICE)
        labels = labels.to(DEVICE) if CLASSIFY else data

        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # We run the model twice. Once with the time steps from the data points
        #  for the data loss, then again with a more dense selection of time
        #  steps over the domain
        pred_y = model(data_domain.view(-1,1))
        # Remove the extraneous dimension from the output of the network so the
        #  shape matches the labels
        pred_y = pred_y.squeeze()

        # Repeat for the dense predictions
        dense_pred_y = model(domain.double().view(-1,1))

        # Compute the loss based on the results
        output = loss(
            pred_y, labels=labels,
            dense_pred_y=dense_pred_y, domain=domain, params=params
        )
        (mech_loss, data_loss, ic_loss) = output
        output = (mech_loss + data_loss + ic_loss)/3

        # This happens in place, so we don't need to return loss_over_time
        loss_over_time.append((j, output.item()))

        # Backpropagate through the adjoint of the NODE to compute gradients
        #  WRT each parameter
        output.backward()

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step()

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"{mech_loss=}")
            print(f"{data_loss=}")
            print(f"{ic_loss=}")
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")
        # print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")


def ann_training_epoch(itr: int, loader: DataLoader, model: ANN,
                       optimizer: optim.AdamW, loss_over_time: list, domain: torch.Tensor, info: dict,
                       toy_data: bool=False):

    for j, (data, labels) in enumerate(loader):
        # Check how many patients are in the batch (as it may be less than
        #  BATCH_SIZE if it's the last batch)
        batch_size_ = data.size(0)
        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,2]
        elif toy_data and INPUT_CHANNELS == 2:
            data = data[...,[2,3]]
        else:
            data = data[...,1:]
        data = data.double().to(DEVICE)
        labels = labels.to(DEVICE) if CLASSIFY else data

        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # We pass the data from the entire batch at once, shaped to fit the ANN
        pred_y = model(
            data.reshape(batch_size_, INPUT_CHANNELS)
        )
        # Remove the extraneous dimension from the output of the network so the
        #  shape matches the labels
        # We also need to reshape the output so that it has the number of
        #  columns necessary
        pred_y = pred_y.squeeze(-1).reshape(-1, labels.size(1), labels.size(2))

        # Compute the loss based on the results
        output = loss(
            pred_y, labels=labels,
            dense_pred_y=pred_y, domain=domain
        )

        # This happens in place, so we don't need to return loss_over_time
        loss_over_time.append((j, output.item()))

        # Backpropagate through the adjoint of the NODE to compute gradients
        #  WRT each parameter
        output.backward()

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step()

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")


def rnn_training_epoch_mech(itr: int, loader: DataLoader, model: RNN,
                            readout: nn.Linear, optimizer: optim.AdamW,
                            loss_over_time: list, domain: torch.Tensor,
                            params: dict, info: dict, toy_data: bool=False):

    for j, (data, labels) in enumerate(loader):
        data_domain = data[...,0].contiguous()

        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,2]
        elif toy_data and INPUT_CHANNELS == 2:
            data = data[...,[2,3]]
        else:
            data = data[...,1:]
        data = data.to(DEVICE)
        labels = data

        data = data.to(torch.double)
        labels = labels.to(torch.double)

        # Let's create a spline over the data points and then use that to compute
        #  the data loss (so that we can force points in between datapoints to
        #  more closely align)
        coeffs = natural_cubic_spline_coeffs(data_domain.squeeze(), labels)
        spline = NaturalCubicSpline(coeffs)
        info.update({'spline': spline})

        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # We pass the data from the entire batch at once, shaped to fit the ANN

        # We run the model twice. Once with the time steps from the data points
        #  for the data loss, then again with a more dense selection of time
        #  steps over the domain
        # pred_y = model(data_domain.view(-1,1)).squeeze()

        # for using only the datapoints for the data loss, rather than a spline
        def find_nearest(array, value):
            idx = (torch.abs(array-value)).argmin()
            return idx

        # Repeat for the dense predictions
        pred_y = model(domain.view(-1,1))
        labels = spline.evaluate(domain)
        # pred_y_datapoints = pred_y[[find_nearest(domain, t) for t in data_domain.squeeze()],...]

        # if loss_over_time and loss_over_time[-1][1] <= 1:
        #     set_trace()

        # Compute the loss based on the results
        output = loss(
            # pred_y_datapoints, labels=labels,
            pred_y, labels=labels,
            dense_pred_y=pred_y, domain=domain,
            params=params, info=info
        )
        if not toy_data:
            (mech_loss, data_loss, ic_loss) = output
            output = (3*mech_loss + data_loss)/4
        else:
            (mech_loss, data_loss, ic_loss) = output
            output = (mech_loss + data_loss + ic_loss)/3

        # This happens in place, so we don't need to return loss_over_time
        loss_over_time.append((j, output.item()))

        # Backpropagate through the adjoint of the NODE to compute gradients
        #  WRT each parameter
        output.backward(retain_graph=True)

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step()

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f} (mech_loss = {mech_loss.item():.6f}, data_loss = {data_loss.item():.6f}, ic_loss = {ic_loss.item() if ic_loss else 0:.6f})")


def rnn_training_epoch(itr: int, loader: DataLoader, model: RNN,
                       readout: nn.Linear, optimizer: optim.AdamW,
                       loss_over_time: list, domain: torch.Tensor, info: dict, toy_data: bool=False):

    for j, (data, labels) in enumerate(loader):
        # Check how many patients are in the batch (as it may be less than
        #  BATCH_SIZE if it's the last batch)
        batch_size_ = data.size(0)
        # Ensure we have assigned the data and labels to the correct
        #  processing device
        if CORT_ONLY:
            # If we are only using CORT, we can discard the middle column as it
            #  contains the ACTH concentrations
            data = data[...,2]
        elif toy_data and INPUT_CHANNELS == 2:
            data = data[...,[2,3]]
        else:
            data = data[...,1:]
        data = data.double().to(DEVICE)
        labels = labels.to(DEVICE) if CLASSIFY else data

        # Zero the gradient from the previous data
        optimizer.zero_grad()

        # We pass the data from the entire batch at once, shaped to fit the ANN
        pred_y = model(
            data.reshape(batch_size_, INPUT_CHANNELS)
        )
        # Remove the extraneous dimension from the output of the network so the
        #  shape matches the labels
        pred_y = readout(pred_y).squeeze(-1).reshape(
            -1, labels.size(1), labels.size(2)
        ) if CLASSIFY else pred_y.squeeze(-1)

        # Compute the loss based on the results
        output = loss(pred_y, labels, domain)

        # This happens in place, so we don't need to return loss_over_time
        loss_over_time.append((j, output.item()))

        # Backpropagate through the adjoint of the NODE to compute gradients
        #  WRT each parameter
        output.backward()

        # Use the gradients calculated through backpropagation to adjust the
        #  parameters
        optimizer.step()

        # If this is the first iteration, or a multiple of 10, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")


def loss(pred_y: torch.Tensor, labels: torch.Tensor=torch.ones((1,20,4)),
         dense_pred_y: torch.Tensor | None=None,
         domain: torch.Tensor | None=None,
         params: dict={}, info: dict={}, **kwargs):
    """To compute the loss with potential mechanistic components"""
    toy_data = info.get('toy_data', False)
    spline = info.get('spline')
    if CLASSIFY:
        return binary_cross_entropy_with_logits(pred_y, labels)

    if MECHANISTIC:
        mech_loss_total = 0
        # We loop through all of the patients in the batch and compute the
        #  mechanistic loss for each (I can probably get this to run in one
        #  shot, but this is easier for the moment).
        if NETWORK_TYPE == 'NODE':
            for y in dense_pred_y.reshape(BATCH_SIZE, domain.size(0), INPUT_CHANNELS if NETWORK_TYPE not in ['ANN', 'RNN'] else 0).to(DEVICE):
                mech_loss = mechanistic_loss(y, domain, params, info)
                mech_loss_total += mech_loss
        elif not toy_data:
            mech_loss = mechanistic_loss_tsst(dense_pred_y, domain, params, info)
            mech_loss_total = mech_loss
        else:
            mech_loss = mechanistic_loss(dense_pred_y, domain, params, info)
            mech_loss_total = mech_loss
        mech_loss = mech_loss_total/BATCH_SIZE

        labels = labels.squeeze()

        data_loss = l1_loss(dense_pred_y if toy_data else pred_y[...,[1,2]], labels)
        if GRAPH_FLAG:
            graph_data_loss(domain, pred_y, dense_pred_y, labels, spline)
        ic_loss = l1_loss(torch.tensor([1, 7.14, 2.38, 2]), pred_y[...,0,:]) if toy_data else None
        return (mech_loss, data_loss, ic_loss)

    return l1_loss(pred_y, labels.reshape(pred_y.shape))
        # data_loss = mse_loss(dense_pred_y if toy_data else pred_y[...,[1,2]], labels)
        # if GRAPH_FLAG:
        #     graph_data_loss(domain, pred_y, dense_pred_y, labels, spline)
        # ic_loss = mse_loss(torch.tensor([1, 7.14, 2.38, 2]), pred_y[...,0,:]) if toy_data else None
        # return (mech_loss, data_loss, ic_loss)

    # return mse_loss(pred_y, labels.reshape(pred_y.shape))


# def mechanistic_loss(pred_crh: torch.Tensor, pred_acth: torch.Tensor,
#                      pred_cort: torch.Tensor, pred_gr: torch.Tensor,
#                      domain: torch.Tensor, params: dict):
def mechanistic_loss(pred_y, domain, params, info):
    """Here we combine the predicted y with the mechanistic knowledge and then
    compute the loss against the experimental data"""
    pred_dy = torch.zeros_like(pred_y).to(DEVICE)
    pred_dy[...,0] = torch.autograd.grad(
        pred_y[...,0], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]
    pred_dy[...,1] = torch.autograd.grad(
        pred_y[...,1], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]
    pred_dy[...,2] = torch.autograd.grad(
        pred_y[...,2], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]
    pred_dy[...,3] = torch.autograd.grad(
        pred_y[...,3], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Apologies for the mess here, but typing out ODEs in Python is a bit of a
    #  chore
    stress = torch.zeros_like(domain, requires_grad=False)
    for idx, t in enumerate(domain):
        if t < 30:
            stress[idx] = 0
            continue
        stress[idx] = params['stressStr']*torch.exp(-params['kdStress']*(t-30))
    mechanistic_dcrh = torch.abs(stress)*((params['Ki']**params['n2'])/(params['Ki']**params['n2'] + torch.sign(pred_y[...,3])*(torch.abs(pred_y[...,3])**params['n2']))) - params['VS3']*(pred_y[...,0]/(params['Km1'] + pred_y[...,0])) - params['Kd1']*pred_y[...,0]
    mechanistic_dacth = params['KP2']*pred_y[...,0]*((params['Ki']**params['n2'])/(params['Ki']**params['n2'] + torch.sign(pred_y[...,3])*(torch.abs(pred_y[...,3])**params['n2']))) - params['VS4']*(pred_y[...,1]/(params['Km2'] + pred_y[...,0])) - params['Kd2']*pred_y[...,0]
    mechanistic_dcort = params['KP3']*pred_y[...,1] - params['VS5']*(pred_y[...,2]/(params['Km3'] + pred_y[...,2])) - params['Kd3']*pred_y[...,2]
    mechanistic_dgr = params['Kb']*pred_y[...,2]*(params['Gtot'] - pred_y[...,3]) + params['VS2']*(torch.sign(pred_y[...,3])*(torch.abs(pred_y[...,3])**params['n1'])/(params['K1']**params['n1'] + torch.sign(pred_y[...,3])*(torch.abs(pred_y[...,3])**params['n1']))) - params['Kd5']*pred_y[...,3]
    mechanistic_dy = torch.cat([mechanistic_dcrh.view(-1,1), mechanistic_dacth.view(-1,1), mechanistic_dcort.view(-1,1), mechanistic_dgr.view(-1,1)], dim=1)

    if GRAPH_FLAG:
        graph_mech_loss(domain, pred_dy, mechanistic_dy)

    return mse_loss(pred_dy, mechanistic_dy)


def mechanistic_loss_tsst(pred_y, domain, params, info):
    """Here we combine the predicted y with the mechanistic knowledge and then
    compute the loss against the experimental data"""
    stress = info.get('stress', torch.zeros_like(domain))

    pred_dy = torch.zeros_like(pred_y).to(DEVICE)
    pred_dy[...,0] = torch.autograd.grad(
        pred_y[...,0], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]
    pred_dy[...,1] = torch.autograd.grad(
        pred_y[...,1], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]
    pred_dy[...,2] = torch.autograd.grad(
        pred_y[...,2], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]
    pred_dy[...,3] = torch.autograd.grad(
        pred_y[...,3], domain,
        grad_outputs=torch.ones_like(domain),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Apologies for the mess here, but typing out ODEs in Python is a bit of a
    #  chore
    stress = torch.zeros_like(domain, requires_grad=False)
    for idx, t in enumerate(domain):
        if t < 30:
            stress[idx] = 0
            continue
        stress[idx] = torch.exp(-params['kdStress']*(t-30))
    wCRH = params['R0CRH'] + params['RCRH_CRH']*pred_y[...,0] \
        + params['RSS_CRH']*torch.abs(stress.squeeze()) + params['RGR_CRH']*pred_y[...,3]
    FCRH = (params['MaxCRH']*params['tsCRH'])/(1 + torch.exp(-params['sigma']*wCRH))
    mechanistic_dcrh = FCRH - params['tsCRH']*pred_y[...,0]

    wACTH = params['R0ACTH'] + params['RCRH_ACTH']*pred_y[...,0] \
        + params['RGR_ACTH']*pred_y[...,3]
    FACTH = (params['MaxACTH']*params['tsACTH'])/(1 + torch.exp(-params['sigma']*wACTH)) + params['BasalACTH']
    mechanistic_dacth = FACTH - params['tsACTH']*pred_y[...,1]

    wCORT = params['R0CORT'] + params['RACTH_CORT']*pred_y[...,1]
    FCORT = (params['MaxCORT']*params['tsCORT'])/(1 + torch.exp(-params['sigma']*wCORT)) + params['BasalCORT']
    mechanistic_dcort = FCORT - params['tsCORT']*pred_y[...,2]

    wGR = params['R0GR'] + params['RCORT_GR']*pred_y[...,2] + params['RGR_GR']*pred_y[...,3]
    FGR = params['ksGR']/(1 + torch.exp(-params['sigma']*wGR))
    mechanistic_dgr = FGR - params['kdGR']*pred_y[...,3]

    mechanistic_dy = torch.cat([mechanistic_dcrh.view(-1,1), mechanistic_dacth.view(-1,1), mechanistic_dcort.view(-1,1), mechanistic_dgr.view(-1,1)], dim=1)

    if GRAPH_FLAG:
        graph_mech_loss(domain, pred_dy, mechanistic_dy)


    return l1_loss(pred_dy, mechanistic_dy)


def graph_mech_loss(domain: torch.tensor, pred_dy: torch.tensor, mechanistic_dy: torch.tensor) -> None:
    with torch.no_grad():
        fig, axes = plt.subplots(nrows=4, figsize=(10,8))
        for idx in range(pred_dy.shape[-1]):
            axes[idx].plot(domain, pred_dy[...,idx], label='Predicted dy')
            axes[idx].plot(domain, mechanistic_dy[...,idx], label='Mechanistic dy')
            axes[idx].legend(fancybox=True, shadow=True, loc='upper right')
        plt.savefig(f'Results/pred_dy_vs_mech_dy_{ITR}ITERS_{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}.png', dpi=300)
        plt.close(fig)


def graph_data_loss(domain: torch.tensor, pred_y: torch.tensor, dense_pred_y: torch.tensor, labels: torch.tensor, spline) -> None:
    with torch.no_grad():
        fig, axes = plt.subplots(nrows=2, figsize=(10,8))
        for idx in range(labels.shape[-1]):
            axes[idx].plot(domain, pred_y[...,idx+1], label='Predicted y')
            # axes[idx].plot(domain, dense_pred_y[...,idx+1], label='Dense Predicted y')
            axes[idx].plot(domain, labels[...,idx], label='Data y')
            axes[idx].legend(fancybox=True, shadow=True, loc='upper right')
        plt.savefig(f'Results/pred_y_vs_data_{ITR}ITERS_{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}.png', dpi=300)
        plt.close(fig)


def de_func(params, y0):
    """ODE System computation for the differential_evolution runs"""
    if isinstance(params, dict):
        vals = []
        for _, val in params.items():
            vals.append(val.item())
        [R0CRH, RCRH_CRH, RGR_CRH, RSS_CRH, sigma, tsCRH, R0ACTH, RCRH_ACTH, RGR_ACTH, tsACTH, R0CORT, RACTH_CORT, tsCORT, R0GR, RCORT_GR, RGR_GR, kdStress, MaxCRH, MaxACTH, MaxCORT, BasalACTH, BasalCORT, ksGR, kdGR] = vals
    else:
        [R0CRH, RCRH_CRH, RGR_CRH, RSS_CRH, sigma, tsCRH, R0ACTH, RCRH_ACTH, RGR_ACTH, tsACTH, R0CORT, RACTH_CORT, tsCORT, R0GR, RCORT_GR, RGR_GR, kdStress, MaxCRH, MaxACTH, MaxCORT, BasalACTH, BasalCORT, ksGR, kdGR] = params
    """This function solves the system to find the true mechanistic component
    for graphing."""
    def stress(t):
        if t < 30:
            return 0
        return np.exp(-kdStress*(t-30))
    def ode_rhs(y, t):
        dy = np.zeros(4)

        # Apologies for the mess here, but typing out ODEs in Python is a bit of a
        #  chore
        wCRH = R0CRH + RCRH_CRH*y[0] \
            + RSS_CRH*stress(t) + RGR_CRH*y[3]
        FCRH = (MaxCRH*tsCRH)/(1 + np.exp(-sigma*wCRH))
        dy[0] = FCRH - tsCRH*y[0]

        wACTH = R0ACTH + RCRH_ACTH*y[0] \
            + RGR_ACTH*y[3]
        FACTH = (MaxACTH*tsACTH)/(1 + np.exp(-sigma*wACTH)) + BasalACTH
        dy[1] = FACTH - tsACTH*y[1]

        wCORT = R0CORT + RACTH_CORT*y[1]
        FCORT = (MaxCORT*tsCORT)/(1 + np.exp(-sigma*wCORT)) + BasalCORT
        dy[2] = FCORT - tsCORT*y[2]

        wGR = R0GR + RCORT_GR*y[2] + RGR_GR*y[3]
        FGR = ksGR/(1 + np.exp(sigma*wGR))
        dy[3] = FGR - kdGR*y[3]
        return dy

    t_eval = torch.tensor((0, 15, 30, 40, 50, 65, 80, 95, 110, 125, 140))
    gflow = odeint(ode_rhs, y0, t_eval)
    gflow = np.concatenate((t_eval.view(-1,1), gflow), axis=1)
    gflow = torch.from_numpy(gflow)
    return gflow


def de_loss(params, patient, y0):
    """Loss for the differential_evolution computations"""
    gflow = de_func(params, y0)

    mse = gflow[:,[2,3]] - patient[...,1:]
    mse = mse**2
    mse = torch.mean(mse)
    return mse


def save_network(model: NeuralCDE | NeuralODE | ANN | RNN,
                 readout: nn.Linear | None, optimizer: optim.AdamW,
                 info: dict):
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
    if not virtual and len(PATIENT_GROUPS) == 1:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'{PATIENT_GROUPS[0] if not toy_data else "Toy Dataset"}/'
        )
    elif not virtual and control_combination:
        directory = (
            f'Network States/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
            f'Control vs {PATIENT_GROUPS[1]}/'
            f'{PATIENT_GROUPS[0]} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1]} {mdd_num} {mdd_combination}/'
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
            f'{PATIENT_GROUPS[0]} '
            f'{ctrl_num} {control_combination}/'
            f'{PATIENT_GROUPS[1]} {mdd_num} {mdd_combination}/'
        )
    else:
        directory = (
            f'Network States (VPOP Training)/'
            f'{"Classification" if CLASSIFY else "Prediction"}/'
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
            f'{"_irregularSamples" if IRREGULAR_T_SAMPLES else ""}.txt'
        )
    elif INDIVIDUAL_NUMBER and len(PATIENT_GROUPS) == 1:
        filename = (
            f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
            f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
            f'batchsize{BATCH_SIZE}_'
            f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}.txt'
        )
    elif not virtual and len(PATIENT_GROUPS) == 1:
        filename = (
            f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
            f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
            f'batchsize{BATCH_SIZE}_'
            f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}.txt'
        )
    elif not virtual:
        filename = (
            f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
            f'Control_vs_{PATIENT_GROUPS[1]}_'
            f'batchsize{BATCH_SIZE}_'
            f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}.txt'
        )
    else:
        filename = (
            f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
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
        )

    # Add _setup to the filename before the .txt extension
    setup_filename = "".join([filename[:-4], "_setup", filename[-4:]])

    # Add _readout to the filename before the .txt extension if readout is an
    #  instance of nn.Linear
    readout_filename = None
    if isinstance(readout, nn.Linear):
        readout_filename = "".join([filename[:-4], '_readout', filename[-4:]])

    # Save the network state dictionary
    torch.save(
        model.state_dict(), os.path.join(directory, filename)
    )
    # If defined, save the readout state dictionary
    if readout_filename:
        torch.save(
            readout.state_dict(), os.path.join(directory, readout_filename)
        )

    # Write the hyperparameters to the setup file
    with open(os.path.join(directory, setup_filename), 'w+') as file:
        file.write(
            f'Model Setup for {METHOD+"virtual " if virtual else ""}'
            '{PATIENT_GROUPS} Trained Network:\n\n'
        )
        file.write(
            f'{NETWORK_TYPE} Network Architecture Parameters\n'
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

