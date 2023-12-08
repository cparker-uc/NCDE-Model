"""New file for testing RNN-based PINNs, so hopefully this will be easier to
read"""


"""Code for testing trained networks and saving summaries of classification
success rates into Excel spreadsheets"""

import os
import torch
import torch.nn as nn
import torchcde
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy.integrate import odeint
from neural_cde import DEControl, NeuralCDE
from get_data import NelsonData

# The constants listed here are to be defined in classification.py. The train()
#  function will iterate through the parameter names passed in the
#  parameter_dict argument and set values to these global variables for use in
#  all of the functions in this namespace
# I just set them here so that linters don't throw warnings
# Network architecture parameters
NETWORK_TYPE: str = 'NCDE'
INPUT_CHANNELS: int = 5
HDIM: int = 64
OUTPUT_CHANNELS: int = 5
SEQ_LENGTH: int = 50
CLASSIFY: bool = False # For use in choosing between classification/prediction
MECHANISTIC: bool = True # Should the mechanistic components be included?

# Training hyperparameters
ITERS: int = 1000
SAVE_FREQ: int = 500
LR: float = 1e-3
DECAY: float = 0.
OPT_RESET: int|None = 250
ATOL: float = 1e-6
RTOL: float = 1e-8

# Training data selection parameters
PATIENT_GROUPS: list = ['Control']
INDIVIDUAL_NUMBER: int = 1
METHOD: str = 'None'
NORMALIZE_STANDARDIZE: str = 'None'
NOISE_MAGNITUDE: float = 0.
IRREGULAR_T_SAMPLES: bool = False
NUM_PER_PATIENT: int = 0
POP_NUMBER: int = 0
BATCH_SIZE: int = 1
LABEL_SMOOTHING: float = 0
DROPOUT: float = 0.
CORT_ONLY: bool = False
T_END: int = 140

# Define the device with which to train networks
DEVICE = torch.device('cpu')

ITR = 0

# Flag to track whether we should be graphing the mechanistic and predicted dy
#  on the current iteration
GRAPH_FLAG:bool = False


def main(individual_number: int=0, random_idx: int | None=None, res=None):
    global INDIVIDUAL_NUMBER
    INDIVIDUAL_NUMBER = individual_number
    loader = load_data(
        patient_groups=PATIENT_GROUPS, test=True
    )
    dense_domain = torch.linspace(0, T_END, SEQ_LENGTH, dtype=torch.float32)
    domain = torch.tensor([0,15,30,40,50,65,80,95,110,125,140], dtype=torch.float32)
    model = NeuralCDE(
        INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS,
        domain, device=DEVICE, prediction=True, dense_domain=dense_domain,
        interpolation='mechanistic'
    )

    param_init_tsst(model)

    directory = (
        f'Network States/'
        f'{"Classification" if CLASSIFY else "Prediction"}/'
        f'{PATIENT_GROUPS[0]}/'
    )

    # Loop over the state dictionaries based on the number of iterations, from
    #  100 to MAX_ITR*100
    n_saves = int(np.floor(ITERS/SAVE_FREQ))
    for itr in range(1,n_saves+1):
        # Set the filename for the network state_dict
        state_file = (
            f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
            f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
            f'batchsize{BATCH_SIZE}_'
            f'{itr*SAVE_FREQ}ITER_{NORMALIZE_STANDARDIZE}_'
            f'smoothing{LABEL_SMOOTHING}_'
            f'dropout{DROPOUT}_noParamGrads.txt'
        )

        state_filepath = os.path.join(directory, state_file)
        # Check that the file exists
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f'Failed to load file: {state_filepath}')
        # Load the state dictionary from the file and set the model to that
        #  state
        state_dict = torch.load(state_filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)
        params = model._parameters

        # Loop through the test patients
        for (data, _) in loader:
            y0 = torch.cat([torch.tensor([0]), data[0,0,1:], torch.tensor([0])])
            sol = func(res.x, y0)
            sol = sol.to(torch.float32)

            # coeffs = torchcde.\
            #     hermite_cubic_coefficients_with_backward_differences(
            #         sol, t=domain
            #     )
            control = DEControl(sol, params, dense_domain)

            # Repeat for the dense predictions
            pred_y = model(control)

            state = model.state_dict()
            y0 = torch.cat((torch.tensor([0]), data[...,0,1:].squeeze(), torch.tensor([0])))
            true_mechanistic = func(state, y0)

            patient_id = f'{PATIENT_GROUPS[0]} {INDIVIDUAL_NUMBER}'

            # Graph the results
            graph_results(pred_y, dense_domain, data, patient_id,
                          itr, true_mechanistic, random_idx)


def load_data(patient_groups: list=[], test: bool=True):
    dataset = NelsonData(
        patient_groups=patient_groups,
        normalize_standardize=NORMALIZE_STANDARDIZE,
        individual_number=INDIVIDUAL_NUMBER,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=2000, shuffle=False
    )
    return loader


def param_init_tsst(model: nn.Module):
    """Initialize the parameters for the mechanistic loss, and set them to
    require gradient"""
    R0CRH = torch.nn.Parameter(torch.tensor(-0.52239, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RCRH_CRH = torch.nn.Parameter(torch.tensor(0.97555, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RGR_CRH = torch.nn.Parameter(torch.tensor(-2.0241, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RSS_CRH = torch.nn.Parameter(torch.tensor(9.8594, device=DEVICE, dtype=torch.float32), requires_grad=False)
    sigma = torch.nn.Parameter(torch.tensor(4.974, device=DEVICE, dtype=torch.float32), requires_grad=False)
    tsCRH = torch.nn.Parameter(torch.tensor(0.10008, device=DEVICE, dtype=torch.float32), requires_grad=False)
    R0ACTH = torch.nn.Parameter(torch.tensor(-0.29065, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RCRH_ACTH = torch.nn.Parameter(torch.tensor(6.006, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RGR_ACTH = torch.nn.Parameter(torch.tensor(-10.004, device=DEVICE, dtype=torch.float32), requires_grad=False)
    tsACTH = torch.nn.Parameter(torch.tensor(0.046655, device=DEVICE, dtype=torch.float32), requires_grad=False)
    R0CORT = torch.nn.Parameter(torch.tensor(-0.95265, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RACTH_CORT = torch.nn.Parameter(torch.tensor(0.022487, device=DEVICE, dtype=torch.float32), requires_grad=False)
    tsCORT = torch.nn.Parameter(torch.tensor(0.048451, device=DEVICE, dtype=torch.float32), requires_grad=False)
    R0GR = torch.nn.Parameter(torch.tensor(-0.49428, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RCORT_GR = torch.nn.Parameter(torch.tensor(0.02745, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RGR_GR = torch.nn.Parameter(torch.tensor(0.10572, device=DEVICE, dtype=torch.float32), requires_grad=False)
    kdStress = torch.nn.Parameter(torch.tensor(0.19604, device=DEVICE, dtype=torch.float32), requires_grad=False)
    MaxCRH = torch.nn.Parameter(torch.tensor(1.0011, device=DEVICE, dtype=torch.float32), requires_grad=False)
    MaxACTH = torch.nn.Parameter(torch.tensor(140.2386, device=DEVICE, dtype=torch.float32), requires_grad=False)
    MaxCORT = torch.nn.Parameter(torch.tensor(30.3072, device=DEVICE, dtype=torch.float32), requires_grad=False)
    BasalACTH = torch.nn.Parameter(torch.tensor(0.84733, device=DEVICE, dtype=torch.float32), requires_grad=False)
    BasalCORT = torch.nn.Parameter(torch.tensor(0.29757, device=DEVICE, dtype=torch.float32), requires_grad=False)
    ksGR = torch.nn.Parameter(torch.tensor(0.40732, device=DEVICE, dtype=torch.float32), requires_grad=False)
    kdGR = torch.nn.Parameter(torch.tensor(0.39307, device=DEVICE, dtype=torch.float32), requires_grad=False)

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


def graph_results(pred_y: torch.Tensor | tuple, pred_t: torch.Tensor,
                  data: torch.Tensor, patient_id: str,
                  save_num: int, true_mechanistic: torch.Tensor,
                  random_idx: int | None):

    # Convert the save number to the number of iterations
    itr = save_num*SAVE_FREQ

    # Set the directory name based on which type of dataset was used for the
    #  training
    directory = (
        f'Results/'
        f'{"Classification" if CLASSIFY else "Prediction"}/'
        f'{PATIENT_GROUPS[0]}/'
    )

    # Make sure the directory exists before we try to write anything
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set the filename for the network state_dict
    filename = (
        f'{NETWORK_TYPE}_{HDIM}nodes_'
        f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
        f'batchsize{BATCH_SIZE}_'
        f'{ITERS}maxITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}_'
    )

    pred_y = pred_y.squeeze()
    nrows = 4
    fig, axes = plt.subplots(nrows=nrows, figsize=(10,10))
    for ax in range(nrows):

        if ax in [1,2]:
            for t in range(11):
                color = 'blue'
                if random_idx and t in random_idx:
                    color = 'red'
                axes[ax].scatter(data[:,t,0], data[:,t,ax], color=color)
        axes[ax].plot(
            pred_t, pred_y[:,ax], color='orange', label=f'Predicted y ({itr} iterations)'
        )
        axes[ax].plot(
            true_mechanistic[...,0], true_mechanistic[...,ax+1], color='green', label=f'True Mechanistic Solution'
        )
        axes[ax].set(title=patient_id, xlabel='Time (normalized)', ylabel='Concentration')
        axes[ax].legend(fancybox=True, shadow=True, loc='upper right')

    plt.savefig(os.path.join(directory, filename+patient_id+f'_{itr}iterations.png'), dpi=300)
    plt.close(fig)


# def func(params, y0):
#     """This function solves the system to find the true mechanistic component
#     for graphing."""
#     def stress(t):
#         if t < 30:
#             return 0
#         return torch.exp(-params['kdStress']*(t-30))

#     def ode_rhs(y, t):
#         y = torch.from_numpy(y).view(-1)
#         dy = torch.zeros(4)

#         # Apologies for the mess here, but typing out ODEs in Python is a bit of a
#         #  chore

#         wCRH = params['R0CRH'] + params['RCRH_CRH']*y[...,0] \
#             + params['RSS_CRH']*stress(t) + params['RGR_CRH']*y[...,3]
#         FCRH = (params['MaxCRH']*params['tsCRH'])/(1 + torch.exp(-params['sigma']*wCRH))
#         dy[0] = FCRH - params['tsCRH']*y[...,0]

#         wACTH = params['R0ACTH'] + params['RCRH_ACTH']*y[...,0] \
#             + params['RGR_ACTH']*y[...,3]
#         FACTH = (params['MaxACTH']*params['tsACTH'])/(1 + torch.exp(-params['sigma']*wACTH)) + params['BasalACTH']
#         dy[1] = FACTH - params['tsACTH']*y[...,1]

#         wCORT = params['R0CORT'] + params['RACTH_CORT']*y[...,1]
#         FCORT = (params['MaxCORT']*params['tsCORT'])/(1 + torch.exp(-params['sigma']*wCORT)) + params['BasalCORT']
#         dy[2] = FCORT - params['tsCORT']*y[...,2]

#         wGR = params['R0GR'] + params['RCORT_GR']*y[...,2] + params['RGR_GR']*y[...,3]
#         FGR = params['ksGR']/(1 + torch.exp(-params['sigma']*wGR))
#         dy[3] = FGR - params['kdGR']*y[...,3]
#         return dy

#     t_eval = torch.linspace(0,140,100)
#     gflow = odeint(ode_rhs, y0, t_eval)
#     gflow = torch.from_numpy(gflow)
#     gflow = torch.cat((t_eval.view(-1,1), gflow), dim=1)
#     return gflow


def func(params, y0):
    """ODE System computation for the differential_evolution runs"""
    if isinstance(params, dict):
        vals = []
        for _, val in params.items():
            if val.shape == ():
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

    t_eval = torch.linspace(0, T_END, SEQ_LENGTH)
    gflow = odeint(ode_rhs, y0, t_eval)
    gflow = np.concatenate((t_eval.view(-1,1), gflow), axis=1)
    gflow = torch.from_numpy(gflow)
    return gflow


if __name__ == '__main__':
    with torch.no_grad():
        for INDIVIDUAL_NUMBER in range(1, 16):
            main(INDIVIDUAL_NUMBER)
