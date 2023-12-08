"""Code for training RNN-based PINNs"""

import os
import time
from pandas.core.computation.ops import Op
import torch
import torch.nn as nn
import torch.optim as optim
import torchcde
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from torch.nn.functional import (mse_loss, l1_loss)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from neural_cde import DEControl, NeuralCDE
from get_data import NelsonData
from testing_ncde_mech import main as test

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

class Control(nn.Module):
    def __init__(self):
        pass


def main():
    print(f'Starting Training w/ {PATIENT_GROUPS[0]} #{INDIVIDUAL_NUMBER}')

    loader = load_data(PATIENT_GROUPS)

    # domain = torch.linspace(0, T_END, SEQ_LENGTH, requires_grad=True, dtype=torch.double).to(DEVICE)
    dense_domain = torch.linspace(0, T_END, SEQ_LENGTH, dtype=torch.float32, requires_grad=False)
    domain = torch.tensor([0,15,30,40,50,65,80,95,110,125,140], dtype=torch.float32)

    model = NeuralCDE(
        INPUT_CHANNELS, HDIM, OUTPUT_CHANNELS, t_interval=domain, prediction=True,
        device=DEVICE, dense_domain=torch.linspace(0, T_END, SEQ_LENGTH),
        interpolation='mechanistic'
    ).to(torch.float32)

    params = param_init_tsst(model)
    bounds = []
    for key, val in params.items():
        val = val.item()
        d = np.abs(val)*0.5
        bounds.append((val-d, val+d))

    for (data, _) in loader:
        y0 = torch.cat([torch.tensor([0]), data[0,0,1:], torch.tensor([0])])
        res = differential_evolution(de_loss, bounds, args=(data,y0), disp=True, maxiter=100)
        sol = func(res.x, y0)
        sol = sol.to(torch.float32)

    for idx, key in enumerate(params.keys()):
        model.register_parameter(key, nn.Parameter(torch.tensor(res.x[idx])))

    for p in model.parameters():
        if p.dtype != torch.float32:
            p = p.to(torch.float32)
    start_time = time.time()

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=DECAY
    )

    loss_over_time = []
    # Random time step to be removed (for testing irregularly sampled series)
    random_idx = torch.randperm(11)[:2]

    for j, (data, labels) in enumerate(loader):
        data_tmp = torch.zeros((1,0,3))
        # print(f"{random_idx=}")
        # for t in range(data.shape[1]):
        #     if t in random_idx:
        #         continue
        #     data_tmp = torch.cat([data_tmp, data[...,t,:].view(1,1,3)], dim=1)
        # data = data_tmp
        labels = data.to(DEVICE)
        labels = labels.to(torch.float32)

        # coeffs = torchcde.\
        #     hermite_cubic_coefficients_with_backward_differences(
        #         sol.view(1, SEQ_LENGTH, 5), t=dense_domain
        #     )
        control = DEControl(sol, params, t=dense_domain)


        for itr in range(1, ITERS+1):
            global ITR, GRAPH_FLAG
            ITR = itr
            if ITR % 1000 == 0:
                GRAPH_FLAG = True
            else:
                GRAPH_FLAG = False
            # Zero the gradient from the previous data
            optimizer.zero_grad()

            pred_y = model(control)

            # Compute the loss based on the results
            (mech_loss, data_loss, ic_loss) = loss(
                pred_y, labels=labels,
                domain=domain,
                params=params,
                mech_solution=sol,
            )
            # output = (mech_loss + data_loss)/2
            output = data_loss

            loss_over_time.append((j, mech_loss.item()))

            # Backpropagate the loss
            output.backward(retain_graph=True)

            # Use the gradients calculated through backpropagation to adjust the
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 10, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f} (mech_loss = {mech_loss.item():.6f}, data_loss = {data_loss.item():.6f}, ic_loss = {ic_loss.item() if ic_loss else 0:.6f})")
            if itr % SAVE_FREQ == 0:
                runtime = time.time() - start_time
                save_network(model, optimizer, itr, runtime, loss_over_time)
            if OPT_RESET and itr % OPT_RESET == 0:
                optimizer = optim.AdamW(
                    model.parameters(), lr=LR, weight_decay=DECAY
                )

        with torch.no_grad():
            test(individual_number=INDIVIDUAL_NUMBER, res=res)


def load_data(patient_groups):
    dataset = NelsonData(
        patient_groups=patient_groups,
        normalize_standardize=NORMALIZE_STANDARDIZE,
        individual_number=INDIVIDUAL_NUMBER,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True,
    )
    return loader


def param_init_tsst(model: nn.Module):
    """Initialize the parameters for the mechanistic loss, and set them to
    require gradient"""
    R0CRH = torch.nn.Parameter(torch.tensor(-0.52239, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RCRH_CRH = torch.nn.Parameter(torch.tensor(0.97555, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RGR_CRH = torch.nn.Parameter(torch.tensor(-20.0241, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RSS_CRH = torch.nn.Parameter(torch.tensor(9.8594, device=DEVICE, dtype=torch.float32), requires_grad=False)
    sigma = torch.nn.Parameter(torch.tensor(4.974, device=DEVICE, dtype=torch.float32), requires_grad=False)
    tsCRH = torch.nn.Parameter(torch.tensor(0.10008, device=DEVICE, dtype=torch.float32), requires_grad=False)
    R0ACTH = torch.nn.Parameter(torch.tensor(-0.29065, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RCRH_ACTH = torch.nn.Parameter(torch.tensor(16.006, device=DEVICE, dtype=torch.float32), requires_grad=False)
    RGR_ACTH = torch.nn.Parameter(torch.tensor(-20.004, device=DEVICE, dtype=torch.float32), requires_grad=False)
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


def func(params, y0):
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

    t_eval = torch.linspace(0, T_END, SEQ_LENGTH)
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


def loss(pred_y: torch.Tensor, labels: torch.Tensor,
         domain: torch.Tensor | None=None,
         params: dict={}, mech_solution: torch.Tensor | None=None):
    """To compute the loss with potential mechanistic components"""

    def find_nearest(array, value):
        idx = (torch.abs(array-value)).argmin()
        return idx

    pred_y = pred_y.squeeze()
    # mech_loss = mechanistic_loss_tsst(pred_y, domain, params)
    crh_loss = l1_loss(pred_y[:,1], mech_solution[:,1])
    acth_loss = l1_loss(pred_y[:,1], mech_solution[:,2])
    cort_loss = l1_loss(pred_y[:,2], mech_solution[:,3])
    gr_loss = l1_loss(pred_y[:,3], mech_solution[:,4])
    mech_loss = (10*crh_loss + acth_loss + 2*cort_loss + 10*gr_loss)/23
    # mech_loss = (acth_loss + 2*cort_loss)/3

    labels = labels.squeeze()

    pred_y_datapoints = pred_y[[find_nearest(domain, t) for t in labels[...,0].squeeze()],...]

    data_loss = l1_loss(pred_y_datapoints[...,[0,2,3]], labels)
    ic_loss = l1_loss(labels[...,0,[1,2]], pred_y[...,0,[2,3]])
    # ic_loss = l1_loss(torch.cat([torch.tensor([0]), labels[...,0,[1,2]].squeeze(), torch.tensor([0])]), pred_y[...,0,:])

    return (mech_loss, data_loss, ic_loss)


def mechanistic_loss_tsst(pred_y, domain, params):
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

    # In order to weight CRH and GR more heavily, as they are so small, let's
    #  multiply pred_y and pred_dy by 3 for those columns
    pdy = pred_dy.clone()
    mdy = mechanistic_dy.clone()
    pdy[...,:,[0,3]] = pred_dy[...,:,[0,3]]*10
    mdy[...,:,[0,3]] = mechanistic_dy[...,:,[0,3]]*10


    if GRAPH_FLAG:
        with torch.no_grad():
            fig, axes = plt.subplots(nrows=4, figsize=(10,8))
            for idx in range(pred_dy.shape[-1]):
                axes[idx].plot(domain, pred_dy[...,idx], label='Predicted dy')
                axes[idx].plot(domain, mechanistic_dy[...,idx], label='Mechanistic dy')
                axes[idx].legend(fancybox=True, shadow=True, loc='upper right')
            plt.savefig(f'Results/pred_dy_vs_mech_dy_{ITR}ITERS_{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}.png', dpi=300)
            plt.close(fig)

    return l1_loss(pdy, mdy)


def save_network(model: NeuralCDE, optimizer: optim.AdamW, itr: int, runtime: float, loss_over_time: list[float]):
    """Save the network state_dict and the training hyperparameters in the
    relevant directory"""
    directory = (
        f'Network States/'
        f'{"Classification" if CLASSIFY else "Prediction"}/'
        f'{PATIENT_GROUPS[0]}/'
    )
    # Make sure the directory exists before we try to write anything
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = (
        f'NN_state_{HDIM}nodes_{NETWORK_TYPE}_'
        f'{PATIENT_GROUPS[0]}{INDIVIDUAL_NUMBER}_'
        f'batchsize{BATCH_SIZE}_'
        f'{itr}ITER_{NORMALIZE_STANDARDIZE}_'
        f'smoothing{LABEL_SMOOTHING}_'
        f'dropout{DROPOUT}_noParamGrads.txt'
    )
    # Add _setup to the filename before the .txt extension
    setup_filename = "".join([filename[:-4], "_setup", filename[-4:]])

    # Add _readout to the filename before the .txt extension if readout is an
    #  instance of nn.Linear
    # Save the network state dictionary
    torch.save(
        model.state_dict(), os.path.join(directory, filename)
    )

    # Write the hyperparameters to the setup file
    with open(os.path.join(directory, setup_filename), 'w+') as file:
        file.write(
            f'Model Setup for '
            '{PATIENT_GROUPS} #{INDIVIDUAL_NUMBER} Trained Network:\n\n'
        )
        file.write(
            f'{NETWORK_TYPE} Network Architecture Parameters\n'
            f'Input channels={INPUT_CHANNELS}\n'
            f'Hidden channels={HDIM}\n'
            f'Output channels={OUTPUT_CHANNELS}\n\n'
            'Training hyperparameters\n'
            f'Optimizer={optimizer}\n'
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
            f'Training batch size={BATCH_SIZE}\n'
            'Training Results:\n'
            f'Runtime={runtime}\n'
            f'Loss over time={loss_over_time}'
        )


if __name__ == '__main__':
    for INDIVIDUAL_NUMBER in range(1, 16):
        main()
