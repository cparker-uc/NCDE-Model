# File Name: toy_dataset.py
# Author: Christopher Parker
# Created: Tue Aug 01, 2023 | 10:09P EDT
# Last Modified: Wed Sep 13, 2023 | 08:54P EDT

"""Creates a toy dataset for use in validation of the NCDE architecture.
Apologies for the haphazard nature of the code, it was done quickly and
inefficiently. I'll refactor at some point"""

import os
import torch
import numpy as np
from scipy.integrate import odeint
from get_augmented_data import NelsonVirtualPopulation, ToyDataset

def func(params, y0):
    [k_stress, k_i, V_S3, K_m1, K_P2, V_S4, K_m2, K_P3, V_S5, K_m3, K_d1, K_d2,
     K_d3, n1, n2, K_b, G_tot, V_S2, K1, K_d5] = params
    def ode_rhs(y, t):
        dy = np.zeros(4)
        dy[0] = k_stress*(k_i**n2/(k_i**n2 + y[3]**n2)) \
            - V_S3*(y[0]/(K_m1 + y[0])) - K_d1*y[0]
        dy[1] = K_P2*y[0]*(k_i**n2/(k_i**n2 + y[3]**n2)) \
            - V_S4*(y[1]/(K_m2 + y[1])) - K_d2*y[1]
        dy[2] = K_P3*y[1] - V_S5*(y[2]/(K_m3 + y[2])) - K_d3*y[2]
        dy[3] = K_b*y[2]*(G_tot - y[3]) + V_S2*(y[3]**n1/(K1**n1 + y[3]**n1)) \
            - K_d5*y[3]
        return dy
    t_eval = torch.linspace(0,24,20)
    gflow = odeint(ode_rhs, y0, t_eval)
    gflow = torch.from_numpy(gflow)
    return torch.cat((t_eval.view(20,1), gflow), 1)


def generate_dataset():
    """Use a system of ordinary diff eqs to generate time-series data"""
    ctrl_params = [10.1, 1.51, 3.25, 1.74, 8.3, 0.907, 0.112, 0.945, 0.00535,
                   0.0768, 0.00379, 0.00916, 0.356, 5.43, 5.1, 0.0202, 3.28,
                   0.0509, 0.645, 0.0854]
    mdd_params = [13.7, 1.6, 3.25, 1.74, 8.3, 0.907, 0.112, 0.945, 0.00535,
                  0.0768, 0.00379, 0.00916, 0.356, 5.43, 5.1, 0.0202, 3.28,
                  0.0509, 0.645, 0.0854]
    ctrl_gflow = func(ctrl_params, [1, 7.14, 2.38, 2])
    # ctrl_gflow = torch.from_numpy(ctrl_gflow)
    mdd_gflow = func(mdd_params, [1, 10.833, 1.90, 2])
    # mdd_gflow = torch.from_numpy(mdd_gflow)

    noise = 0.50
    t_end = 24
    # ctrl_vpop = augment_gflow(ctrl_gflow, 1000, 'Uniform', noise)
    # mdd_vpop = augment_gflow(mdd_gflow, 1000, 'Uniform', noise)
    # torch.save(ctrl_vpop, f'Virtual Populations/Toy_Control_Uniform{noise}_None_1000_{t_end}hr_test.txt')
    # torch.save(mdd_vpop, f'Virtual Populations/Toy_Atypical_Uniform{noise}_None_1000_{t_end}hr_test.txt')
    torch.save(ctrl_gflow, f'Virtual Populations/Toy_Control_NoNoise_None_1_{t_end}hr_test.txt')
    torch.save(mdd_gflow, f'Virtual Populations/Toy_Atypical_NoNoise_None_1_{t_end}hr_test.txt')


def uniform_noise(input_tensor, noise_magnitude):
    output_tensor = torch.zeros_like(input_tensor)
    for idx, pt in enumerate(input_tensor):
        scaled = pt*noise_magnitude
        noise_range = 2*scaled
        lower_bound = pt - scaled
        output_tensor[idx] = torch.rand((1,))*noise_range + lower_bound
    return output_tensor


def augment_gflow(input_data, number, method, noise_magnitude=0.05):
    vpop = torch.zeros((number, 20, 5), dtype=float)
    match method:
        case 'Uniform':
            for vpt in vpop:
                vpt[...,0] = input_data[...,0]
                vpt[...,1] = uniform_noise(input_data[...,1], noise_magnitude)
                vpt[...,2] = uniform_noise(input_data[...,2], noise_magnitude)
                vpt[...,3] = uniform_noise(input_data[...,3], noise_magnitude)
                vpt[...,4] = uniform_noise(input_data[...,4], noise_magnitude)
        case 'Normal':
            for vpt in vpop:
                vpt[...,0] = input_data[...,0]
                vpt[...,1] = torch.normal(input_data[...,1], torch.tensor(noise_magnitude))
                vpt[...,2] = torch.normal(input_data[...,2], torch.tensor(noise_magnitude))
                vpt[...,3] = torch.normal(input_data[...,3], torch.tensor(noise_magnitude))
                vpt[...,4] = torch.normal(input_data[...,4], torch.tensor(noise_magnitude))
        case _:
            print("Unsupported augmentation strategy")
            return

    return vpop


def check_pop_stats():
    """Generate 5-number summary of vpops and non-augmented data"""
    ctrl_minimum = torch.ones((1,5))*100
    mdd_minimum = torch.ones((1,5))*100
    ctrl_maximum = torch.zeros((1,5))
    mdd_maximum = torch.zeros((1,5))

    # iqr = np.zeros(0)
    # full_range = np.zeros(0)
    # q1 = np.zeros(0)
    # q3 = np.zeros(0)
    noise = 0.50
    t_end = 2.35
    pop = ToyDataset(
        test=False,
        method='Uniform',
        noise_magnitude=noise,
        normalize_standardize='None',
        t_end=t_end,
    )
    test_pop = ToyDataset(
        test=False,
        method='Uniform',
        noise_magnitude=noise,
        normalize_standardize='None',
        t_end=t_end,
    )

    for patient in pop:
        (data, label) = patient
        if label == 1:
            cat = np.concatenate(
                (mdd_minimum.reshape(1,5), data.view(20,5)), 0
            )
            mdd_minimum = np.min(cat, axis=0)
            cat = np.concatenate(
                (mdd_maximum.reshape(1,5), data.view(20,5)), 0
            )
            mdd_maximum = np.max(cat, axis=0)
        else:
            cat = np.concatenate(
                (ctrl_minimum.reshape(1,5), data.view(20,5)), 0
            )
            ctrl_minimum = np.min(cat, axis=0)
            cat = np.concatenate(
                (ctrl_maximum.reshape(1,5), data.view(20,5)), 0
            )
            ctrl_maximum = np.max(cat, axis=0)
    for patient in test_pop:
        (data, label) = patient
        if label == 1:
            cat = np.concatenate(
                (mdd_minimum.reshape(1,5), data.view(20,5)), 0
            )
            mdd_minimum = np.min(cat, axis=0)
            cat = np.concatenate(
                (mdd_maximum.reshape(1,5), data.view(20,5)), 0
            )
            mdd_maximum = np.max(cat, axis=0)
        else:
            cat = np.concatenate(
                (ctrl_minimum.reshape(1,5), data.view(20,5)), 0
            )
            ctrl_minimum = np.min(cat, axis=0)
            cat = np.concatenate(
                (ctrl_maximum.reshape(1,5), data.view(20,5)), 0
            )
            ctrl_maximum = np.max(cat, axis=0)
    mdd_range = mdd_maximum - mdd_minimum
    ctrl_range = ctrl_maximum - ctrl_minimum

    mdd_sum = torch.zeros((1,5), dtype=torch.float64)
    ctrl_sum = torch.zeros((1,5), dtype=torch.float64)
    for patient in pop:
        (data, label) = patient
        if label == 1:
            tmp_sum = torch.sum(data, dim=0, dtype=torch.float64)
            tmp_cat = torch.concatenate((tmp_sum.view(1,5), mdd_sum.view(1,5)), dim=0)
            mdd_sum = torch.sum(tmp_cat, dim=0)
        else:
            tmp_sum = torch.sum(data, dim=0, dtype=torch.float64)
            tmp_cat = torch.concatenate((tmp_sum.view(1,5), ctrl_sum.view(1,5)), dim=0)
            ctrl_sum = torch.sum(tmp_cat, dim=0)
    for patient in test_pop:
        (data, label) = patient
        if label == 1:
            tmp_sum = torch.sum(data, dim=0, dtype=torch.float64)
            tmp_cat = torch.concatenate((tmp_sum.view(1,5), mdd_sum.view(1,5)), dim=0)
            mdd_sum = torch.sum(tmp_cat, dim=0)
        else:
            tmp_sum = torch.sum(data, dim=0, dtype=torch.float64)
            tmp_cat = torch.concatenate((tmp_sum.view(1,5), ctrl_sum.view(1,5)), dim=0)
            ctrl_sum = torch.sum(tmp_cat, dim=0)
    mdd_mean = mdd_sum/40000 # 2000 patients * 20 time points per
    ctrl_mean = ctrl_sum/40000

    mdd_pop = torch.zeros((0,20,5))
    ctrl_pop = torch.zeros((0,20,5))

    for patient in pop:
        (data, label) = patient
        data = data.view(1,20,5)
        if label == 1:
            mdd_pop = torch.cat((mdd_pop, data), dim=0)
        else:
            ctrl_pop = torch.cat((ctrl_pop, data), dim=0)
    for patient in test_pop:
        (data, label) = patient
        data = data.view(1,20,5)
        if label == 1:
            mdd_pop = torch.cat((mdd_pop, data), dim=0)
        else:
            ctrl_pop = torch.cat((ctrl_pop, data), dim=0)

    mdd_std = torch.zeros((1,5), dtype=torch.float64)
    ctrl_std = torch.zeros((1,5), dtype=torch.float64)
    mdd_std = torch.std(mdd_pop, dim=0)
    ctrl_std = torch.std(ctrl_pop, dim=0)
    mdd_std = torch.sum(mdd_std, dim=0)/20
    ctrl_std = torch.sum(ctrl_std, dim=0)/20

    # for patient in pop:
    #     (data, label) = patient
    #     if label == 1:
    #         tmp_std = torch.std(data, axis=0)
    #         mdd_std = torch.sum(torch.cat((tmp_std.view(1,5)/20, mdd_std.view(1,5)), axis=0), axis=0)
    #     else:
    #         tmp_std = torch.std(data, axis=0)
    #         ctrl_std = torch.sum(torch.cat((tmp_std.view(1,5)/20, ctrl_std.view(1,5)), axis=0), axis=0)

    with open(f'pop_stats_{noise}_{t_end}hr.txt', 'w+') as file:
        file.write(f"{mdd_minimum=}\n")
        file.write(f"{mdd_maximum=}\n")
        file.write(f"{mdd_range=}\n")
        file.write(f"{ctrl_minimum=}\n")
        file.write(f"{ctrl_maximum=}\n")
        file.write(f"{ctrl_range=}\n")
        file.write(f"{mdd_mean=}\n")
        file.write(f"{mdd_std=}\n")
        file.write(f"{ctrl_mean=}\n")
        file.write(f"{ctrl_std=}\n")


def normalize_data():
    noise = 0.1
    t_end = 24
    if noise == 0.05:
        mdd_min = [0, 0.30450401, 10.29135227, 1.80526507, 1.89663088]
        mdd_range = [10, 2.93194214, 9.67846584, 43.60776365, 1.29089105]
        ctrl_min = [0, 0.23755702, 5.03343105, 2.26103497, 1.89087498]
        ctrl_range = [10, 1.52874592, 5.71150208, 21.30146313, 1.06240237]
    elif noise == 0.1:
        mdd_min = [0, 0.28851172, 9.75164413, 1.71114516, 1.7931894]
        mdd_range = [10, 3.10140231, 11.16587257, 45.86234117, 1.54139781]
        ctrl_min = [0, 0.22536331, 4.768466, 2.14328814, 1.79135418]
        ctrl_range = [10, 1.62490994, 6.48696804, 22.55258703, 1.30256915]
    elif noise == 0.25:
        mdd_min = [0, 0.24047485, 8.12894821, 1.42532361, 1.49781454]
        mdd_range = [10, 3.85286999, 23.77321625, 54.06806946, 3.79435682]
        ctrl_min = [0, 0.18751946, 3.98012114, 1.7873224, 1.49384415]
        ctrl_range = [10, 1.91521868, 8.80539751, 26.26207411, 2.0220288]

    pop = ToyDataset(
        test=False,
        method='Uniform',
        noise_magnitude=0.1,
        normalize_standardize='None',
        t_end=t_end,
    )
    test_pop = ToyDataset(
        test=False,
        method='Uniform',
        noise_magnitude=0.1,
        normalize_standardize='None',
        t_end=t_end,
    )

    mdd_pop = torch.zeros((0,20,5))
    mdd_test_pop = torch.zeros((0,20,5))
    ctrl_pop = torch.zeros((0,20,5))
    ctrl_test_pop = torch.zeros((0,20,5))

    for patient in pop:
        (data, label) = patient
        data = data.view(1,20,5)
        if label == 1:
            mdd_pop = torch.cat((mdd_pop, data), dim=0)
        else:
            ctrl_pop = torch.cat((ctrl_pop, data), dim=0)
    for patient in test_pop:
        (data, label) = patient
        data = data.view(1,20,5)
        if label == 1:
            mdd_test_pop = torch.cat((mdd_test_pop, data), dim=0)
        else:
            ctrl_test_pop = torch.cat((ctrl_test_pop, data), dim=0)


def standardize_data():
    noise = 0.50
    t_end = 24
    if t_end == 2.35:
        if noise == 0.05:
            mdd_mean = torch.tensor([5, 1.3265, 16.2799, 32.1144, 2.6987])
            mdd_std = torch.tensor([0, 0.0383, 0.4686, 0.9269, 0.0779])
            ctrl_mean = torch.tensor([5, 0.7979, 8.2685, 17.2691, 2.5032])
            ctrl_std = torch.tensor([0, 0.0231, 0.2385, 0.4976, 0.0728])
        elif noise == 0.1:
            mdd_mean = torch.tensor([5, 1.3266, 16.2828, 32.1128, 2.6990])
            mdd_std = torch.tensor([0, 0.0762, 0.9384, 1.8486, 0.1557])
            ctrl_mean = torch.tensor([5, 0.7971, 8.2670, 17.2662, 2.5031])
            ctrl_std = torch.tensor([0, 0.0459, 0.4769, 0.9983, 0.1442])
        elif noise == 0.25:
            mdd_mean = torch.tensor([ 1.1750,  2.4963, 14.2715, 12.5210,  2.1155])
            mdd_std = torch.tensor([0.0000, 0.3594, 2.0662, 1.8254, 0.3056])
            ctrl_mean = torch.tensor([1.1750, 1.4960, 8.5365, 8.5272, 2.0536])
            ctrl_std = torch.tensor([0.0000, 0.2159, 1.2354, 1.2310, 0.2970])
        elif noise == 0.5:
            mdd_mean = torch.tensor([ 1.1750,  2.4985, 14.3014, 12.5161,  2.1164])
            mdd_std = torch.tensor([0.0000, 0.7216, 4.1354, 3.5957, 0.6146])
            ctrl_mean = torch.tensor([1.1750, 1.4961, 8.5441, 8.5455, 2.0581])
            ctrl_std = torch.tensor([0.0000, 0.4304, 2.4606, 2.4761, 0.5910])
    elif t_end == 10:
        if noise == 0.05:
            mdd_mean = torch.tensor([5, 1.3265, 16.2799, 32.1144, 2.6987])
            mdd_std = torch.tensor([0, 0.0383, 0.4686, 0.9269, 0.0779])
            ctrl_mean = torch.tensor([5, 0.7979, 8.2685, 17.2691, 2.5032])
            ctrl_std = torch.tensor([0, 0.0231, 0.2385, 0.4976, 0.0728])
        elif noise == 0.1:
            mdd_mean = torch.tensor([5, 1.3266, 16.2828, 32.1128, 2.6990])
            mdd_std = torch.tensor([0, 0.0762, 0.9384, 1.8486, 0.1557])
            ctrl_mean = torch.tensor([5, 0.7971, 8.2670, 17.2662, 2.5031])
            ctrl_std = torch.tensor([0, 0.0459, 0.4769, 0.9983, 0.1442])
        elif noise == 0.25:
            mdd_mean = torch.tensor([5, 1.3256, 16.2918, 32.0422, 2.7007])
            mdd_std = torch.tensor([0, 0.1911, 2.3510, 4.6491, 0.3922])
            ctrl_mean = torch.tensor([5, 0.7985, 8.2694, 17.2730, 2.5051])
            ctrl_std = torch.tensor([0, 0.1152, 1.1870, 2.4883, 0.3609])
    elif t_end == 24:
        if noise == 0.05:
            mdd_mean = torch.tensor([12.0000,  0.7844, 11.2638, 28.1726,  2.8130])
            mdd_std = torch.tensor([0.0000, 0.0227, 0.3239, 0.8158, 0.0815])
            ctrl_mean = torch.tensor([12.0000,  0.7645,  4.6261, 11.2904,  2.4476])
            ctrl_std = torch.tensor([0.0000, 0.0222, 0.1327, 0.3253, 0.0706])
        elif noise == 0.1:
            mdd_mean = torch.tensor([12.0000,  0.7843, 11.2667, 28.1726,  2.8138])
            mdd_std = torch.tensor([0.0000, 0.0452, 0.6500, 1.6266, 0.1623])
            ctrl_mean = torch.tensor([12.0000,  0.7650,  4.6222, 11.2874,  2.4455])
            ctrl_std = torch.tensor([0.0000, 0.0442, 0.2679, 0.6519, 0.1413])
        elif noise == 0.25:
            mdd_mean = torch.tensor([12.0000,  0.7846, 11.2482, 28.1917,  2.8080])
            mdd_std = torch.tensor([0.0000, 0.1135, 1.6317, 4.0710, 0.4064])
            ctrl_mean = torch.tensor([12.0000,  0.7661,  4.6211, 11.2795,  2.4443])
            ctrl_std = torch.tensor([0.0000, 0.1104, 0.6628, 1.6223, 0.3544])
        elif noise == 0.5:
            mdd_mean = torch.tensor([12.0000,  0.7829, 11.2924, 28.1665,  2.8123])
            mdd_std = torch.tensor([0.0000, 0.2263, 3.2436, 8.1067, 0.8138])
            ctrl_mean = torch.tensor([12.0000,  0.7671,  4.6319, 11.2290,  2.4448])
            ctrl_std = torch.tensor([0.0000, 0.2214, 1.3273, 3.2338, 0.7093])

    pop = ToyDataset(
        test=False,
        method='Uniform',
        noise_magnitude=noise,
        normalize_standardize='None',
        t_end=t_end,
    )
    test_pop = ToyDataset(
        test=False,
        method='Uniform',
        noise_magnitude=noise,
        normalize_standardize='None',
        t_end=t_end,
    )

    mdd_pop = torch.zeros((0,20,5))
    mdd_test_pop = torch.zeros((0,20,5))
    ctrl_pop = torch.zeros((0,20,5))
    ctrl_test_pop = torch.zeros((0,20,5))

    for patient in pop:
        (data, label) = patient
        data = data.view(1,20,5)
        if label == 1:
            mdd_pop = torch.cat((mdd_pop, data), dim=0)
        else:
            ctrl_pop = torch.cat((ctrl_pop, data), dim=0)
    for patient in test_pop:
        (data, label) = patient
        data = data.view(1,20,5)
        if label == 1:
            mdd_test_pop = torch.cat((mdd_test_pop, data), dim=0)
        else:
            ctrl_test_pop = torch.cat((ctrl_test_pop, data), dim=0)

    mdd_pop_standard = (mdd_pop[...,1:] - mdd_mean[...,1:])/mdd_std[...,1:]
    mdd_pop_standard = torch.cat(((mdd_pop[...,0]/t_end).view(1000,20,1), mdd_pop_standard), dim=2)
    mdd_test_pop_standard = (mdd_test_pop[...,1:] - mdd_mean[...,1:])/mdd_std[...,1:]
    mdd_test_pop_standard = torch.cat(((mdd_test_pop[...,0]/t_end).view(1000,20,1), mdd_test_pop_standard), dim=2)

    ctrl_pop_standard = (ctrl_pop[...,1:] - ctrl_mean[...,1:])/ctrl_std[...,1:]
    ctrl_pop_standard = torch.cat(((ctrl_pop[...,0]/t_end).view(1000,20,1), ctrl_pop_standard), dim=2)
    ctrl_test_pop_standard = (ctrl_test_pop[...,1:] - ctrl_mean[...,1:])/ctrl_std[...,1:]
    ctrl_test_pop_standard = torch.cat(((ctrl_test_pop[...,0]/t_end).view(1000,20,1), ctrl_test_pop_standard), dim=2)

    torch.save(mdd_pop_standard, f"Virtual Populations/Toy_Atypical_Uniform{noise}_Standardize_1000_{t_end}hr.txt")
    torch.save(mdd_test_pop_standard, f"Virtual Populations/Toy_Atypical_Uniform{noise}_Standardize_1000_{t_end}hr_test.txt")
    torch.save(ctrl_pop_standard, f"Virtual Populations/Toy_Control_Uniform{noise}_Standardize_1000_{t_end}hr.txt")
    torch.save(ctrl_test_pop_standard, f"Virtual Populations/Toy_Control_Uniform{noise}_Standardize_1000_{t_end}hr_test.txt")


def main():
    generate_dataset()
    # check_pop_stats()
    # standardize_data()

if __name__ == '__main__':
    main()

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

