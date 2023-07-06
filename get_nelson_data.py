# File Name: get_nelson_data.py
# Author: Christopher Parker
# Created: Thu Apr 27, 2023 | 05:10P EDT
# Last Modified: Thu Jul 06, 2023 | 03:22P EDT

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class NelsonData(Dataset):
    def __init__(self, data_dir='Nelson TSST Individual Patient Data',
                 patient_groups=['Melancholic', 'Control'],
                 normalize_standardize=None):
        super().__init__()
        self.data_dir = data_dir
        self.patient_groups = patient_groups
        self.normalize_standardize = normalize_standardize

        length = self.__len__()

        # x contains 11 time points with 3 input channels (time, acth, cort)
        #  for each time, with 58 total patients
        self.X = torch.zeros((0, 11, 3))
        # self.X_standard = torch.zeros((length, 11, 3))
        # self.X_normal = torch.zeros((length, 11, 3))

        # self.y indicates whether the patient data is MDD
        self.y = torch.zeros((0,))

        for group in self.patient_groups:
            if group == 'Control':
                X_tmp = torch.zeros((15, 11, 3))
                y_tmp = torch.zeros((15,))
                for idx in range(15):
                    ACTHdata_path = os.path.join(
                        self.data_dir, f'{group}patient{idx+1}_ACTH.txt'
                    )
                    CORTdata_path = os.path.join(
                        self.data_dir, f'{group}patient{idx+1}_CORT.txt'
                    )

                    ACTHdata = np.genfromtxt(ACTHdata_path)

                    CORTdata = np.genfromtxt(CORTdata_path)

                    X_tmp[idx,...] = torch.from_numpy(
                        np.concatenate((ACTHdata, CORTdata), 1)[:,[0,1,3]]
                    )
                # We normalize the time steps, so that they are between 0
                #  and 1. Not sure this will have any impact, but it will
                #  make things somewhat cleaner
                X_tmp[...,0] = normalize_time_series(X_tmp[...,0])
                self.X = torch.cat((self.X, X_tmp), 0)
                self.y = torch.cat((self.y, y_tmp), 0)
            elif group == 'Melancholic':
                X_tmp = torch.zeros((15, 11, 3))
                y_tmp = torch.ones((15,))
                for idx in range(15):
                    ACTHdata_path = os.path.join(
                        self.data_dir, f'{group}patient{idx+1}_ACTH.txt'
                    )
                    CORTdata_path = os.path.join(
                        self.data_dir, f'{group}patient{idx+1}_CORT.txt'
                    )

                    ACTHdata = np.genfromtxt(ACTHdata_path)

                    CORTdata = np.genfromtxt(CORTdata_path)

                    X_tmp[idx,...] = torch.from_numpy(
                        np.concatenate((ACTHdata, CORTdata), 1)[:,[0,1,3]]
                    )
                # We normalize the time steps, so that they are between 0
                #  and 1. Not sure this will have any impact, but it will
                #  make things somewhat cleaner
                X_tmp[...,0] = normalize_time_series(X_tmp[...,0])
                self.X = torch.cat((self.X, X_tmp), 0)
                self.y = torch.cat((self.y, y_tmp), 0)
            else:
                X_tmp = torch.zeros((14, 11, 3))
                y_tmp = torch.ones((14,))
                for idx in range(14):
                    ACTHdata_path = os.path.join(
                        self.data_dir, f'{group}patient{idx+1}_ACTH.txt'
                    )
                    CORTdata_path = os.path.join(
                        self.data_dir, f'{group}patient{idx+1}_CORT.txt'
                    )

                    ACTHdata = np.genfromtxt(ACTHdata_path)

                    CORTdata = np.genfromtxt(CORTdata_path)

                    X_tmp[idx,...] = torch.from_numpy(
                        np.concatenate((ACTHdata, CORTdata), 1)[:,[0,1,3]]
                    )
                # We normalize the time steps, so that they are between 0
                #  and 1. Not sure this will have any impact, but it will
                #  make things somewhat cleaner
                X_tmp[...,0] = normalize_time_series(X_tmp[...,0])
                self.X = torch.cat((self.X, X_tmp), 0)
                self.y = torch.cat((self.y, y_tmp), 0)

        if self.normalize_standardize == 'Normalize':
            self.X = torch.cat(
                (
                    self.X[...,0].reshape(length,11,1),
                    normalize_nelson_data(self.X[...,1], 'ACTH').reshape(length,11,1),
                    normalize_nelson_data(self.X[...,2], 'CORT').reshape(length,11,1)
                ), 2
            )
        elif self.normalize_standardize == 'Standardize':
            self.X = torch.cat(
                (
                    self.X[...,0].reshape(length,11,1),
                    standardize_nelson_data(self.X[...,1], 'ACTH').reshape(length,11,1),
                    standardize_nelson_data(self.X[...,2], 'CORT').reshape(length,11,1)
                ), 2
            )
        elif self.normalize_standardize == 'NormalizeAll':
            self.X = torch.cat(
                (
                    self.X[...,0].reshape(length,11,1),
                    normalize_data(self.X[...,1], 'ACTH').reshape(length,11,1),
                    normalize_data(self.X[...,2], 'CORT').reshape(length,11,1)
                ), 2
            )
        elif self.normalize_standardize == 'StandardizeAll':
            self.X = torch.cat(
                (
                    self.X[...,0].reshape(length,11,1),
                    standardize_data(self.X[...,1], 'ACTH').reshape(length,11,1),
                    standardize_data(self.X[...,2], 'CORT').reshape(length,11,1)
                ), 2
            )


    def __len__(self):
        length = 0
        for group in self.patient_groups:
            if group in ['Control', 'Melancholic']:
                length += 15
            else:
                length += 14

        return length

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx]


class AblesonData(Dataset):
    def __init__(self, data_dir='Ableson TSST Individual Patient Data (Without First 30 Min)',
                 patient_groups=['MDD', 'Control'],
                 normalize_standardize=None):
        super().__init__()
        self.data_dir = data_dir
        self.patient_groups = patient_groups
        self.normalize_standardize = normalize_standardize

        length = self.__len__()

        # x contains 11 time points with 3 input channels (time, acth, cort)
        #  for each time, with 58 total patients
        self.X = torch.zeros((0, 11, 3))
        # self.X_standard = torch.zeros((length, 11, 3))
        # self.X_normal = torch.zeros((length, 11, 3))

        # self.y indicates whether the patient data is MDD
        self.y = torch.zeros((0,))

        for group in self.patient_groups:
            if group == 'Control':
                X_tmp = torch.zeros((37, 11, 3))
                y_tmp = torch.zeros((37,))
                for idx in range(37):
                    data_path = os.path.join(
                        self.data_dir, f'{group}Patient{idx+1}.txt'
                    )
                    data = np.genfromtxt(data_path)
                    X_tmp[idx,...] = torch.from_numpy(data)
                # We normalize the time steps, so that they are between 0
                #  and 1. Not sure this will have any impact, but it will
                #  make things somewhat cleaner
                X_tmp[...,0] = normalize_time_series(X_tmp[...,0])
                self.X = torch.cat((self.X, X_tmp), 0)
                self.y = torch.cat((self.y, y_tmp), 0)
            else:
                X_tmp = torch.zeros((13, 11, 3))
                y_tmp = torch.ones((13,))
                for idx in range(13):
                    data_path = os.path.join(
                        self.data_dir, f'{group}Patient{idx+1}.txt'
                    )
                    data = np.genfromtxt(data_path)
                    X_tmp[idx,...] = torch.from_numpy(data)
                # We normalize the time steps, so that they are between 0
                #  and 1. Not sure this will have any impact, but it will
                #  make things somewhat cleaner
                X_tmp[...,0] = normalize_time_series(X_tmp[...,0])
                self.X = torch.cat((self.X, X_tmp), 0)
                self.y = torch.cat((self.y, y_tmp), 0)

        if self.normalize_standardize == 'NormalizeAll':
            self.X = torch.cat(
                (
                    self.X[...,0].reshape(length,11,1),
                    normalize_data(self.X[...,1], 'ACTH').reshape(length,11,1),
                    normalize_data(self.X[...,2], 'CORT').reshape(length,11,1)
                ), 2
            )
        elif self.normalize_standardize == 'StandardizeAll':
            self.X = torch.cat(
                (
                    self.X[...,0].reshape(length,11,1),
                    standardize_data(self.X[...,1], 'ACTH').reshape(length,11,1),
                    standardize_data(self.X[...,2], 'CORT').reshape(length,11,1)
                ), 2
            )


    def __len__(self):
        length = 0
        for group in self.patient_groups:
            if group == 'Control':
                length += 37
            else:
                length += 13

        return length

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx]


class VirtualPopulation(Dataset):
    def __init__(self, patient_groups, method, normalize_standardize,
                 num_per_patient, num_patients, pop_number=None,
                 control_combination=None, mdd_combination=None,
                 fixed_perms=False, test=False, label_smoothing=0.):
        self.patient_groups = patient_groups
        self.num_per_patient = num_per_patient
        self.num_patients = num_patients
        self.X = torch.zeros((0, 11, 3), dtype=float)
        self.y = torch.zeros((0,), dtype=float)
        self.test = test
        self.combinations = None
        self.fixed_perms = fixed_perms
        if control_combination and mdd_combination:
            self.combinations = control_combination + mdd_combination

        for group in self.patient_groups:
            if self.test:
                if pop_number:
                    _, data = load_vpop(
                        group, method, normalize_standardize,
                        num_per_patient, num_patients, pop_number
                    )
                elif control_combination and mdd_combination:
                    _, data = load_vpop_combinations(
                        group, method, normalize_standardize,
                        num_per_patient,
                        control_combination if group == 'Control' else mdd_combination,
                        fixed_perms=fixed_perms
                    )
                else:
                    print('Need a pop_number or test patient combination')
            else:
                if pop_number:
                    data, _ = load_vpop(
                        group, method, normalize_standardize,
                        num_per_patient, num_patients, pop_number
                    )
                elif control_combination and mdd_combination:
                    data, _ = load_vpop_combinations(
                        group, method, normalize_standardize,
                        num_per_patient,
                        control_combination if group == 'Control' else mdd_combination,
                        fixed_perms=fixed_perms
                    )
                else:
                    print('Need a pop_number or test patient combination')
            self.X = torch.cat((self.X, data), 0)
            if group == 'Control':
                if self.test:
                    self.y = torch.cat(
                        (self.y, torch.zeros(15 - num_patients)+label_smoothing), 0
                    )
                else:
                    self.y = torch.cat(
                        (self.y, torch.zeros(num_per_patient*num_patients)+label_smoothing), 0
                    )
            else:
                if self.test:
                    if self.fixed_perms:
                        self.y = torch.cat(
                            (self.y, torch.ones(len(mdd_combination))-label_smoothing), 0
                        )
                    else:
                        if group == 'Melancholic':
                            self.y = torch.cat(
                                (self.y, torch.ones(15 - num_patients)-label_smoothing), 0
                            )
                        else:
                            self.y = torch.cat(
                                (self.y, torch.ones(14 - num_patients)-label_smoothing), 0
                            )
                else:
                    self.y = torch.cat(
                        (self.y, torch.ones(num_per_patient*num_patients)-label_smoothing), 0
                    )

    def __len__(self):
        length = 0
        if self.fixed_perms and self.test:
            return len(self.combinations)
        for group in self.patient_groups:
            if self.test:
                if group in ['Control', 'Melancholic']:
                    length += 15 - self.num_patients
                else:
                    length += 14 - self.num_patients
            if self.combinations:
                len_mdd = len(self.combinations) - 5
                if group in ['Control', 'Melancholic']:
                    length += (10*self.num_per_patient)
                else:
                    length += (14 - len_mdd)*self.num_per_patient
            else:
                length += self.num_per_patient*self.num_patients

        return length

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx]


class FullVirtualPopulation(Dataset):
    def __init__(self, method, noise_magnitude,
                 normalize_standardize, num_per_patient,
                 control_combination, mdd_combination,
                 test=False, label_smoothing=0.):
        self.patient_groups = ['Control', 'MDD']
        self.num_per_patient = num_per_patient
        self.num_mdd_patients = 56 - len(mdd_combination)
        self.num_control_patients = 52 - len(control_combination)
        self.X = torch.zeros((0, 11, 3), dtype=float)
        self.y = torch.zeros((0,), dtype=float)
        self.test = test
        self.combinations = control_combination + mdd_combination

        for group in self.patient_groups:
            if self.test:
                _, data = load_full_vpop_combinations(
                    group, method, normalize_standardize,
                    num_per_patient,
                    control_combination if group == 'Control' else mdd_combination,
                    self.num_control_patients if group == 'Control' else self.num_mdd_patients,
                    noise_magnitude=noise_magnitude
                )
            else:
                data, _ = load_full_vpop_combinations(
                    group, method, normalize_standardize,
                    num_per_patient,
                    control_combination if group == 'Control' else mdd_combination,
                    self.num_control_patients if group == 'Control' else self.num_mdd_patients,
                    noise_magnitude=noise_magnitude
                )
            self.X = torch.cat((self.X, data), 0)
            if group == 'Control':
                if self.test:
                    self.y = torch.cat(
                        (self.y, torch.zeros(len(control_combination))+label_smoothing), 0
                    )
                else:
                    self.y = torch.cat(
                        (self.y, torch.zeros(num_per_patient*self.num_control_patients)+label_smoothing), 0
                    )
            else:
                if self.test:
                    self.y = torch.cat(
                        (self.y, torch.ones(len(mdd_combination))-label_smoothing), 0
                    )
                else:
                    self.y = torch.cat(
                        (self.y, torch.ones(num_per_patient*self.num_mdd_patients)-label_smoothing), 0
                    )

    def __len__(self):
        if self.test:
            return len(self.combinations)
        else:
            return (self.num_control_patients + self.num_mdd_patients)*self.num_per_patient

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx]


def load_vpop(patient_group, method, normalize_standardize, num_per_patient,
              num_patients, pop_number):
    vpop_and_train = torch.load(f'Virtual Populations/{patient_group}_{method}'
                                f'_{normalize_standardize}_{num_per_patient}_'
                                f'{num_patients}_{pop_number}.txt')
    vpop = vpop_and_train[:1000,...]
    test = vpop_and_train[1000:,...]

    return vpop, test


def load_vpop_combinations(patient_group, method, normalize_standardize, num_per_patient,
              combination, fixed_perms, noise_magnitude=None):
    if fixed_perms:
        vpop_and_train = torch.load(f'Virtual Populations/{patient_group}'
                                    f'_{method}{noise_magnitude if noise_magnitude else ""}'
                                    f'_{normalize_standardize}_{num_per_patient}_'
                                    f'testPatients{combination}_fixedperms.txt')
    else:
        vpop_and_train = torch.load(f'Virtual Populations/{patient_group}_{method}{noise_magnitude if noise_magnitude else ""}'
                                    f'_{normalize_standardize}_{num_per_patient}_'
                                    f'testPatients{combination}.txt')
    if patient_group in ['Atypical', 'Neither']:
        vpop = vpop_and_train[:(14 - len(combination))*num_per_patient,...]
        test = vpop_and_train[(14 - len(combination))*num_per_patient:,...]
    else:
        vpop = vpop_and_train[:1000,...]
        test = vpop_and_train[1000:,...]

    return vpop, test


def load_full_vpop_combinations(patient_group, method, normalize_standardize, num_per_patient,
              combination, num_patients, noise_magnitude=None):
    vpop_and_train = torch.load(f'Virtual Populations/{patient_group}'
                                f'_{method}{noise_magnitude if noise_magnitude else ""}'
                                f'_{normalize_standardize}_{num_per_patient}_'
                                f'testPatients{combination}_fixedperms.txt')
    vpop = vpop_and_train[:num_patients*num_per_patient,...]
    test = vpop_and_train[num_patients*num_per_patient:,...]
    return vpop, test


def normalize_time_series(series_tensor):
    min = torch.min(series_tensor)
    s_range = torch.max(series_tensor) - torch.min(series_tensor)
    for series in series_tensor:
        series -= min
        series /= s_range
    return series_tensor


def standardize_time_series(series_tensor):
    mean = torch.mean(series_tensor)
    std = torch.std(series_tensor)
    for series in series_tensor:
        series -= mean
        series /= std
    return series_tensor


def normalize_nelson_data(series_tensor, hor):
    ACTHmin = torch.tensor(5.6, dtype=float)
    CORTmin = torch.tensor(2., dtype=float)
    ACTHrange = torch.tensor(111., dtype=float)
    CORTrange = torch.tensor(44., dtype=float)
    if hor == 'ACTH':
        for series in series_tensor:
            series -= ACTHmin
            series /= ACTHrange
    elif hor == 'CORT':
        for series in series_tensor:
            series -= CORTmin
            series /= CORTrange
    return series_tensor


def standardize_nelson_data(series_tensor, hor):
    ACTHmean = torch.tensor(23.2961, dtype=float)
    CORTmean = torch.tensor(9.0687, dtype=float)
    ACTHstd = torch.tensor(11.1307, dtype=float)
    CORTstd = torch.tensor(5.4818, dtype=float)
    if hor == 'ACTH':
        for series in series_tensor:
            series -= ACTHmean
            series /= ACTHstd
    elif hor == 'CORT':
        for series in series_tensor:
            series -= CORTmean
            series /= CORTstd
    return series_tensor


def normalize_data(series_tensor, hor):
    ACTHmin = torch.tensor(5.6, dtype=float)
    CORTmin = torch.tensor(1.2, dtype=float)
    ACTHrange = torch.tensor(160.5, dtype=float)
    CORTrange = torch.tensor(44.8, dtype=float)
    if hor == 'ACTH':
        for series in series_tensor:
            series -= ACTHmin
            series /= ACTHrange
    elif hor == 'CORT':
        for series in series_tensor:
            series -= CORTmin
            series /= CORTrange
    return series_tensor


def standardize_data(series_tensor, hor):
    ACTHmean = torch.tensor(23.6205, dtype=float)
    CORTmean = torch.tensor(9.2878, dtype=float)
    ACTHstd = torch.tensor(4.8611, dtype=float)
    CORTstd = torch.tensor(15.2958, dtype=float)
    if hor == 'ACTH':
        for series in series_tensor:
            series -= ACTHmean
            series /= ACTHstd
    elif hor == 'CORT':
        for series in series_tensor:
            series -= CORTmean
            series /= CORTstd
    return series_tensor


def compute_data_summary_stats():
    nelson_dataset = NelsonData(patient_groups=['Control', 'Melancholic', 'Neither', 'Atypical'])
    ableson_dataset = AblesonData()
    dataset = ConcatDataset((nelson_dataset, ableson_dataset))
    loader = DataLoader(dataset, batch_size=1)
    sum = torch.tensor((0,0), dtype=float)
    mean = torch.tensor((0,0), dtype=float)
    minimum = torch.tensor((110,110), dtype=float)
    maximum = torch.tensor((0,0), dtype=float)
    range_ = torch.tensor((0,0), dtype=float)
    full_acth = torch.zeros((0,))
    full_cort = torch.zeros((0,))
    for batch in loader:
        for i in range(2):
            batch = batch[0]
            sum[i] += torch.sum(batch[...,i+1])
            min_tmp = torch.min(batch[...,i+1])
            minimum[i] = torch.min(minimum[i], min_tmp)
            max_tmp = torch.max(batch[...,i+1])
            maximum[i] = torch.max(maximum[i], max_tmp)
            if i == 1:
                full_acth = torch.cat((full_acth, batch[...,i+1].reshape(-1)), 0)
            else:
                full_cort = torch.cat((full_cort, batch[...,i+1].reshape(-1)), 0)
    print(f'{torch.std(full_acth)=}')
    print(f'{torch.std(full_cort)=}')
    mean = sum/(len(dataset)*11)
    range_ = maximum - minimum
    print(f'{mean=}, {minimum=}, {maximum=}, {range_=}')


if __name__ == '__main__':
    # For debugging purposes
    # dataset = NelsonData(patient_groups=['Control', 'Atypical'], normalize_standardize='Normalize')
    # dataset = VirtualPopulation(['Control', 'Atypical'], 'Uniform', 'Standardize', 100, 10, 1)
    # loader = DataLoader(dataset, batch_size=100, shuffle=True)
    # for batch in loader:
    #     print(batch)
    # compute_data_summary_stats()
    test = FullVirtualPopulation('Uniform', 0.05, 'StandardizeAll', 100, (47,35,5,44,28), (36,35,8,50,23))

    loader = DataLoader(test)
    for batch in loader:
        print(f'{batch=}')
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

