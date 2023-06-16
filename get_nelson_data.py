# File Name: get_nelson_data.py
# Author: Christopher Parker
# Created: Thu Apr 27, 2023 | 05:10P EDT
# Last Modified: Thu Jun 15, 2023 | 07:42P EDT

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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


class VirtualPopulation(Dataset):
    def __init__(self, patient_groups, method, normalize_standardize,
                 num_per_patient, num_patients, pop_number, label_smoothing=0):
        self.patient_groups = patient_groups
        self.num_per_patient = num_per_patient
        self.num_patients = num_patients
        self.X = torch.zeros((0, 11, 3), dtype=float)
        self.y = torch.zeros((0,), dtype=float)

        for group in self.patient_groups:
            vpop, _ = load_vpop(
                group, method, normalize_standardize,
                num_per_patient, num_patients, pop_number
            )
            self.X = torch.cat((self.X, vpop), 0)
            if group == 'Control':
                self.y = torch.cat(
                    (self.y, torch.zeros(num_per_patient*num_patients)+label_smoothing), 0
                )
            else:
                self.y = torch.cat(
                    (self.y, torch.ones(num_per_patient*num_patients)-label_smoothing), 0
                )

    def __len__(self):
        length = 0
        for group in self.patient_groups:
            length += self.num_per_patient*self.num_patients

        return length

    def __getitem__(self, idx):
        return self.X[idx,...], self.y[idx]




def load_vpop(patient_group, method, normalize_standardize, num_per_patient,
              num_patients, pop_number):
    vpop_and_train = torch.load(f'Virtual Populations/{patient_group}_{method}'
                                f'_{normalize_standardize}_{num_per_patient}_'
                                f'{num_patients}_{pop_number}.txt')
    vpop = vpop_and_train[:1000,...]
    train = vpop_and_train[1000:,...]

    return vpop, train


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


def compute_nelson_data_summary_stats():
    dataset = NelsonData(patient_groups=['Control', 'Atypical'])
    loader = DataLoader(dataset, batch_size=1)
    sum = torch.tensor(0, dtype=float)
    mean = torch.tensor(0, dtype=float)
    minimum = torch.tensor(110, dtype=float)
    maximum = torch.tensor(0, dtype=float)
    range_ = torch.tensor(0, dtype=float)
    full = torch.zeros((0,))
    for batch in loader:
        batch = batch[0]
        sum += torch.sum(batch[...,1])
        min_tmp = torch.min(batch[...,1])
        minimum = torch.min(minimum, min_tmp)
        max_tmp = torch.max(batch[...,1])
        maximum = torch.max(maximum, max_tmp)
        full = torch.cat((full, batch[...,1].reshape(-1)), 0)
    print(torch.std(full))
    mean = sum/(len(dataset)*11)
    range_ = maximum - minimum
    print(mean, minimum, maximum, range_)



if __name__ == '__main__':
    # For debugging purposes
    # dataset = NelsonData(patient_groups=['Control', 'Atypical'], normalize_standardize='Normalize')
    dataset = VirtualPopulation(['Control', 'Atypical'], 'Uniform', 'Standardize', 100, 10, 1)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    for batch in loader:
        print(batch)

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

