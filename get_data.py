# File Name: get_nelson_data.py
# Author: Christopher Parker
# Created: Thu Apr 27, 2023 | 05:10P EDT
# Last Modified: Mon Dec 04, 2023 | 03:06P EST

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NonAugmentedDataset(Dataset):
    """Parent class for all of the following dataset classes, including the
    NelsonData, AblesonData, VirtualPopulation and FullVirtualPopulation
    classes"""

    def __init__(self, data_dir: str,
                 normalize_standardize: str):
        super().__init__()

        # Directory containing data files to load
        self.data_dir = data_dir

        # To be set in the child class
        self.group_info = {}
        self.patient_groups = []
        self.individual_number = 0
        self.sim = False

        # self.X contains 11 time points with 3 input channels
        #  (time, ACTH, CORT) for each time
        self.X = torch.zeros((0, 11, 3))
        # self.y indicates whether the patient was diagnosed with MDD
        self.y = torch.zeros((0,))

        # String to set the data to be Stardardized, Normalized or None (as
        #  it was collected)
        self.normalize_standardize = normalize_standardize

    def load_group(self, len: int, group: str, label: int,
                   X: torch.Tensor, y: torch.Tensor):
        """Load a patient group (Control, Atypical, MDD, etc...) and cat it
        onto the X and y tensors passed to this function. This iteratively
        builds the data and label tensors as the function is called for each
        patient group requested"""
        X_tmp = torch.zeros((len, 11, 3)) if not self.sim else torch.zeros((len, 20, 5))
        y_tmp = torch.zeros((len,)) if label==0 else torch.ones((len,))
        if self.sim:
            data_path = os.path.join(
                'Virtual Populations', f'Toy_{group}_NoNoise_{self.normalize_standardize}_1_24hr_test.txt'
            )
            X_tmp[0,...] = torch.load(data_path)
        else:
            for idx in range(len):
                data_path = os.path.join(
                    self.data_dir, f'{group}Patient{idx+1}.txt'
                )
                data = np.genfromtxt(data_path)
                X_tmp[idx,...] = torch.from_numpy(data)
        # We normalize the time steps, so that they are between 0
        #  and 1. Shouldn't have any impact, but it will
        #  make things somewhat cleaner
        # X_tmp[...,0] = normalize_time_series(X_tmp[...,0])
        X = torch.cat((X, X_tmp), 0)
        y = torch.cat((y, y_tmp), 0)
        return (X, y)

    def norm_data(self, X):
        """This function takes the unmodified data and normalizes or
        standardizes it based on the option selected, then returns it in the
        proper format"""
        if self.sim:
            return standardize_sriram_data(X)
        match self.normalize_standardize:
            # We break X down into columns, pass them to the relevant func
            #  then cat it back together along axis=2 (columns)
            case 'Normalize':
                # This option and the next are only valid for Nelson data, not
                #  Ableson data
                X = normalize_nelson_data(X)
            case 'Standardize':
                X = standardize_nelson_data(X)
            case 'StandardizeAbleson':
                X = standardize_ableson_data(X)
            case 'NormalizeAll':
                X = normalize_data(X)
            case 'StandardizeAll':
                X = standardize_data(X)
            case _:
                # Generic case returns X unchanged
                pass
        return X

    def __len__(self):
        "Return the total number of patients in the dataset"
        length = 0
        for group in self.patient_groups:
            length += self.group_info[group][0]
        return length

    def __getitem__(self, idx: int):
        "Get the requested patient data and label for index idx"
        if self.individual_number and len(self.patient_groups) == 1:
            return self.X.to(torch.float32), self.y.to(torch.float32)
        if self.sim and len(self.patient_groups) == 1:
            return self.X, self.y
        return self.X[idx,...], self.y[idx]


class NelsonData(NonAugmentedDataset):
    """This class loads the original Nelson TSST data, with no
    virtual patients included"""
    def __init__(self, patient_groups: list[str],
                 normalize_standardize: str, individual_number: int=0,
                 data_dir: str='Nelson TSST Individual Patient Data'):
        super().__init__(data_dir, normalize_standardize)

        # Length and label for each patient group in the dataset
        self.group_info = {
            'Control': [15, 0],
            'Melancholic': [15, 1],
            'Atypical': [14, 1],
            'Neither': [14, 1]
        }
        self.patient_groups = patient_groups
        self.individual_number = individual_number

        # Loop through the patient groups calling self.load_group
        for group in self.patient_groups:
            (self.X, self.y) = self.load_group(
                len=self.group_info[group][0],
                group=group,
                label=self.group_info[group][1],
                X=self.X, y=self.y
            )

        self.X = self.norm_data(self.X)

        if len(patient_groups) == 1 and individual_number:
            self.group_info[patient_groups[0]][0] = 1
            self.X = self.X[individual_number-1,...]
            self.y = self.y[individual_number-1]


class AblesonData(NonAugmentedDataset):
    """This class loads the original Ableson TSST data, with no
    virtual patients included"""
    def __init__(self, patient_groups: list[str],
                 normalize_standardize: str, individual_number: int=0,
                 data_dir: str='Ableson TSST Individual Patient Data '
                               '(Without First 30 Min)'):
        super().__init__(data_dir, normalize_standardize)

        # Just for testing MDD patients against Nelson Atypical
        if 'Atypical' in patient_groups:
            patient_groups[patient_groups.index('Atypical')] = 'MDD'
        # Length and label for each patient group in the dataset
        self.group_info = {
            'Control': [37, 0],
            'MDD': [13, 1],
        }
        self.patient_groups = patient_groups
        self.individual_number = individual_number

        # Loop through the patient groups calling self.load_group
        for group in self.patient_groups:
            (self.X, self.y) = self.load_group(
                len=self.group_info[group][0],
                group=group,
                label=self.group_info[group][1],
                X=self.X, y=self.y
            )
        self.X = self.norm_data(self.X)

        if len(patient_groups) == 1 and individual_number:
            self.group_info[patient_groups[0]][0] = 1
            self.X = self.X[individual_number-1,...]
            self.y = self.y[individual_number-1]

class SriramSimulation(NonAugmentedDataset):
    """This class loads the original Nelson TSST data, with no
    virtual patients included"""
    def __init__(self, patient_groups: list[str],
                 normalize_standardize: str,
                 data_dir: str='Nelson TSST Individual Patient Data'):
        super().__init__(data_dir, normalize_standardize)

        # Length and label for each patient group in the dataset
        self.group_info = {
            'Control': [1, 0],
            'Melancholic': [1, 1],
            'Atypical': [1, 1],
            'Neither': [1, 1]
        }
        self.patient_groups = patient_groups
        self.sim = True

        # self.X contains 20 time points with 5 input channels
        #  (time, ACTH, CORT) for each time
        self.X = torch.zeros((0, 20, 5))
        # self.y indicates whether the patient was diagnosed with MDD
        self.y = torch.zeros((0,))

        # Loop through the patient groups calling self.load_group
        for group in self.patient_groups:
            (self.X, self.y) = self.load_group(
                len=self.group_info[group][0],
                group=group,
                label=self.group_info[group][1],
                X=self.X, y=self.y
            )

        # self.X = self.norm_data(self.X)


def normalize_time_series(series_tensor):
    "Normalize the time steps so that they are in [0,1]"
    min = torch.min(series_tensor)
    s_range = torch.max(series_tensor) - torch.min(series_tensor)
    for series in series_tensor:
        series -= min
        series /= s_range
    return series_tensor


def normalize_nelson_data(X: torch.Tensor):
    """Normalize the ACTH and CORT of Nelson patients with the min and range of
    all patients in the dataset"""
    ACTHmin = torch.tensor(5.6, dtype=torch.float32)
    CORTmin = torch.tensor(2., dtype=torch.float32)
    ACTHrange = torch.tensor(160.5, dtype=torch.float32)
    CORTrange = torch.tensor(44., dtype=torch.float32)
    for series in X[...,1]:
        series -= ACTHmin
        series /= ACTHrange
    for series in X[...,2]:
        series -= CORTmin
        series /= CORTrange
    return X


def standardize_nelson_data(X: torch.Tensor):
    """Standardize the ACTH and CORT of Nelson patients with the mean and std
    of all patients in the dataset"""
    ACTHmean = torch.tensor(23.2961, dtype=torch.float32)
    CORTmean = torch.tensor(9.0687, dtype=torch.float32)
    ACTHstd = torch.tensor(11.1307, dtype=torch.float32)
    CORTstd = torch.tensor(5.4818, dtype=torch.float32)
    for series in X[...,1]:
        series -= ACTHmean
        series /= ACTHstd
    for series in X[...,2]:
        series -= CORTmean
        series /= CORTstd
    return X


def standardize_ableson_data(X: torch.Tensor):
    """Standardize the ACTH and CORT of Nelson patients with the mean and std
    of all patients in the dataset"""
    ACTHmean = torch.tensor(21.31, dtype=torch.float32)
    CORTmean = torch.tensor(10.043, dtype=torch.float32)
    ACTHstd = torch.tensor(12.462, dtype=torch.float32)
    CORTstd = torch.tensor(4.2194, dtype=torch.float32)
    for series in X[...,1]:
        series -= ACTHmean
        series /= ACTHstd
    for series in X[...,2]:
        series -= CORTmean
        series /= CORTstd
    return X


def standardize_sriram_data(X: torch.Tensor):
    """Standardize the ACTH and CORT of Nelson patients with the mean and std
    of all patients in the dataset"""
    # CRHmean = torch.tensor(0.7646, dtype=float)
    # ACTHmean = torch.tensor(4.6248, dtype=float)
    # CORTmean = torch.tensor(11.2867, dtype=float)
    # GRmean = torch.tensor(2.4473, dtype=float)
    # CRHstd = torch.tensor(0.5680, dtype=float)
    # ACTHstd = torch.tensor(3.6620, dtype=float)
    # CORTstd = torch.tensor(7.6905, dtype=float)
    # GRstd = torch.tensor(0.3119, dtype=float)
    # for series in X[...,1]:
    #     series -= CRHmean
    #     series /= CRHstd
    # for series in X[...,2]:
    #     series -= ACTHmean
    #     series /= ACTHstd
    # for series in X[...,3]:
    #     series -= CORTmean
    #     series /= CORTstd
    # for series in X[...,4]:
    #     series -= GRmean
    #     series /= GRstd
    # return X
    CRHmean = torch.tensor(0.5211, dtype=float)
    ACTHmean = torch.tensor(6.5110, dtype=float)
    CORTmean = torch.tensor(16.4809, dtype=float)
    GRmean = torch.tensor(1.6901, dtype=float)
    CRHstd = torch.tensor(0.5680, dtype=float)
    ACTHstd = torch.tensor(3.6620, dtype=float)
    CORTstd = torch.tensor(7.6905, dtype=float)
    GRstd = torch.tensor(0.3119, dtype=float)
    X = X.squeeze()
    X[...,1] = X[...,1] - CRHmean
    X[...,1] = X[...,1]/CRHstd
    X[...,2] = X[...,2] - ACTHmean
    X[...,2] = X[...,2]/ACTHstd
    X[...,3] = X[...,3] - CORTmean
    X[...,3] = X[...,3]/CORTstd
    X[...,4] = X[...,4] - GRmean
    X[...,4] = X[...,4]/GRstd
    return X.view(1, 20, 5)


def normalize_data(X: torch.Tensor):
    """Normalize data from Nelson or Ableson using the min and range of all
    patients in both datasets"""
    ACTHmin = torch.tensor(5.6, dtype=float)
    CORTmin = torch.tensor(1.2, dtype=float)
    ACTHrange = torch.tensor(160.5, dtype=float)
    CORTrange = torch.tensor(44.8, dtype=float)
    for series in X[...,1]:
        series -= ACTHmin
        series /= ACTHrange
    for series in X[...,2]:
        series -= CORTmin
        series /= CORTrange
    return X


def standardize_data(X: torch.Tensor):
    """Standardize data from Nelson or Ableson using the mean and std of all
    patients in both datasets"""
    ACTHmean = torch.tensor(23.6205, dtype=float)
    CORTmean = torch.tensor(9.2878, dtype=float)
    ACTHstd = torch.tensor(4.8611, dtype=float)
    CORTstd = torch.tensor(15.2958, dtype=float)
    for series in X[...,1]:
        series -= ACTHmean
        series /= ACTHstd
    for series in X[...,2]:
        series -= CORTmean
        series /= CORTstd
    return X


if __name__ == '__main__':
    # For debugging purposes
    dataset = NelsonData(patient_groups=['Control', 'Atypical'], normalize_standardize='Standardize')
    # dataset = VirtualPopulation(['Control', 'Atypical'], 'Uniform', 'Standardize', 100, 10, 1)
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


