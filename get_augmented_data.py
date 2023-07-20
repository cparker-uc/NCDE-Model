# File Name: get_augmented_data.py
# Author: Christopher Parker
# Created: Thu Jul 20, 2023 | 03:19P EDT
# Last Modified: Thu Jul 20, 2023 | 03:33P EDT

"""Loads the datasets that have been augmented with noise to create virtual
patients"""

import torch
from torch.utils.data import Dataset


class BaseVirtualPopulation(Dataset):
    """Base class to be inherited by dataset-specific classes below"""
    def __init__(self, patient_groups: list[str], method: str,
                 normalize_standardize: str, num_per_patient: int,
                 num_patients: int, pop_number: int=0,
                 control_combination: tuple=(), mdd_combination: tuple=(),
                 fixed_perms: bool=False, test: bool=False,
                 label_smoothing: float=0., noise_magnitude: float=0.):
        pass

class NelsonVirtualPopulation(BaseVirtualPopulation):
    def __init__(self, patient_groups, method, normalize_standardize,
                 num_per_patient, num_patients, pop_number=None,
                 control_combination=None, mdd_combination=None,
                 fixed_perms=False, test=False, label_smoothing=0.,
                 noise_magnitude=None):
        super().__init__(patient_groups, method, normalize_standardize,
                         num_per_patient, num_patients, pop_number,
                         control_combination, mdd_combination, fixed_perms,
                         test, label_smoothing, noise_magnitude)
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
                        fixed_perms=fixed_perms, noise_magnitude=noise_magnitude
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
                        fixed_perms=fixed_perms, noise_magnitude=noise_magnitude

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


class FullVirtualPopulation_ByLab(Dataset):
    def __init__(self, method, noise_magnitude,
                 normalize_standardize, num_per_patient,
                 nelson_combination, ableson_combination,
                 test=False):
        self.num_per_patient = num_per_patient
        self.num_nelson_patients = 58 - len(nelson_combination)
        self.num_ableson_patients = 50 - len(ableson_combination)
        self.X = torch.zeros((0, 11, 3), dtype=float)
        self.y = torch.zeros((0,), dtype=float)
        self.test = test
        self.combinations = nelson_combination + ableson_combination

        for lab in ['Nelson', 'Ableson']:
            if self.test:
                _, data = load_full_vpop_combinations_by_lab(
                    lab, method, normalize_standardize,
                    num_per_patient,
                    nelson_combination if lab == 'Nelson' else ableson_combination,
                    self.num_nelson_patients if lab == 'Nelson' else self.num_ableson_patients,
                    noise_magnitude=noise_magnitude
                )
            else:
                data, _ = load_full_vpop_combinations_by_lab(
                    lab, method, normalize_standardize,
                    num_per_patient,
                    nelson_combination if lab == 'Nelson' else ableson_combination,
                    self.num_nelson_patients if lab == 'Nelson' else self.num_ableson_patients,
                    noise_magnitude=noise_magnitude
                )
            self.X = torch.cat((self.X, data), 0)
            if lab == 'Nelson':
                if self.test:
                    self.y = torch.cat(
                        (self.y, torch.zeros(len(nelson_combination))), 0
                    )
                else:
                    self.y = torch.cat(
                        (self.y, torch.zeros(num_per_patient*self.num_nelson_patients)), 0
                    )
            else:
                if self.test:
                    self.y = torch.cat(
                        (self.y, torch.ones(len(ableson_combination))), 0
                    )
                else:
                    self.y = torch.cat(
                        (self.y, torch.ones(num_per_patient*self.num_ableson_patients)), 0
                    )

    def __len__(self):
        if self.test:
            return len(self.combinations)
        else:
            return (self.num_nelson_patients + self.num_ableson_patients)*self.num_per_patient

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


def load_full_vpop_combinations(patient_group, method, normalize_standardize,
                                num_per_patient, combination, num_patients,
                                noise_magnitude=None):
    vpop_and_train = torch.load(f'Virtual Populations/{patient_group}'
                                f'_{method}{noise_magnitude if noise_magnitude else ""}'
                                f'_{normalize_standardize}_{num_per_patient}_'
                                f'testPatients{combination}_fixedperms.txt')
    vpop = vpop_and_train[:num_patients*num_per_patient,...]
    test = vpop_and_train[num_patients*num_per_patient:,...]
    return vpop, test


def load_full_vpop_combinations_by_lab(lab, method, normalize_standardize, num_per_patient,
              combination, num_patients, noise_magnitude):
    vpop_and_train = torch.load(f'Virtual Populations/{lab}'
                                f'_{method}{noise_magnitude}'
                                f'_{normalize_standardize}_{num_per_patient}_'
                                f'testPatients{combination}_fixedperms.txt')
    vpop = vpop_and_train[:num_patients*num_per_patient,...]
    test = vpop_and_train[num_patients*num_per_patient:,...]
    return vpop, test



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

