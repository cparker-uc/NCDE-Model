# File Name: get_augmented_data.py
# Author: Christopher Parker
# Created: Thu Jul 20, 2023 | 03:19P EDT
# Last Modified: Fri Aug 11, 2023 | 10:12P EDT

"""Loads the datasets that have been augmented with noise to create virtual
patients"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BaseVirtualPopulation(Dataset):
    """Base class to be inherited by dataset-specific classes below"""
    def __init__(self, method: str, normalize_standardize: str,
                 num_per_patient: int, control_combination: tuple=(),
                 mdd_combination: tuple=(),
                 test: bool=False, label_smoothing: float=0.,
                 noise_magnitude: float=0., pop_number: int=0,
                 no_test_patients: bool=False):
        # TO BE SET IN THE CHILD CLASS
        self.group_info = {}
        self.patient_groups = []

        # PARAMETERS DESCRIBING HOW THE VIRTUAL POPULATION WAS GENERATED
        self.method = method
        self.label_smoothing = label_smoothing
        self.noise_magnitude = noise_magnitude
        # Number of virtual patients corresponding to each real patient
        self.num_per_patient = num_per_patient
        # Number of patients used for creating virtual patients
        self.num_patients = (0,)
        self.normalize_standardize = normalize_standardize

        # self.X contains 11 time points with 3 input channels
        #  (time, ACTH, CORT) for each time
        self.X = torch.zeros((0, 11, 3))
        # self.y indicates whether the patient was diagnosed with MDD
        self.y = torch.zeros((0,))

        # Flag indicating whether we need the training or test data
        self.test = test

        # If we have combinations of test patients provided, combine them into
        #  a tuple. Otherwise, set the population number to load
        self.combinations = ()
        if control_combination and mdd_combination:
            self.combinations = (control_combination, mdd_combination)
        if pop_number:
            self.pop_number = pop_number

        self.no_test_patients = no_test_patients
        self.plus_ableson_mdd = False
        self.toy_data = False

    def load_group(self, group: str, label: int,
                   X: torch.Tensor, y: torch.Tensor, test: bool):
        """Load a patient group (Control, Atypical, MDD, etc...) and cat it
        onto the X and y tensors passed to this function. This iteratively
        builds the data and label tensors as the function is called for each
        patient group requested"""
        # Note that we use label to index self.combinations and
        #  self.num_patients in this function. This is valid because we always
        #  expect the values for Control to appear first in these tuples
        if self.toy_data:
            vpop_and_train = torch.cat((
                torch.load(f'Virtual Populations/Toy_{group}_{self.method}{self.noise_magnitude}_{self.normalize_standardize}_{self.num_per_patient}_{self.t_end}hr.txt'),
                torch.load(f'Virtual Populations/Toy_{group}_{self.method}{self.noise_magnitude}_{self.normalize_standardize}_{self.num_per_patient}_{self.t_end}hr_test.txt')
            ))
        elif not self.combinations and self.no_test_patients:
            vpop_and_train = None
            vpop = torch.load(
                f'Virtual Populations/{group}_{self.method}{self.noise_magnitude}_'
                f'{self.normalize_standardize}_{self.num_per_patient}_'
                f'noTestPatients.txt'
            )
        elif not self.combinations:
            vpop_and_train = torch.load(
                f'Virtual Populations/{group}_{self.method}_'
                f'{self.normalize_standardize}_{self.num_per_patient}_'
                f'{self.num_patients[0]}_{self.pop_number}.txt'
            )
        else:
            vpop_and_train = torch.load(
                f'Virtual Populations/{group}_'
                f'{self.method}{self.noise_magnitude if self.noise_magnitude else ""}_'
                f'{self.normalize_standardize}_{self.num_per_patient}_'
                f'testPatients{self.combinations[label]}_'
                f'{"fixedperms" if not self.plus_ableson_mdd else "plusAblesonMDD0and12"}.txt'
            )
        if not isinstance(vpop_and_train, torch.Tensor):
            X_tmp = vpop
        elif test:
            X_tmp = vpop_and_train[self.num_patients[label]*self.num_per_patient:,...]
        else:
            X_tmp = vpop_and_train[
                :self.num_patients[label]*self.num_per_patient,...
            ]

        X = torch.cat((X, X_tmp), 0)

        y_tmp = torch.zeros((X_tmp.shape[0],)) if label==0 \
                    else torch.ones((X_tmp.shape[0],))
        y = torch.cat((y, y_tmp), 0)
        return (X, y)

    def __len__(self):
        "Return the total number of patients in the dataset"
        length = 0
        if self.test:
            # If we have fixed combinations of test patients, return the sum of
            #  the combination lengths
            if self.combinations:
                return np.sum([len(combo) for combo in self.combinations])

            if self.toy_data:
                return np.sum(self.num_patients)*self.num_per_patient
            # Otherwise, we must be using the NelsonVirtualPopulation with
            #  the old pop_number way of dividing the groups for testing, so
            #  just add the group length minus num_patients for each group
            for idx, group in enumerate(self.patient_groups):
                length += self.group_info[group][0] - self.num_patients[idx]
            return length

        return np.sum(self.num_patients)*self.num_per_patient

    def __getitem__(self, idx):
        "Get the requested patient data and label for index idx"
        return self.X[idx,...], self.y[idx]


class NelsonVirtualPopulation(BaseVirtualPopulation):
    def __init__(self, patient_groups, method, normalize_standardize,
                 num_per_patient, pop_number=0,
                 control_combination=(), mdd_combination=(),
                 test=False, label_smoothing=0.,
                 noise_magnitude=0., no_test_patients=False,
                 plus_ableson_mdd=False):
        super().__init__(method, normalize_standardize,
                         num_per_patient, control_combination, mdd_combination,
                         test, label_smoothing, noise_magnitude, pop_number,
                         no_test_patients)

        # Length and label for each patient group in the dataset
        self.group_info = {
            'Control': (15, 0),
            'Melancholic': (15, 1),
            'Atypical': (14, 1),
            'Neither': (14, 1)
        }
        self.patient_groups = patient_groups

        if not self.combinations and no_test_patients:
            self.num_patients = (
                self.group_info[patient_groups[0]][0],
                self.group_info[patient_groups[1]][0]
            )
        elif not self.combinations:
            # Because we will never be using more than 2 groups at a time, and
            #  all of the old virtual populations generated have
            #  num_patients=10, I'm just setting this to a tuple of 10s for
            #  convenience
            self.num_patients = (10, 10)
        else:
            # Otherwise, we set num_patients to the length of the groups minus the
            #  corresponding number of test patients
            self.num_patients = (
                self.group_info[patient_groups[0]][0]-len(control_combination),
                self.group_info[patient_groups[1]][0]-len(mdd_combination)
            )

        for group in self.patient_groups:
            if group == 'Atypical': self.plus_ableson_mdd = plus_ableson_mdd
            (self.X, self.y) = self.load_group(
                group, self.group_info[group][1],
                self.X, self.y, self.test
            )


class FullVirtualPopulation(BaseVirtualPopulation):
    def __init__(self, method, noise_magnitude,
                 normalize_standardize, num_per_patient,
                 control_combination, mdd_combination,
                 test=False, label_smoothing=0.):
        super().__init__(
            method, normalize_standardize, num_per_patient,
            control_combination, mdd_combination, test, label_smoothing,
            noise_magnitude
        )
        # Length and label for each patient group in the dataset
        self.group_info = {
            'Control': (52, 0),
            'MDD': (56, 1)
        }
        self.patient_groups = ['Control', 'MDD']

        self.num_patients = (
            self.group_info['Control'][0] - len(control_combination),
            self.group_info['MDD'][0] - len(mdd_combination)
        )

        for group in self.patient_groups:
            (self.X, self.y) = self.load_group(
                group, self.group_info[group][1],
                self.X, self.y, test
            )


class FullVirtualPopulation_ByLab(BaseVirtualPopulation):
    def __init__(self, method, noise_magnitude,
                 normalize_standardize, num_per_patient,
                 nelson_combination, ableson_combination,
                 test=False):
        super().__init__(
            method, normalize_standardize, num_per_patient,
            nelson_combination, ableson_combination,
            test, noise_magnitude=noise_magnitude
        )
        # Length and label for each patient group in the dataset
        self.group_info = {
            'Nelson': (58, 0),
            'Ableson': (50, 1)
        }
        self.patient_groups = ['Nelson', 'Ableson']

        self.num_patients = (
            self.group_info['Nelson'][0] - len(nelson_combination),
            self.group_info['Ableson'][0] - len(ableson_combination)
        )

        for group in self.patient_groups:
            (self.X, self.y) = self.load_group(
                group, self.group_info[group][1],
                self.X, self.y, test
            )


class ToyDataset(BaseVirtualPopulation):
    def __init__(self, test, noise_magnitude=0.05, method='Uniform',
                 num_per_patient=1000, normalize_standardize='None',
                 t_end=24):
        super().__init__(method, normalize_standardize, num_per_patient,
                         noise_magnitude=noise_magnitude, test=test)

        self.group_info = {
            'Control': (2, 0),
            'Atypical': (2, 1)
        }

        self.num_patients = (1, 1)

        self.toy_data = True
        self.X = torch.zeros(0, 20, 5)
        self.t_end = t_end

        for group in ['Control', 'Atypical']:
            (self.X, self.y) = self.load_group(
                group, self.group_info[group][1],
                self.X, self.y, test
            )


# For debugging purposes
if __name__ == '__main__':
    # data = NelsonVirtualPopulation(
    #     ['Control', 'Atypical'], 'Uniform', 'Standardize', 100,
    #     control_combination=(0,9,3,7,4), mdd_combination=(0,7,12,8,11),
    # )
    data = FullVirtualPopulation(
        'Uniform', 0.1, 'StandardizeAll', 100, (30,11,3,38,29),
        (41,8,15,16,33)
    )
    loader = DataLoader(data)
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

