# File Name: cnn_classification.py
# Author: Christopher Parker
# Created: Fri Dec 01, 2023 | 10:50P EST
# Last Modified: Thu Feb 01, 2024 | 08:36P EST


"""Use the CNN architecture to classify based on the weight matrices of
networks trained to fit the data"""

ITERS = 25
BATCH_SIZE = 1
DIRECTORY = 'Network States/Fitting Set 1'

from copy import copy
import re
import os
import torch
import numpy as np
from itertools import combinations
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from cnn import CNN

class FittingWeights(Dataset):
    def __init__(self, directory, file_list):
        super().__init__()
        self.directory = directory
        self.y = torch.zeros(0,)
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        model_state = torch.load(os.path.join(self.directory, filename))
        mat = model_state['hpa_net.2.weight'].to(torch.float64).view(1,11,11)
        return mat, torch.tensor(0, dtype=torch.float64) if filename[0] == 'C' else torch.tensor(1, dtype=torch.float64)


def main(ctrl_combination, mdd_combination):
    model = CNN(1, 16, 16, 32, 3, stride=1, padding=1, batch_size=BATCH_SIZE)
    model = model.double()
    files = os.listdir(DIRECTORY)
    no_dup_files = copy(files)
    # Remove all but the first fitting set from the list of files, so that we
    #  can make sure we aren't training and testing on NODEs from the same
    #  patient. We will add back in the appropriate files after removing test
    #  patients.
    pattern = re.compile('\s{1}\d+[.]+')
    for filename in files:
        if pattern.search(filename):
            no_dup_files.remove(filename)

    train_files = copy(no_dup_files)
    test_files = copy(no_dup_files)
    # Remove any extraneous files
    for filename in no_dup_files:
        if filename[:3] not in ['Con', 'Nei']:#, 'Nei', 'Nei', 'MDD']:
            train_files.remove(filename)
            test_files.remove(filename)
    # Remove the test patients
    filenames = []
    for filename in train_files:
        if filename[:3] == 'Con' and int(filename.split('ControlPatient')[1].split('.txt')[0]) in ctrl_combination:
            filenames.append(filename)
        if filename[:3] == 'Nei' and int(filename.split('NeitherPatient')[1].split('.txt')[0]) in mdd_combination:
            filenames.append(filename)
    for f in filenames:
        train_files.remove(f)

    # Remove the training patients
    filenames = []
    for filename in test_files:
        if filename[:3] == 'Con' and int(filename.split('ControlPatient')[1].split('.txt')[0]) not in ctrl_combination:
            filenames.append(filename)
        if filename[:3] == 'Nei' and int(filename.split('NeitherPatient')[1].split('.txt')[0]) not in mdd_combination:
            filenames.append(filename)
    for f in filenames:
        test_files.remove(f)

    # Now we need to add back in the NODEs from Fitting Sets 2, 3, 5 and 6
    # tmp_train_files = copy(train_files)
    # for idx, filename in enumerate(tmp_train_files):
    #     train_files.append(filename[:-4] + ' 2.txt')
    #     train_files.append(filename[:-4] + ' 3.txt')
        # train_files.append(filename[:-4] + ' 5.txt')
        # train_files.append(filename[:-4] + ' 6.txt')

    train_data = FittingWeights(DIRECTORY, train_files)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

    for _ in range(1,ITERS+1):
        for data, labels in loader:
            # Generate and train on 100 virtual patients for each real patient
            # for i in range(100):

            # Uniformly distributed noise from [0,1)
            # random_noise = torch.rand_like(data)
            # # Scale the noise to be 5% of the data magnitude
            # noise_magnitude = data*0.01
            # random_noise *= noise_magnitude
            # data = data + random_noise
            optimizer.zero_grad()
            pred_y = model(data)
            pred_y = pred_y.squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(pred_y, labels)
            loss.backward()
            optimizer.step()
            # If this is the first iteration, or a multiple of 10, present the
            #  user with a progress report
            # if (itr == 1) or (itr % 10 == 0):
            #     print(f"Iter {itr:04d} Batch {j}: loss = {loss.item():.6f}")

    test_data = FittingWeights(DIRECTORY, test_files)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    count = 0
    idxs = []
    combo_labels = torch.zeros((0,))
    combo_pred_y = torch.zeros((0,))
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            pred_y = model(data)
            pred_y = nn.functional.sigmoid(pred_y)

            combo_labels = torch.cat([combo_labels, labels], 0)
            combo_pred_y = torch.cat([combo_pred_y, pred_y], 0)

            if abs(pred_y - labels) < 0.5:
                idxs.append(i)
                count += 1
    return count, idxs, combo_labels, combo_pred_y

if __name__ == '__main__':
    combos = [
        [14, 12,  9, 15,  7,  3, 13, 11,  6,  2,  1,  10,  4,  8,  5],
        [ 1,  8, 13,  9, 12,  3,  7,  10,  4,  6,  5,  2, 14, 11],
        [11, 14,  5,  3,  4, 13, 15,  7,  9,  10,  6,  8,  1, 12,  2],
        [ 6, 13,  8, 14, 12,  10,  5,  2,  1,  3,  4, 11,  7,  9]
    ]
    loop_combos = [(i,j) for i in range(3) for j in range(3)]
    counts = []
    correct_patients = []
    all_labels = torch.zeros((0,))
    all_pred_y = torch.zeros((0,))
    # for idx, combination in enumerate(combinations(range(28), 3)):
    for idx, ctrl_combination in enumerate(combinations(range(1,16), 2)):
        for idx2, mdd_combination in enumerate(combinations(range(1,15), 2)):
    # for idx, (ctrl_num, mdd_num) in enumerate(loop_combos):
    #     ctrl_combination = tuple(
    #         combos[0][ctrl_num*5:(ctrl_num+1)*5]
    #     )
    #     mdd_combination = tuple(
    #         combos[2][mdd_num*5:((mdd_num+1)*5)
    #                         if (mdd_num+1)*5 < len(combos[2])
    #                         else len(combos[2])]
    #     )
            print(f'Training for combination: {ctrl_combination, mdd_combination=}')
            [count, idxs, labels, pred_ys] = main(ctrl_combination, mdd_combination)
            all_labels = torch.cat([all_labels, labels], 0)
            all_pred_y = torch.cat([all_pred_y, pred_ys], 0)
            counts.append(count)
            print(f"\tCumulative Mean = {np.mean(counts)/10}")
            correct_patients.append(idxs)
    [fpr, tpr, thresholds] = roc_curve(
        all_labels, all_pred_y,
    )
    roc_auc = roc_auc_score(
        all_labels, all_pred_y,
    )
    disp = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        estimator_name='Control vs Neither'
    )
    disp.plot()
    plt.show()

    np.savetxt('CNN_FittingSet1_Neither_labels_4test_allcombos.txt', all_labels)
    np.savetxt('CNN_FittingSet1_Neither_pred_y_4test_allcombos.txt', all_pred_y)
    torch.save(counts, 'NeitherCorrectCounts_FittingSet1_batchsize1_25iter_4test_allcombos.txt')
    torch.save(correct_patients, 'NeitherCorrectPatients_FittingSet1_batchsize1_25iter_4test_allcombos.txt')

    print(f"Overall success rate: {np.mean(counts)/10}")
