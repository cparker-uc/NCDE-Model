# File Name: cnn_classification.py
# Author: Christopher Parker
# Created: Fri Dec 01, 2023 | 10:50P EST
# Last Modified: Tue Dec 19, 2023 | 10:50P EST


"""Use the CNN architecture to classify based on the weight matrices of
networks trained to fit the data"""

ITERS = 25
BATCH_SIZE = 1
DIRECTORY = 'Network States/New Individual Fittings (11 nodes)'

from copy import copy
import os
import torch
import numpy as np
from itertools import combinations
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from cnn import CNN, CNN_oldstyle

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
        # mat = model_state['net.2.weight'].to(torch.float64).view(1,32,32)
        # mat = model_state['net.2.weight'].to(torch.float64).view(1,11,11)
        return mat, torch.tensor(0, dtype=torch.float64) if filename[0] == 'C' else torch.tensor(1, dtype=torch.float64)


def main(combination):
    model = CNN(1, 16, 16, 32, 3, stride=1, padding=1, batch_size=BATCH_SIZE)
    model = model.double()
    files = os.listdir(DIRECTORY)
    train_files = copy(files)
    test_files = copy(files)
    # Remove any extraneous files
    for filename in files:
        if filename[:3] not in ['Con', 'Aty']:#, 'Mel', 'Nei', 'MDD']:
            train_files.remove(filename)
            test_files.remove(filename)
    # Remove the test patients
    filenames = []
    for idx, filename in enumerate(train_files):
        if idx in combination:
            filenames.append(filename)
    for f in filenames:
        train_files.remove(f)

    # Remove the training patients
    filenames = []
    for idx, filename in enumerate(test_files):
        if idx not in combination:
            filenames.append(filename)
    for f in filenames:
        test_files.remove(f)

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
    counts = []
    correct_patients = []
    all_labels = torch.zeros((0,))
    all_pred_y = torch.zeros((0,))
    for idx, combination in enumerate(combinations(range(29), 3)):
        if idx % 1 == 0:
            print(f'Training for combination: {combination}')
        [count, idxs, labels, pred_ys] = main(combination)
        all_labels = torch.cat([all_labels, labels], 0)
        all_pred_y = torch.cat([all_pred_y, pred_ys], 0)
        counts.append(count)
        print(f"\tCumulative Mean = {np.mean(counts)/3}")
        correct_patients.append(idxs)
    [fpr, tpr, thresholds] = roc_curve(
        all_labels, all_pred_y,
    )
    roc_auc = roc_auc_score(
        all_labels, all_pred_y,
    )
    disp = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        estimator_name='Control vs Atypical'
    )
    disp.plot()
    plt.show()

    torch.save(counts, 'AtypicalCorrectCounts_newFits_batchsize1_25iter.txt')
    torch.save(correct_patients, 'AtypicalCorrectPatients_newFits_batchsize1_25iter.txt')

    print(f"Overall success rate: {np.mean(counts)/3}")
