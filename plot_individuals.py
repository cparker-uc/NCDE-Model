# File Name: plot_individuals.py
# Author: Christopher Parker
# Created: Tue Aug 29, 2023 | 10:49P EDT
# Last Modified: Tue Aug 29, 2023 | 02:55P EDT

"""Code to plot individual patients and comparison graphs based on the
results of the individual classification success rate analysis"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from get_data import NelsonData, AblesonData


# So the indices for Nelson patients are:
#  - 0-14 Control
#  - 15-28 Atypical
#  - 29-43 Melancholic
#  - 44-57 Neither
#
# And for Ableson patients, they are:
#  - 0-36 Control
#  - 37-49 MDD

# But when we look at the Full VPOP, how are they organized?
# 
# For MDD, should be:
#  - Nelson Atypical (0-13), Melancholic (14-28), Neither (29-42)
#  - Ableson MDD (43-55)
# For Control, should be:
#  - Nelson Control (0-14)
#  - Ableson Control (15-51)

def main():
    nelson_dataset = NelsonData(
        patient_groups=['Control', 'Atypical', 'Melancholic', 'Neither'],
        normalize_standardize='None'
    )
    ableson_dataset = AblesonData(
        patient_groups=['Control', 'MDD'],
        normalize_standardize='None'
    )
    dataset = ConcatDataset((nelson_dataset, ableson_dataset))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10,10))
    for idx, (patient, _) in enumerate(dataset):
        # These are the 100% correctly classified control patients in the Full VPOP
        if idx in (0, 5, 6, 7, 60, 62, 63, 64, 65, 67, 69, 74, 76, 77, 79, 91, 92, 94):
            ax1.plot(patient[:,0], patient[:,1])
            ax2.plot(patient[:,0], patient[:,2])

        if idx in (61, 68, 70, 71, 75, 82, 86, 90, 93):
            ax3.plot(patient[:,0], patient[:,1])
            ax4.plot(patient[:,0], patient[:,2])

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#                                 MIT License                               #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#     Copyright (c) 2023 Christopher John Parker <parkecp@mail.uc.edu>      #
#                                                                           #
# Permission is hereby granted, free of charge, to any person obtaining a   #
# copy of this software and associated documentation files (the "Software"),#
# to deal in the Software without restriction, including without limitation #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,  #
# and/or sell copies of the Software, and to permit persons to whom the     #
# Software is furnished to do so, subject to the following conditions:      #
#                                                                           #
# The above copyright notice and this permission notice shall be included   #
# in all copies or substantial portions of the Software.                    #
#                                                                           #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS   #
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF                #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.    #
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY      #
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,      #
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE         #
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                    # 
#                                                                           #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

