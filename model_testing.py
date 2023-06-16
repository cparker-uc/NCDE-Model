# File Name: model_testing.py
# Author: Christopher Parker
# Created: Fri Apr 28, 2023 | 09:23P EDT
# Last Modified: Thu May 04, 2023 | 12:35P EDT

import torch
import torchcde
import pandas as pd
from get_nelson_data import nelsonData_indiv
from model_training import NeuralCDE

"""This script will run each trained NCDE model against the single excluded
patient and catalogue the number of successes and failures."""

def test_indiv(state_dict, excluded_num):
    """Run a single network by loading the state_dict and testing on
    Patient #excluded_num"""

    # Load patient #exclude_num data and label
    data, label = nelsonData_indiv(excluded_num)

    device = torch.device('cpu')

    model = NeuralCDE(3, 11, 1)
    model.load_state_dict(state_dict)
    model.to(device)

    # Compute the Hermite cubic spline coefficients for use by the network
    data_coeffs = torchcde.\
        hermite_cubic_coefficients_with_backward_differences(data)

    # Pass the coeffs through the network and record the output
    pred_y = model(data_coeffs).squeeze(-1)
    print(pred_y)
    pred_y = torch.sigmoid(pred_y)
    print(pred_y)

    # The difference between the label and the predicted y is the error in the
    #  prediction
    error = torch.abs(label - pred_y)

    # Rounding the predicted y to see if it was successful
    rounded_y = torch.round(pred_y)
    success = not torch.abs(label - rounded_y)

    return success, pred_y.detach().numpy(), error.detach().numpy()
def main():
    "Main function that loops over all patients and calls test_indiv"
    performance_df = pd.DataFrame(columns=('Success', 'Prediction', 'Error'))

    for patient_num in range(58):
        state_file = ('NCDE_state_128Hnodes_8Hchannels_control'
                      f'-vs-MDD_classification_exclude{patient_num}.txt')
        state_dict = torch.load(state_file)
        success, prediction, error = test_indiv(state_dict, patient_num)
        print(success, prediction, error)

        # Add the performance metrics to the DataFrame as a new row
        performance_df = pd.concat(
            (
                performance_df,
                pd.DataFrame.from_dict(
                    {
                        'Success': (success,),
                        'Prediction': prediction,
                        'Error': error,
                    },
                )
            ),
        )
        performance_df.index = range(1, patient_num+2)

    with pd.ExcelWriter(
            'NCDE_MDDvsControl_classification_performance_cross-entropy.xlsx') as writer:
        performance_df.to_excel(writer)
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
