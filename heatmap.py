# File Name: heatmap.py
# Description: 
# Author: Christopher Parker
# Created: Wed Dec 13, 2023 | 11:33P EST
# Last Modified: Thu Dec 14, 2023 | 03:16P EST

import torch
import matplotlib.pyplot as plt

control_model_state = torch.load('/Users/christopher/Documents/PTSD/NCDE Model.nosync/Network States/Old Individual Fittings (11 nodes)/ControlPatient0.txt')
# atypical_model_state = torch.load('/Users/christopher/Documents/PTSD/NCDE Model.nosync/Network States/Old Individual Fittings (11 nodes)/AtypicalPatient0.txt')

control_mat = control_model_state['hpa_net.2.weight']
# atypical_mat = atypical_model_state['hpa_net.2.weight']

fig, ax = plt.subplots()
hmap = ax.imshow(control_mat, cmap='hot')
fig.colorbar(hmap)
ax.set_title('Nelson Control Patient 1 NODE Weight Matrix Heatmap')
# fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 1.25]})

# control_hmap = ax1.imshow(control_mat, cmap='hot')
# atypical_hmap = ax2.imshow(atypical_mat, cmap='hot')
# fig.colorbar(atypical_hmap)
# fig.suptitle('Nelson Control vs Atypical NODE Weight Heatmaps')
# ax1.set_title('Control')
# ax2.set_title('Atypical')

plt.show()


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

