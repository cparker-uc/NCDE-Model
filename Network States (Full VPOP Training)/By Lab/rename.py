# File Name: rename.py
# Author: Christopher Parker
# Created: Fri Jul 28, 2023 | 03:55P EDT
# Last Modified: Fri Jul 28, 2023 | 04:06P EDT

"""Rename all files in the By Lab directories from Control vs MDD to Nelson vs
Ableson"""


import os
import sys

for directory in os.listdir():
    if not directory.split(' ')[0] == 'Nelson':
        continue

    for inner_directory in os.listdir(directory):
        if not inner_directory.split(' ')[0] == 'Ableson':
            continue
        current_full_directory = os.path.join(directory, inner_directory)

        for network_state_file in os.listdir(current_full_directory):
            if not network_state_file.split('_')[0] == 'NN':
                continue
            current_filepath = os.path.join(current_full_directory, network_state_file)
            new_filename = network_state_file.split('Control')[0] + 'Nelson' + network_state_file.split('Control')[1]
            new_filename = new_filename.split('MDD')[0] + 'Ableson' + new_filename.split('MDD')[1]
            new_filepath = os.path.join(current_full_directory, new_filename)
            os.rename(current_filepath, new_filepath)
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

