# File Name: neural_cde.py
# Author: Christopher Parker
# Created: Thu Jul 20, 2023 | 12:43P EDT
# Last Modified: Fri Nov 24, 2023 | 09:36P EST

"""Classes for implementation of Neural CDE networks"""

import torch
import torch.nn as nn
import numpy as np
import torchcde

class DEControl(nn.Module):
    """Passed to the cdeint function as the control, contains the mechanistic
    solution to the equation"""

    def __init__(self, sol, params, t=None, device=torch.device('cpu'), **kwargs):
        """
        """
        super().__init__()

        if t is None:
            t = torch.linspace(0, 140, 1)

        self.register_buffer('_t', t)
        self.register_buffer('_sol', sol)
        self.device = device

        for key, val in params.items():
            self.register_buffer(key, val)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)
        maxlen = self._b.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        """Return the nearest time point from the solution curve"""
        def find_nearest(array, value):
            idx = (torch.abs(array-value)).argmin()
            return idx

        sol_pt = self._sol[[find_nearest(t, u) for u in self._sol[...,0].squeeze()],...]
        return sol_pt

    def derivative(self, t):
        sol_pt = self.evaluate(t)
        dy = self.ode_rhs(sol_pt, t)
        return dy

    def stress(self, t):
        if t < 30:
            return 0
        return torch.exp(-self.kdStress*(t-30))

    def ode_rhs(self, y, t):
        dy = torch.zeros_like(y, requires_grad=False)

        # Apologies for the mess here, but typing out ODEs in Python is a bit of a
        #  chore
        wCRH = self.R0CRH + self.RCRH_CRH*y[...,0] \
            + self.RSS_CRH*self.stress(t.detach()) + self.RGR_CRH*y[...,3]
        FCRH = (self.MaxCRH*self.tsCRH)/(1 + torch.exp(-self.sigma*wCRH))
        dy[...,0] = FCRH - self.tsCRH*y[...,0]

        wACTH = self.R0ACTH + self.RCRH_ACTH*y[...,0] \
            + self.RGR_ACTH*y[...,3]
        FACTH = (self.MaxACTH*self.tsACTH)/(1 + torch.exp(-self.sigma*wACTH)) + self.BasalACTH
        dy[...,1] = FACTH - self.tsACTH*y[...,1]

        wCORT = self.R0CORT + self.RACTH_CORT*y[...,1]
        FCORT = (self.MaxCORT*self.tsCORT)/(1 + torch.exp(-self.sigma*wCORT)) + self.BasalCORT
        dy[...,2] = FCORT - self.tsCORT*y[...,2]

        wGR = self.R0GR + self.RCORT_GR*y[...,2] + self.RGR_GR*y[...,3]
        FGR = self.ksGR/(1 + torch.exp(self.sigma*wGR))
        dy[...,3] = FGR - self.kdGR*y[...,3]
        return dy


class CDEFunc(torch.nn.Module):
    """CDEs are defined as: z_t = z_0 + \\int_{0}^t f_{theta}(z_s) dX_s, where
    f_{theta} is a neural network (and X_s is a rough path controlling the
    diff eq. This class defines f_{theta}"""
    def __init__(self, input_channels, hidden_channels, device):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Define the layers of the NN, with 128 hidden nodes (arbitrary)
        self.linear1 = torch.nn.Linear(hidden_channels, 128).to(device)
        self.linear2 = torch.nn.Linear(128, hidden_channels*input_channels).to(device)

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, t, z):
        """t is passed as an argument by the solver, but it is unused in most
        cases"""
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        # The first author of the NCDE paper (Kidger) suggests that using tanh
        #  for the final activation leads to better results
        z = z.tanh()

        # The output represents a linear map from R^input_channels to
        #  R^hidden_channels, so it takes the form of a
        #  (hidden_channels x input_channels) matrix
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    """This class packages the CDEFunc class with the torchcde NCDE solver,
    so that when we call the instance of NeuralCDE it solves the system"""
    def __init__(self, input_channels: int, hidden_channels: int,
                 output_channels: int,
                 t_interval: torch.Tensor=torch.tensor((0,1), dtype=torch.float32),
                 device: torch.device=torch.device('cpu'),
                 interpolation: str='cubic', dropout: float=0.,
                 prediction: bool=False, dense_domain: torch.Tensor=torch.linspace(0,140,50),
                 atol=1e-6, rtol=1e-4, adjoint_atol=1e-6, adjoint_rtol=1e-4):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels, device)

        # self.initial represents l_{theta}^2 in the equation
        #  z_0 = l_{theta}^2 (x)
        # This is essentially augmenting the dimension with a linear map,
        #  something Massaroli et al warned against
        self.initial = torch.nn.Linear(input_channels, hidden_channels).to(device)
        self.dropout = torch.nn.Dropout(p=dropout, inplace=True).to(device)

        # self.readout represents l_{theta}^1 in the equation
        #  y ~ l_{theta}^1 (z_T)
        self.readout = torch.nn.Linear(hidden_channels, output_channels).to(device)

        self.interpolation = interpolation
        # This flag tells us whether we want only the last point of the
        #  integration or all of the points along the way
        self.prediction = prediction
        if self.prediction:
            self.t_interval = t_interval.to(device)
        else:
            self.t_interval = torch.tensor((0,1), dtype=torch.float32)
        self.dense_domain = dense_domain

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

        self.atol = atol
        self.rtol = rtol
        self.adjoint_atol = adjoint_atol
        self.adjoint_rtol = adjoint_rtol

    def forward(self, coeffs):
        """coeffs is the coefficients that describe the spline between the
        datapoints. In the case of cubic interpolation (the default), this
        is a, b, 2c and 3d because the derivative of the spline is used more
        often with cubic Hermite splines"""
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=self.t_interval)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        elif self.interpolation == 'mechanistic':
            X = coeffs # Probably should change the name of the input
        else:
            raise ValueError(
                "Only cubic and linear interpolation are implemented"
            )

        # The initial hidden state
        #   (a linear map from the first observation)
        X0 = X.evaluate(X.interval[0]) # evaluate the spline at its first point
        z0 = self.initial(X0)
        self.dropout(z0)

        # Solve the CDE
        z_T = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=self.dense_domain,
            **{
                'atol': self.atol, 'rtol': self.rtol,
                'adjoint_atol': self.adjoint_atol,
                'adjoint_rtol': self.adjoint_rtol,
                'method': 'rk4', 'adjoint_method': 'rk4',
            }
        )
        # cdeint returns the initial value and terminal value from the
        #  integration, we only need the terminal
        if not self.prediction:
            z_T = z_T[:,1]

        # Convert z_T to y with the readout linear map
        pred_y = self.readout(z_T)

        return pred_y


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

