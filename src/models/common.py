# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, layers,
                       nonlinearity=nn.ReLU,
                       use_norm=True,
                       ):
        '''
        :param layers: list of layer size (including input/output)
        '''
        super(MLP, self).__init__()

        nonlinarg = None
        if nonlinearity == nn.LeakyReLU:
            nonlinarg = [0.2]

        in_size = layers[0]
        out_channels = layers[1:]

        # input layer
        layers = []
        init_layer = nn.Linear(in_size, out_channels[0])
        layers.append(init_layer)
        # now the rest
        for layer_idx in range(1, len(out_channels)):
            fc_layer = nn.Linear(out_channels[layer_idx-1], out_channels[layer_idx])
            if use_norm:
                norm_layer = nn.LayerNorm(out_channels[layer_idx-1])
                layers.append(norm_layer)
            if nonlinarg is not None:
                layers.extend([nonlinearity(*nonlinarg), fc_layer])
            else:
                layers.extend([nonlinearity(), fc_layer])
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
        return x


def car_dynamics(kinematics, a, ddh,
                 dt, xix, yix, hix, six,
                 hdotix, vehicle_length, 
                 max_hdot, max_s):
    '''
    Based on kinematic Bicycle model. 
    Note car can't go backwards.
    '''
    newhdot = (kinematics[:, :, hdotix] + ddh * dt).clamp(-max_hdot, max_hdot)
    newh = kinematics[:, :, hix] + dt * kinematics[:, :, six].abs() / vehicle_length * newhdot
    news = (kinematics[:, :, six] + a * dt).clamp(0.0, max_s)
    newy = kinematics[:, :, yix] + news * newh.sin() * dt
    newx = kinematics[:, :, xix] + news * newh.cos() * dt

    newstate = torch.empty_like(kinematics)
    newstate[:, :, xix] = newx
    newstate[:, :, yix] = newy
    newstate[:, :, hix] = newh
    newstate[:, :, six] = news
    newstate[:, :, hdotix] = newhdot

    return newstate