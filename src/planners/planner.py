# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

class PlannerConfig(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class Planner(object):
    def __init__(self, map_info, weights_path, device):
        self.map_info = map_info # map of environment
        self.weights_path = weights_path
        self.device = device

    def reset(self, init_state, vehicle_atts):
        '''
        Reset for a new rollout starting from the given initial state.
        '''
        pass

    def rollout(self, agent_obs, num_steps):
        '''
        Rollout agent given past observations of other agents for num_steps.

        Returns the resulting kinematic trajectory.
        '''
        pass

class PlannerNusc(object):
    def __init__(self, map_env, cfg):
        self.map_env = map_env # env information
        self.cfg = cfg

    def reset(self, init_state, vehicle_atts):
        '''
        Prepare for a new rollout starting from the given initial state.
        '''
        pass

    def rollout(self, agent_obs, num_steps, init_state=None):
        '''
        Rollout agent given full sequence of past (and possibly future) observations
        of all other agents for num_steps.
        If init_state is given starts from this state rather than the current
        init_state set in self.reset().

        Returns the resulting kinematic trajectory.
        '''
        pass