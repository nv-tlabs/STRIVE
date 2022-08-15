# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn

import numpy as np

from losses.common import log_normal
from utils.transforms import transform2frame
import datasets.nuscenes_utils as nutils

class TgtMatchingLoss(nn.Module):
    '''
    Loss to encourage future pred to be close to some target while staying
    likely under the motion prior.
    '''
    def __init__(self, loss_weights):
        '''
        :param loss_weights: dict of weightings for loss terms
        '''
        super(TgtMatchingLoss, self).__init__()
        self.loss_weights = loss_weights
        self.motion_prior_loss = MotionPriorLoss()

    def forward(self, future_pred, tgt_traj, z, prior_out):
        ''' 
        :param future_pred: NA x T x 4 UNNORMALIZED
        :param tgt_traj: NA x T x 4 UNNORMALIZED
        :param z: NA x D
        :param prior_out: tuple of (mean, var) each of size (NA x D)
        '''
        loss_out = {}
        loss = 0.0

        if self.loss_weights['match_ext'] > 0.0:
            # matching error
            tgt_loss = torch.sum((future_pred - tgt_traj)**2, dim=-1)
            loss = loss + self.loss_weights['match_ext']*tgt_loss.mean()
            loss_out['match_ext_loss'] = tgt_loss

        if self.loss_weights['motion_prior_ext'] > 0.0:
            # motion prior
            motion_prior_loss = self.motion_prior_loss(z, prior_out)
            loss = loss + self.loss_weights['motion_prior_ext']*tgt_loss.mean()
            loss_out['motion_prior_ext_loss'] = motion_prior_loss

        loss_out['loss'] = loss

        return loss_out

class AdvGenLoss(nn.Module):
    '''
    Loss to encourage agents to create adversarial scenario for a given target.
    '''
    def __init__(self, loss_weights, veh_att, mapixes, map_env, init_z, ptr,
                    veh_coll_buffer=0.0,
                    crash_loss_min_time=0,
                    crash_loss_min_infront=None
                    ):
        '''
        :param loss_weights: dict of weightings for loss terms
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param mapixes: map index corresponding to ALL agents (NA,)
        :param map_env: the map environment holding map info
        :param init_z: (NA-B) x D initial latent z of all non-target agents
        :param crash_loss_min_time: only computes loss using times past this threshold
        :param crash_loss_min_infront: if not None [-1, 1], any attacker with cosine similarity < this threshold will
                                        be ignored (i.e. provide no loss to cause a crash) if attacker is being optimized
        '''
        super(AdvGenLoss, self).__init__()
        self.loss_weights = loss_weights
        self.init_z = init_z
        self.motion_prior_loss = MotionPriorLoss()
        # these are only computed on non-planner vehicles
        self.ptr = ptr
        self.graph_sizes = self.ptr[1:] - self.ptr[:-1]
        self.ego_mask = torch.zeros((veh_att.size(0)), dtype=torch.bool)
        self.ego_mask[self.ptr[:-1]] = True
        # vehicle collision will only be on non-ego, so need to adjust ptr accordingly
        self.nonego_ptr = self.ptr - torch.arange(len(self.ptr)).to(self.ptr)
        self.veh_coll_loss = VehCollLoss(veh_att,
                                          buffer_dist=veh_coll_buffer,
                                          ptr=self.ptr)
        self.env_coll_loss = EnvCollLoss(veh_att[~self.ego_mask], mapixes[~self.ego_mask], map_env)
        self.crash_min_t = crash_loss_min_time
        self.crash_min_infront = crash_loss_min_infront
        if self.crash_min_infront is not None:
            assert(self.crash_min_infront >= -1)
            assert(self.crash_min_infront <= 1)       

    def forward(self, future_pred, tgt_traj, z, prior_out,
                    return_mins=False,
                    attack_agt_idx=None):
        ''' 
        :param future_pred: NA x T x 4 UNNORMALIZED output of the motion model where idx=0 is modeling the planner
        :param tgt_traj: B x T x 4 UNNORMALIZED planner trajectory to attack.
        :param z: (NA-B) x D latents for all non-planner agents (potential attackers)
        :param prior_out: tuple of (mean, var) each of size (NA-B) x D
        :param return_mins: returns indices of the current "most likely" of these
        :param attack_agt_idx: list of indices within each scene graph to use as the attacker. These should be global
                                    to the entire batched scene graph.
        '''
        NA = future_pred.size(0)
        B = tgt_traj.size(0)
        adv_crash_loss = min_dist = dist_traj = None
        cur_min_agt = cur_min_t = None
        if 'adv_crash' in self.loss_weights and self.loss_weights['adv_crash'] > 0.0:
            # minimize POSITIONAL distance
            attacker_pred = future_pred[~self.ego_mask][:, self.crash_min_t:, :] # (NA-B, T, 4)
            tgt_pred = tgt_traj[:, self.crash_min_t:, :4]
            tgt_expanded = torch.cat([tgt_pred[b:b+1, :, :].expand(self.graph_sizes[b]-1, -1, -1) for b in range(B)], dim=0)                
            dist_traj = torch.norm(attacker_pred[:,:,:2] - tgt_expanded[:,:,:2], dim=-1) # (NA-B, T)
            min_dist_in = dist_traj
            if self.crash_min_infront is not None:
                behind_steps = check_behind(attacker_pred.detach(), tgt_pred.detach(), self.ptr, self.crash_min_infront)
                behind_traj = torch.sum(behind_steps, dim=1, keepdim=True) == behind_steps.size(1)
                behind_traj = behind_traj.expand_as(behind_steps)
                if torch.sum(behind_traj.reshape((-1))) == behind_steps.size(0)*behind_traj.size(1):
                    # every agent is behind... just optim normally for now
                    behind_traj = torch.zeros_like(behind_traj).to(torch.bool)
                min_dist_in = torch.where(behind_traj, float('inf')*torch.ones_like(min_dist_in), min_dist_in) # set to 0 weight for agent behind the target

            if attack_agt_idx is not None:
                attack_agt_mask = torch.zeros(future_pred.size(0), dtype=torch.bool).to(attack_agt_idx.device)
                attack_agt_mask[attack_agt_idx] = True
                attack_agt_mask = attack_agt_mask[~self.ego_mask].unsqueeze(1).expand_as(min_dist_in)
                min_dist_in = torch.where(~attack_agt_mask, float('inf')*torch.ones_like(min_dist_in), min_dist_in) # set to 0 weight for agent behind the target                

            # soft min over all timesteps and agents
            NT = future_pred.size(1) - self.crash_min_t
            min_dist = [nn.functional.softmin(min_dist_in[self.nonego_ptr[b]:self.nonego_ptr[b+1]].view(-1), dim=0) for b in range(B)]
            # handle case where all frames for all agents are behind and therefore softmin is nan. (set to all 0 prob)
            min_dist = [bdist if torch.isnan(bdist[0]).item() == False else torch.zeros_like(bdist) for bdist in min_dist]

            cur_min_agt = [(torch.max(min_dist_b, dim=0)[1].item() // NT) + 1 for min_dist_b in min_dist]
            cur_min_t = [(torch.max(min_dist_b, dim=0)[1].item() % NT) + self.crash_min_t for min_dist_b in min_dist]

            min_dist = torch.cat(min_dist, dim=0)
            dist_traj = dist_traj.view(-1)**2 # (NA-B*T)
            weighted_adv_crash = min_dist * dist_traj # (NA-B*T)
            adv_crash_loss = [torch.sum(weighted_adv_crash[(self.nonego_ptr[b]*NT):(self.nonego_ptr[b+1]*NT)]) for b in range(B)]
            adv_crash_loss = torch.stack(adv_crash_loss)
            print(adv_crash_loss)

            print('Cur min agt: ' + str(cur_min_agt))
            print('Cur min t: ' + str(cur_min_t))

        # high weight for non attackers
        prior_reweight = min_dist.detach().reshape((future_pred.size(0)-B,  -1)) # NA-B x T
        prior_reweight = 1.0 - torch.sum(prior_reweight, dim=1)

        #
        # motion prior
        #
        motion_prior_loss = None
        if 'motion_prior' in self.loss_weights and self.loss_weights['motion_prior'] > 0.0:
            motion_prior_loss = self.motion_prior_loss(z, prior_out)
            prior_coeff = prior_reweight*self.loss_weights['motion_prior'] + \
                                (1.0 - prior_reweight)*self.loss_weights['motion_prior_atk']
            motion_prior_loss = motion_prior_loss * prior_coeff

        #
        # Regularizers
        #

        # interpolate trajectory to avoid cheating collision tests
        future_pred_interp = interp_traj(future_pred, scale_factor=3)

        # vehicle collision penalty
        veh_coll_loss_val = veh_coll_plan_loss_val = None
        if 'coll_veh' in self.loss_weights or 'coll_veh_plan' in self.loss_weights:
            if self.loss_weights['coll_veh'] > 0.0 or self.loss_weights['coll_veh_plan'] > 0.0:
                veh_coll_pens, veh_coll_mask = self.veh_coll_loss(future_pred_interp, return_raw=True) # only on non-target trajectories

            if self.loss_weights['coll_veh'] > 0.0:
                # need to mask out colls with ego
                non_ego_coll_mask = torch.ones((1, NA, NA), dtype=torch.bool).to(veh_coll_mask.device) # NA x NA
                for b in range(B):
                    non_ego_coll_mask[0, self.ptr[b], :] = False
                    non_ego_coll_mask[0, :, self.ptr[b]] = False
                non_ego_coll_mask = torch.logical_and(veh_coll_mask, non_ego_coll_mask.expand_as(veh_coll_mask))
                if torch.sum(non_ego_coll_mask) == 0:
                    veh_coll_loss_val = torch.Tensor([0.0]).to(veh_coll_pens.device)
                else:
                    veh_coll_loss_val = veh_coll_pens[non_ego_coll_mask]

            if self.loss_weights['coll_veh_plan'] > 0.0:
                # prior_reweight is size (NA-B, )
                # use prior reweight to down-weight possible attackers
                ego_coll_weight = torch.ones((NA)).to(veh_coll_pens.device)
                ego_coll_weight[~self.ego_mask] = prior_reweight
                ego_pen_mat = torch.ones((1, NA, NA)).to(veh_coll_pens.device)
                ego_coll_mask = torch.zeros((1, NA, NA), dtype=torch.bool).to(veh_coll_mask.device) # NA x NA
                for b in range(B):
                    # note this assumes diagonal will be thrown out.
                    ego_pen_mat[0, self.ptr[b], :] = ego_coll_weight
                    ego_pen_mat[0, :, self.ptr[b]] = ego_coll_weight
                    ego_coll_mask[0, self.ptr[b], :] = True
                    ego_coll_mask[0, :, self.ptr[b]] = True
                ego_coll_mask = torch.logical_and(veh_coll_mask, ego_coll_mask.expand_as(veh_coll_mask))
                # directly weight penalties
                veh_plan_weighted_pens = veh_coll_pens * ego_pen_mat.expand_as(veh_coll_pens)
                if torch.sum(ego_coll_mask) == 0:
                    veh_coll_plan_loss_val = torch.Tensor([0.0]).to(veh_coll_pens.device)
                else:
                    veh_coll_plan_loss_val = veh_plan_weighted_pens[ego_coll_mask]

        # env collision penalty
        env_coll_loss_val = None
        if 'coll_env' in self.loss_weights and self.loss_weights['coll_env'] > 0.0: 
            env_coll_loss_val = self.env_coll_loss(future_pred_interp[~self.ego_mask])
        # init loss
        init_loss = None
        if 'init_z' in self.loss_weights and self.loss_weights['init_z'] > 0.0:
            init_loss = torch.sum((self.init_z - z)**2, dim=1)
            # stay close to init for non-attacking vehicles
            init_z_coeff = prior_reweight*self.loss_weights['init_z'] + \
                                (1.0 - prior_reweight)*self.loss_weights['init_z_atk']
            # print(init_z_coeff)
            init_loss = torch.sum(init_loss * init_z_coeff)

        loss = 0.0
        loss_out = {}

        if init_loss is not None:
            # already applied weighting
            loss = loss + init_loss.mean()
            loss_out['init_loss'] = init_loss

        if motion_prior_loss is not None:
            # already applied weighting
            loss = loss + motion_prior_loss.mean() 
            loss_out['motion_prior_loss'] = motion_prior_loss

        if veh_coll_loss_val is not None:
            loss = loss + self.loss_weights['coll_veh']*veh_coll_loss_val.mean()
            loss_out['coll_veh_loss'] = veh_coll_loss_val

        if veh_coll_plan_loss_val is not None:
            loss = loss + self.loss_weights['coll_veh_plan']*veh_coll_plan_loss_val.mean()
            loss_out['coll_veh_plan_loss'] = veh_coll_plan_loss_val

        if env_coll_loss_val is not None:
            loss = loss + self.loss_weights['coll_env']*env_coll_loss_val.mean()
            loss_out['coll_env_loss'] = env_coll_loss_val

        if adv_crash_loss is not None:
            loss = loss + self.loss_weights['adv_crash']*adv_crash_loss.mean()
            loss_out['adv_crash_loss'] = adv_crash_loss

        loss_out['loss'] = loss         

        # output attacking agent and time if not set ahead of time
        if return_mins:
            if cur_min_agt is not None:
                loss_out['min_agt'] = np.array(cur_min_agt, dtype=np.int)
            if cur_min_t is not None:
                loss_out['min_t'] = np.array(cur_min_t, dtype=np.int)

        return loss_out

class AvoidCollLoss(nn.Module):
    '''
    Loss to discourage vehicle/environment collisions with high likelihood under prior.
    '''
    def __init__(self, loss_weights, veh_att, mapixes, map_env, init_z,
                    veh_coll_buffer=0.0,
                    single_veh_idx=None,
                    ptr=None):
        '''
        :param loss_weights: dict of weightings for loss terms
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param mapixes: map index corresponding to ALL agents (NA,)
        :param map_env: the map environment holding map info
        :param init_z:
        :param veh_coll_buffer: adds extra buffer around vehicles to penalize
        :param single_veh_idx: if not None, computes all losses w.r.t to ONE agent index in each batched scene graph.
                                i.e. if single_veh_idx = 0, only collisions involve agent 0
                                will be included in the computed loss. To use this, MUST also pass in ptr from the scene graph.
        '''
        super(AvoidCollLoss, self).__init__()
        self.loss_weights = loss_weights
        self.init_z = init_z
        self.single_veh_idx = single_veh_idx
        self.ptr = ptr            
        self.use_single_agt = self.single_veh_idx is not None
        self.motion_prior_loss = MotionPriorLoss()
        self.veh_coll_loss = VehCollLoss(veh_att,
                                          buffer_dist=veh_coll_buffer,
                                          single_veh_idx=self.single_veh_idx,
                                          ptr=self.ptr)
        if self.use_single_agt:
            assert(self.ptr is not None)
            self.single_mask = torch.zeros((veh_att.size(0)), dtype=torch.bool).to(veh_att.device)
            single_inds = self.ptr[:-1] + self.single_veh_idx
            self.single_mask[single_inds] = True
            veh_att = veh_att[self.single_mask]
            mapixes = mapixes[self.single_mask]
        self.env_coll_loss = EnvCollLoss(veh_att, mapixes, map_env)

    def forward(self, future_pred, z, prior_out):
        ''' 
        IF single_veh_idx is not None, z and prior_out should be (B, D) rather than (NA, D)

        :param future_pred: NA x T x 4
        '''        
        loss = 0.0
        loss_out = {}

        future_pred_interp = interp_traj(future_pred, scale_factor=3)

        if self.loss_weights['coll_veh'] > 0.0:
            # vehicle collision penalty
            veh_coll_loss_val = self.veh_coll_loss(future_pred_interp) # num colliding pairs
            loss = loss + self.loss_weights['coll_veh']*veh_coll_loss_val.mean()
            loss_out['coll_veh_loss'] = veh_coll_loss_val

        if self.loss_weights['coll_env'] > 0.0:
            # env collision penalty
            env_coll_input = future_pred_interp if not self.use_single_agt else future_pred_interp[self.single_mask]
            env_coll_loss_val = self.env_coll_loss(env_coll_input)
            loss = loss + self.loss_weights['coll_env']*env_coll_loss_val.mean()
            loss_out['coll_env_loss'] = env_coll_loss_val

        if self.loss_weights['motion_prior'] > 0.0:
            # motion prior
            motion_prior_loss = self.motion_prior_loss(z, prior_out)
            loss = loss + self.loss_weights['motion_prior']*motion_prior_loss.mean()
            loss_out['motion_prior_loss'] = motion_prior_loss

        if self.loss_weights['init_z'] > 0.0:
            # init loss
            init_loss = torch.sum((self.init_z - z)**2, dim=1)
            loss = loss + self.loss_weights['init_z']*init_loss.mean()
            loss_out['init_loss'] = init_loss

        loss_out['loss'] = loss

        return loss_out

class MotionPriorLoss(nn.Module):
    '''
    Measures negative log-likelihood of latent z under the given prior.
    '''
    def __init__(self):
        super(MotionPriorLoss, self).__init__()

    def forward(self, z, prior_out):
        '''
        :param z: current latent vec (NA x z_dim) or (NA x NS x z_dim)
        :param prior_out: output of prior which contains:
            :prior_mean: mean of the prior output (NA x z_dim)
            :prior_var: variance of the prior output (NA x z_dim)
        :return: log-likelihood of cur_z under prior for all agents NA
        '''
        prior_mean = prior_out[0]
        prior_var = prior_out[1]
        if len(z.size()) == 3:
            # then it's NA x NS x z_dim
            prior_mean = prior_mean.unsqueeze(1)
            prior_var = prior_var.unsqueeze(1)
        return -log_normal(z, prior_mean, prior_var)

class EnvCollLoss(nn.Module):
    def __init__(self, veh_att, mapixes, map_env):
        super(EnvCollLoss, self).__init__()
        self.map_env = map_env
        self.mapixes = mapixes
        self.penalty_dists =  torch.sqrt((veh_att[:, 0]**2 / 4.0) + (veh_att[:, 1]**2 / 4.0))
        self.veh_att = veh_att

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss
        '''
        NA, T, _ = traj.size()
        traj = traj.view(NA*T, 4)
        cur_att = self.veh_att.view(NA, 1, 2).expand(NA, T, 2).reshape(NA*T, 2)

        # get collisions w/ non-drivable (first layer)
        drivable_raster = self.map_env.nusc_raster[:, 0]
        cur_mapixes = self.mapixes.view(NA, 1).expand(NA, T).reshape(NA*T)
        coll_pt = nutils.get_coll_point(drivable_raster,
                                        self.map_env.nusc_dx,
                                        traj.detach(),
                                        cur_att,
                                        cur_mapixes)

        if torch.sum(torch.isnan(coll_pt)) == NA*T*2:
            return torch.Tensor([0.0]).to(traj.device)

        # compute penalties
        valid = ~torch.isnan(torch.sum(coll_pt, axis=1))
        traj_cent = traj[:,:2][valid]
        cur_dists = torch.norm(traj_cent - coll_pt[valid], dim=1)
        cur_pen_dists = self.penalty_dists.view(NA, 1).expand(NA, T).reshape(NA*T)[valid]
        cur_penalties = 1.0 - (cur_dists / cur_pen_dists)

        return cur_penalties

class VehCollLoss(nn.Module):
    '''
    Penalizes collision between vehicles with circle approximation.
    '''
    def __init__(self, veh_att,
                       num_circ=5,
                       buffer_dist=0.0,
                       single_veh_idx=None,
                       ptr=None
                       ):
        '''
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param num_circ: number of circles used to approximate each vehicle.
        :param buffer: extra buffer distance that circles must be apart to avoid being penalized
        :param single_veh_idx: only computes loss w.r.t a single vehicle at the given index within each scene graph.
                                If given must also include.ptr from scene graph as input even if only a single batch.
        :param ptr: if given, treats the traj input as a batch of scene graphs and only computes collisions between
                    vehicles in the same batch. if None, assumes a single batch
        '''
        super(VehCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.single_veh_idx = single_veh_idx
        self.ptr = ptr

        NA = self.veh_att.size(0)
        # construct centroids circles of each agent
        self.veh_rad = self.veh_att[:, 1] / 2. # radius of the discs for each vehicle assuming length > width
        cent_min = -(self.veh_att[:, 0] / 2.) + self.veh_rad
        cent_max = (self.veh_att[:, 0] / 2.) - self.veh_rad
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_circ) for vidx in range(NA)], dim=0).to(veh_att.device)
        # create dummy states for centroids with y=0 and hx,hy=1,0 so can transform later
        self.centroids = torch.stack([cent_x, torch.zeros_like(cent_x), torch.ones_like(cent_x), torch.zeros_like(cent_x)], dim=2)
        self.num_circ = num_circ
        # minimum distance that two vehicle circle centers can be apart without collision
        self.penalty_dists = self.veh_rad.view(NA, 1).expand(NA, NA) + self.veh_rad.view(1, NA).expand(NA, NA) + self.buffer_dist
        # need a mask to ignore self-collisions when computing
        self.off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        if self.ptr is None:
            # assume single batch
            self.ptr = torch.Tensor([0, NA]).to(self.veh_att.device).to(torch.long)
        # ignore collisiosn with other scene graphs in the batch
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True
        self.off_diag_mask = torch.logical_and(self.off_diag_mask, batch_mask)

        if self.single_veh_idx is not None:
            # then only want to use penalties associated with a single agent
            single_mask = torch.zeros((NA), dtype=torch.bool).to(self.off_diag_mask.device)
            single_inds = self.ptr[:-1] + self.single_veh_idx
            single_mask[single_inds] = True # target is always at index 0 of each scene graph
            single_mask_row = single_mask.view(1, NA).expand(NA, NA)
            single_mask_col = single_mask.view(NA, 1).expand(NA, NA)
            single_mask = torch.logical_or(single_mask_row, single_mask_col)
            self.off_diag_mask = torch.logical_and(self.off_diag_mask, single_mask)


    def forward(self, traj, att_inds=None, return_raw=False):
        '''
        :param traj: (N x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :param att_inds: list [N] if not using the full list of agents that was used to init the loss, which indices are you using
        :param return_raw: if True, returns the (T x N x N) penalty cost matrix and (T x N x N) mask where each True entry is a
                            colliding pair that is valid (according to settings)
        :return: loss
        '''
        NA, T, _ = traj.size()

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_centroids = self.centroids
        if att_inds is not None:
            cur_centroids = cur_centroids[att_inds]
        cur_cent = cur_centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] # only need centers
        
        world_cent = world_cent.transpose(0, 1) # T x NA X C x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)

        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)   
        cur_penalty_dists = self.penalty_dists
        if att_inds is not None:
            cur_penalty_dists = cur_penalty_dists[att_inds][:, att_inds]
        cur_penalty_dists = cur_penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        cur_off_diag_mask = self.off_diag_mask
        if att_inds is not None:
            cur_off_diag_mask = cur_off_diag_mask[att_inds][:, att_inds]
            # print(cur_off_diag_mask)
        is_colliding_mask = torch.logical_and(is_colliding_mask, cur_off_diag_mask.view(1, NA, NA))
        if not return_raw and torch.sum(is_colliding_mask) == 0:
            return torch.Tensor([0.0]).to(traj.device)
        # compute penalties
        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (min_pair_dists / cur_penalty_dists)

        if return_raw:
            return cur_penalties, is_colliding_mask
        else:
            cur_penalties = cur_penalties[is_colliding_mask]
            return cur_penalties    

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)

def check_single_veh_coll(traj_tgt, lw_tgt, traj_others, lw_others):
    '''
    Checks if the target trajectory collides with each of the given other trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED. Handles nan frames in traj_others by simply skipping.

    :param traj_tgt: (T x 4)
    :param lw_tgt: (2, )
    :param traj_others: (N x T x 4)
    :param lw_others: (N x 2)

    :returns veh_coll: (N)
    :returns coll_time: (N)
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    NA, FT, _ = traj_others.size()
    traj_tgt = traj_tgt.cpu().numpy()
    lw_tgt = lw_tgt.cpu().numpy()
    traj_others = traj_others.cpu().numpy()
    lw_others = lw_others.cpu().numpy()

    veh_coll = np.zeros((NA), dtype=np.bool)
    coll_time = np.ones((NA), dtype=np.int)*FT
    poly_cache = dict() # for the tgt polygons since used many times
    for aj in range(NA):
        for t in range(FT):
            # compute iou
            if t not in poly_cache:
                ai_state = traj_tgt[t, :]
                ai_corners = nutils.get_corners(ai_state, lw_tgt)
                ai_poly = Polygon(ai_corners)
                poly_cache[t] = ai_poly
            else:
                ai_poly = poly_cache[t]

            aj_state = traj_others[aj, t, :]
            if np.sum(np.isnan(aj_state)) > 0:
                continue
            aj_corners = nutils.get_corners(aj_state, lw_others[aj])
            aj_poly = Polygon(aj_corners)
            cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
            if cur_iou > VEH_COLL_THRESH:
                veh_coll[aj] = True
                coll_time[aj] = t
                break # don't need to check rest of sequence

    return veh_coll, coll_time

def check_pairwise_veh_coll(traj, lw):
    '''
    Computes collision rate for all pairs of given trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED.

    :param traj: (N x T x 4)
    :param lw: (N x 2)

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    NA, FT, _ = traj.size()
    traj = traj.cpu().numpy()
    lw = lw.cpu().numpy()

    veh_coll = np.zeros((NA), dtype=np.bool)
    poly_cache = dict()    
    # loop over every timestep in every sample for this combination
    coll_count = 0
    for ai in range(NA):
        for aj in range(ai+1, NA): # don't double count
            if veh_coll[ai]:
                break # already found a collision, move on.
            for t in range(FT):
                # compute iou
                if (ai, t) not in poly_cache:
                    ai_state = traj[ai, t, :]
                    ai_corners = nutils.get_corners(ai_state, lw[ai])
                    ai_poly = Polygon(ai_corners)
                    poly_cache[(ai, t)] = ai_poly
                else:
                    ai_poly = poly_cache[(ai, t)]

                if (aj, t) not in poly_cache:
                    aj_state = traj[aj, t, :]
                    aj_corners = nutils.get_corners(aj_state, lw[aj])
                    aj_poly = Polygon(aj_corners)
                    poly_cache[(aj, t)] = aj_poly
                else:
                    aj_poly = poly_cache[(aj, t)]

                cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
                if cur_iou > VEH_COLL_THRESH:
                    coll_count += 1
                    veh_coll[ai] = True
                    break # don't need to check rest of sequence

    coll_dict = {
        'num_coll_veh' : float(coll_count),
        'num_traj_veh' : float(NA),
        'did_collide' : veh_coll
    }

    return coll_dict

def interp_traj(future_pred, scale_factor=3):
    '''
    :param future_pred (NA x T x 4) or (NA x NS x T x 4)
    '''
    mult_samp = False
    if len(future_pred.size()) == 4:
        mult_samp = True
        NA, NS, T, _ = future_pred.size()
        future_pred = future_pred.reshape(NA*NS, T, 4)
    future_pred_interp = nn.functional.interpolate(future_pred.transpose(1,2),
                                                        scale_factor=scale_factor,
                                                        mode='linear').transpose(1,2)
    # normalize heading (NOTE: interp hx and hy is not exactly correct, but close)
    interp_pos = future_pred_interp[:, :, :2]
    interp_h = future_pred_interp[:, :, 2:4]
    interp_h = interp_h / torch.norm(interp_h, dim=-1, keepdim=True)
    future_pred_interp = torch.cat([interp_pos, interp_h], dim=-1)
    if mult_samp:
        future_pred_interp = future_pred_interp.reshape(NA, NS, future_pred_interp.size(1), 4)
    return future_pred_interp

def check_behind(attacker_fut, tgt_fut, ptr, crash_min_infront):
    '''
    checks if each attacker is behind the target at each time step.

    :param attacker_fut: future for all attackers (NA-B, T, 4)
    :param tgt_fut: future for targets (B, T, 4)
    :param ptr: graph ptr to start of each batch
    :param crash_min_infront: threshold to determine if behind

    :return behind_steps: (NA-B, T) True if attacker currently behind tgt
    '''
    graph_sizes = ptr[1:] - ptr[:-1]
    B = graph_sizes.size(0)
    attacker_pred = attacker_fut
    tgt_expanded = torch.cat([tgt_fut[b:b+1, :, :].expand(graph_sizes[b]-1, -1, -1) for b in range(B)], dim=0)

    tgt_h = tgt_expanded[:, :, 2:4]
    tgt_pos = tgt_expanded[:, :, :2]
    atk_pos = attacker_pred[:, :, :2]

    tgt2atk = atk_pos - tgt_pos
    tgt2atk = tgt2atk / torch.norm(tgt2atk, dim=-1, keepdim=True)
    cossim = torch.sum(tgt2atk * tgt_h, dim=-1) # (NA-B, T)

    # determine for each agent if:
    #   - behind and stays behind
    behind_steps = cossim < crash_min_infront
    return behind_steps
