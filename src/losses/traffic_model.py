# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import time

import torch
from torch import nn

import numpy as np

from losses.common import kl_normal, log_normal
from utils.transforms import transform2frame
from utils.torch import c2c
import datasets.nuscenes_utils as nutils

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)

class TrafficModelLoss(nn.Module):
    def __init__(self, loss_weights,
                    state_normalizer=None,
                    att_normalizer=None):
        '''
        :param loss_weights: dict of weightings for loss terms
        :param state_normalizer: normalization object for kinematic state
        :param att_normalizer: normalization object for length/width
        '''
        super(TrafficModelLoss, self).__init__()
        self.loss_weights = loss_weights
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer

    def forward(self, scene_graph, pred,
                 map_idx=None,
                 map_env=None):
        '''
        Computes loss.

        :param scene_graph: containing input and GT data
        :param pred: dict of model predictions.
        :param map_idx, map_env: only needed for env collision losses
        '''        

        # we have a different number of agents for every sequence, and 
        #       a different number of timesteps for each agent even within
        #       the same sequence, so we mean over all timesteps so that
        #       the weights can be reliably balanced

        # reconstruction loss
        gt_future = scene_graph.future_gt # NA x FT x 6
        pred_future = pred['future_pred'] # NA x FT x 4

        # only want to compute loss for timesteps we have GT data
        gt_future = gt_future[scene_graph.future_vis == 1.0]
        pred_future = pred_future[scene_graph.future_vis == 1.0]

        # assume variance is 1 (i.e. MSE loss with extra constant)
        recon_loss = -log_normal(pred_future, gt_future[:, :4], torch.ones_like(pred_future))

        # KL divergence loss
        pm, pv = pred['prior_out'] # NA x z_size
        qm, qv = pred['posterior_out']
        # kl_normal(qm, qv, pm, pv)
        kl_loss = kl_normal(qm, qv, pm, pv)

        # total weighted loss
        loss = self.loss_weights['recon']*recon_loss.mean() + self.loss_weights['kl']*kl_loss.mean()

        prior_coll_loss = None
        if self.loss_weights['coll_veh_prior'] > 0.0:
            # compute veh2veh collision
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collisison loss!')
                exit()
            # unnormalize
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            # build the loss function
            veh_coll_loss = VehCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
            # compute for each desired
            if self.loss_weights['coll_veh_prior'] > 0.0 and 'future_samp' in pred:
                prior_traj = self.state_normalizer.unnormalize(pred['future_samp'])
                prior_coll_pens, na_sqr = veh_coll_loss(prior_traj)
                prior_coll_loss = torch.sum(prior_coll_pens) / na_sqr
                loss = loss + self.loss_weights['coll_veh_prior']*prior_coll_loss

        prior_coll_env_loss = None
        if self.loss_weights['coll_env_prior'] > 0.0:
            assert(map_idx is not None and map_env is not None)
            # compute veh2env collision
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collisison loss!')
                exit()

            # only compute these losses on ego vehicles since guaranteed should have no collisions
            ego_inds = scene_graph.ptr[:-1]
            # unnormalize
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw[ego_inds])
            # build the loss function
            env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, pred['future_pred'].size(1))
            # compute for each desired
            if self.loss_weights['coll_env_prior'] > 0.0 and 'future_samp' in pred:
                prior_traj = self.state_normalizer.unnormalize(pred['future_samp'][ego_inds])
                prior_coll_env_loss = env_coll_loss(prior_traj)
                loss = loss + self.loss_weights['coll_env_prior']*prior_coll_env_loss.mean()

        loss_out = {
            'loss' : loss.view((1,)),
            'recon_loss' : recon_loss, # (num_valid_frames, )
            'kl_loss' : kl_loss # (NA, )
        }

        if prior_coll_loss is not None:
            loss_out['coll_veh_prior'] = prior_coll_loss.view((1,))
        if prior_coll_env_loss is not None:
            loss_out['coll_env_prior'] = prior_coll_env_loss.view(-1) # (B, T)

        return loss_out

    def compute_err(self, scene_graph, pred, normalizer):
        '''
        Computes interpretable position and angle errors.

        pos_err is dist from GT averaged over all all timesteps for each future pred
        ang_err is angle diff (absolute degrees) from GT averaged over all timesteps for each future pred
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        pred_future = pred['future_pred'] # NA x FT x 4

        NA, FT, _ = gt_future.size()

        gt_future = normalizer.unnormalize(gt_future)
        pred_future = normalizer.unnormalize(pred_future)

        # only want to compute errors for timesteps we have GT data
        gt_future = gt_future[scene_graph.future_vis == 1.0]
        pred_future = pred_future[scene_graph.future_vis == 1.0]

        # positional distance error
        gt_pos = gt_future[:,:2]
        pred_pos = pred_future[:,:2]
        pos_err = torch.norm(gt_pos - pred_pos, dim=-1)
        # angle distance error
        gt_h = gt_future[:,2:4]
        gt_h = gt_h / torch.norm(gt_h, dim=-1, keepdim=True)
        pred_h = pred_future[:,2:4]
        pred_h = pred_h / torch.norm(pred_h, dim=-1, keepdim=True)
        dotprod = torch.sum(gt_h * pred_h, dim=-1).clamp(-1, 1)
        ang_diff = torch.acos(dotprod)
        ang_err = torch.rad2deg(ang_diff)

        # NLL of posterior mean under the prior
        post_mean = pred['posterior_out'][0]
        z_logprob = log_normal(post_mean, pred['prior_out'][0], pred['prior_out'][1])
        z_mdist =  torch.norm((post_mean - pred['prior_out'][0]) / torch.sqrt(pred['prior_out'][1]), dim=-1)

        err_out = {
            'pos_err' : pos_err, # (num_valid_frames, )
            'ang_err' : ang_err, # (num_valid_frames, )
            'z_logprob' : z_logprob, # NA
            'z_mdist' : z_mdist
        }

        return err_out

class VehCollLoss(nn.Module):
    '''
    Penalizes collision between vehicles with circle approximation.
    '''
    def __init__(self, veh_att, batch, ptr,
                       num_circ=5,
                       buffer_dist=0.0):
        '''
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param batch: from the scene graph
        :param ptr: from the scene_graph
        :param num_circ: number of circles used to approximate each vehicle.
        :param buffer: extra buffer distance that circles must be apart to avoid being penalized
        '''
        super(VehCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.batch = batch
        self.ptr = ptr

        self.graph_sizes = self.ptr[1:] - self.ptr[:-1]
        self.num_pairs = torch.sum(self.graph_sizes*self.graph_sizes - self.graph_sizes)

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
        # need a mask to ignore "self" collisions and "collisions" from other scene graphs in the batch
        off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True

        self.valid_mask = torch.logical_and(off_diag_mask, batch_mask)

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss, number of "interactions" that could have caused a collision, i.e. number of valid vehicle pairs
        '''
        NA, T, _ = traj.size()
        cur_valid_mask = self.valid_mask.view(1, NA, NA).expand(T, NA, NA)

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_cent = self.centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] # only need centers
        world_cent = world_cent.transpose(0, 1) # T x NA X C x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)

        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)
        cur_penalty_dists = self.penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        is_colliding_mask = torch.logical_and(is_colliding_mask,cur_valid_mask)
        # compute penalties
        cur_penalties = torch.where(is_colliding_mask, 1.0 - (min_pair_dists / cur_penalty_dists), torch.zeros_like(cur_penalty_dists))
        cur_penalties = cur_penalties[cur_valid_mask]

        return cur_penalties, self.num_pairs

class EnvCollLoss(nn.Module):
    '''
    Penalizes overlap with non-drivable area.
    '''
    def __init__(self, veh_att, mapixes, map_env, T):
        '''
        :param veh_att: (NA, 2) UNNORMALIZED
        :param mapixes: (NA, )
        :param map_env: 
        :param T: number of steps in trajectories that loss will be computed on
        '''
        super(EnvCollLoss, self).__init__()
        self.map_env = map_env
        # loss will be applied on all timesteps, so update info accordingly
        NA = veh_att.size(0)
        assert(NA == mapixes.size(0))
        self.mapixes = mapixes.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.penalty_dists = torch.sqrt((veh_att[:, 0]**2 / 4.0) + (veh_att[:, 1]**2 / 4.0)) # max dist from center to corner of each vehicle
        self.penalty_dists = self.penalty_dists.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.veh_att = veh_att.view(NA, 1, 2).expand(NA, T, 2).reshape(NA*T, 2)
        self.T = T
        

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss
        '''
        NA = traj.size(0)
        T = self.T
        assert(T == traj.size(1))
        assert(NA*T == self.veh_att.size(0))
        traj = traj.view(NA*T, 4)

        all_penalties = torch.zeros((NA*T)).to(traj.device)
        # get collisions w/ non-drivable (first layer)
        drivable_raster = self.map_env.nusc_raster[:, 0]
        coll_pt = nutils.get_coll_point(drivable_raster,
                                        self.map_env.nusc_dx,
                                        traj.detach(),
                                        self.veh_att,
                                        self.mapixes)
        valid = ~torch.isnan(torch.sum(coll_pt, axis=1))
        if torch.sum(valid) == 0:
            return all_penalties.view(NA, T)

        # compute penalties
        traj_cent = traj[:,:2][valid]
        cur_dists = torch.norm(traj_cent - coll_pt[valid], dim=1)
        cur_pen_dists = self.penalty_dists[valid]
        val_penalties = 1.0 - (cur_dists / cur_pen_dists)
        # return a penalty for every time step, if not colliding is just 0
        all_penalties[valid] = val_penalties

        return all_penalties.view(NA, T)
    
def compute_disp_err(scene_graph, pred, normalizer):
    '''
    Computes sample-based displacement errors.

    ONLY computes for ego vehicle since have guaranteed full past/future motion so the statistics
    will be correct.
    '''
    gt_future = scene_graph.future_gt # NA x FT x 6
    pred_future = pred['future_pred'] # NA x NS x FT x 4

    NA, FT, _ = gt_future.size()
    NS = pred_future.size(1)

    # make sure same length
    FT = pred_future.size(2) if pred_future.size(2) < FT else FT
    pred_future = pred_future[:, :, :FT] # if prediction is longer, make sure only compare the steps we have
    gt_future = gt_future[:, :FT]

    gt_future = normalizer.unnormalize(gt_future).view(NA, 1, FT, 6)
    pred_future = normalizer.unnormalize(pred_future)

    # find index of first agent in each batch
    ego_inds = scene_graph.ptr[:-1]

    # ego-only data
    gt_future = gt_future[ego_inds] # B x 1 x FT x 6
    pred_future = pred_future[ego_inds] # B x NS x FT x 4
    B = gt_future.size(0)

    # positional ADE
    gt_pos = gt_future[:,:,:,:2]
    pred_pos = pred_future[:,:,:,:2]
    diff = torch.norm(gt_pos - pred_pos, dim=-1) # B x NS x FT
    ade = diff.mean(dim=-1) # B x NS
    min_ade = torch.min(ade, dim=1)[0] # B

    # positional APD
    pred_pairwise_pos = pred_pos.view(B, NS, 1, FT, 2).expand(B, NS, NS, FT, 2)
    pairwise_diff = torch.norm(pred_pairwise_pos - pred_pairwise_pos.transpose(1, 2), dim=-1) # B x NS x NS x FT
    all_sum = torch.sum(pairwise_diff, dim=[1, 2]).sum(dim=-1)
    apd = all_sum / (NS*(NS-1)*FT) # don't want to include diagonal elmnts

    # positional FDE
    fde = diff[:,:,-1]
    min_fde = torch.min(fde, dim=1)[0] # B 

    # angular ADE
    gt_h = gt_future[:,:,:,2:4]
    gt_h = gt_h / torch.norm(gt_h, dim=-1, keepdim=True)
    pred_h = pred_future[:,:,:,2:4]
    pred_h = pred_h / torch.norm(pred_h, dim=-1, keepdim=True)
    dotprod = torch.sum(gt_h * pred_h, dim=-1).clamp(-1, 1)
    ang_diff = torch.rad2deg(torch.acos(dotprod)) # B x NS x FT
    ang_ade = ang_diff.mean(dim=-1)
    ang_min_ade = torch.min(ang_ade, dim=1)[0] # B

    # angular FDE
    ang_fde = ang_diff[:,:,-1]
    ang_min_fde = torch.min(ang_fde, dim=1)[0]

    disp_err_dict = {
        'pos_minADE' : min_ade,
        'pos_minFDE' : min_fde,
        'ang_minADE' : ang_min_ade,
        'ang_minFDE' : ang_min_fde,
        'APD' : apd
    }
    return disp_err_dict

def compute_coll_rate_env(scene_graph, map_idx, pred, map_env, state_normalizer, att_normalizer,
                        ego_only=False):
    '''
    Computes number of rollouts that collided with map for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from datasets.utils import get_ego_inds

    if isinstance(pred, torch.Tensor):
        pred_future = pred
    else:
        pred_future = pred['future_pred'] # NA x NS x FT x 4
    NA, NS, FT, _ = pred_future.size()

    veh_att = scene_graph.lw
    mapixes = map_idx[scene_graph.batch]

    if ego_only:
        ego_inds = get_ego_inds(scene_graph)
        pred_future = pred_future[ego_inds]
        veh_att = veh_att[ego_inds]
        mapixes = mapixes[ego_inds]
        NA = pred_future.size(0)

    # unnorm preds and attribs
    pred_future = state_normalizer.unnormalize(pred_future).view(NA*NS*FT, 4)
    veh_att = att_normalizer.unnormalize(veh_att).view(NA, 1, 1, 2).expand(NA, NS, FT, 2).reshape(NA*NS*FT, 2)

    # check if on drivable (first layer)
    drivable_raster = map_env.nusc_raster[:, 0]
    mapixes = mapixes.view(NA, 1, 1).expand(NA, NS, FT).reshape(NA*NS*FT)
    # don't check for nan futures
    valid_frames = ~torch.isnan(pred_future.sum(-1))
    drivable_frac = nutils.check_on_layer(drivable_raster,
                                            map_env.nusc_dx,
                                            pred_future[valid_frames],
                                            veh_att[valid_frames],
                                            mapixes[valid_frames])
    final_drivable_frac = torch.ones(NA*NS*FT).to(drivable_frac) # by default, nan values are not considered colliding
    final_drivable_frac[valid_frames] = drivable_frac
    final_drivable_frac = final_drivable_frac.view(NA, NS, FT)
    coll_frame = (final_drivable_frac < (1.0 - ENV_COLL_THRESH))
    map_coll = torch.sum(coll_frame, dim=2) >= 1 # (NA, NS)

    coll_dict = {
        'num_coll_map' : float(c2c(torch.sum(map_coll))),
        'num_traj_map' : float(NS*NA),
        'did_collide' : map_coll
    }

    return coll_dict

def compute_coll_rate_env_from_traj(pred_future, veh_att, mapixes, map_env):
    '''
    Computes number of rollouts that collided with map for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    :param pred_future: NA x NS x FT x 4 UNNORMALIZED
    :param veh_att:
    :param mapixes: (NA) index of the map for each agent
    :param map_env:

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils

    NA, NS, FT, _ = pred_future.size()

    # unnorm preds and attribs
    pred_future = pred_future.reshape(NA*NS*FT, 4)
    veh_att = veh_att.reshape(NA, 1, 1, 2).expand(NA, NS, FT, 2).reshape(NA*NS*FT, 2)

    # check if on drivable (first layer)
    drivable_raster = map_env.nusc_raster[:, 0]
    mapixes = mapixes.reshape(NA, 1, 1).expand(NA, NS, FT).reshape(NA*NS*FT)
    # don't check for nan futures
    valid_frames = ~torch.isnan(pred_future.sum(-1))
    drivable_frac = nutils.check_on_layer(drivable_raster,
                                            map_env.nusc_dx,
                                            pred_future[valid_frames],
                                            veh_att[valid_frames],
                                            mapixes[valid_frames])
    final_drivable_frac = torch.ones(NA*NS*FT).to(drivable_frac) # by default, nan values are not considered colliding
    final_drivable_frac[valid_frames] = drivable_frac
    final_drivable_frac = final_drivable_frac.view(NA, NS, FT)
    coll_frame = (final_drivable_frac < (1.0 - ENV_COLL_THRESH))
    map_coll = torch.sum(coll_frame, dim=2) >= 1 # (NA, NS)

    coll_dict = {
        'num_coll_map' : float(c2c(torch.sum(map_coll))),
        'num_traj_map' : float(NS*NA),
        'did_collide' : map_coll
    }

    return coll_dict

def compute_coll_rate_veh(scene_graph, pred, state_normalizer, att_normalizer):
    '''
    Computes number of rollouts that collided with other agents for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    WARNING: this function assumes the scene graph edges connect all vehicle pairs that
    need to be checked for collisions. Also assumes all edges are bidirectional, i.e.
    if (3, 5) is a pair then (5, 3) is also one. We ONLY check one of the two.

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    if isinstance(pred, torch.Tensor):
        pred_future = pred
    else:
        pred_future = pred['future_pred'] # NA x NS x FT x 4
    NA, NS, FT, _ = pred_future.size()

    veh_att = scene_graph.lw

    # unnorm preds and attribs
    pred_future = state_normalizer.unnormalize(pred_future)
    veh_att = att_normalizer.unnormalize(veh_att)

    # all the vehicle pairs to go over
    pred_future = pred_future.cpu().numpy()
    veh_att = veh_att.cpu().numpy()
    pairs = scene_graph.edge_index.cpu().numpy().T

    veh_coll = np.zeros((NA, NS), dtype=np.bool)
    poly_cache = dict()    
    # loop over every timestep in every sample for this combination
    coll_count = 0
    for s in range(NS):
        for veh_pair in pairs:
            aj, ai = veh_pair
            if aj <= ai:
                continue # don't double count
            if veh_coll[ai, s]:
                continue # already determined there has been a collision for this agent at this sample, move on
            for t in range(FT):
                # compute iou
                if (ai, s, t) not in poly_cache:
                    ai_state = pred_future[ai, s, t, :]
                    if np.sum(np.isnan(ai_state)) > 0:
                        poly_cache[(ai, s, t)] = None
                        continue # don't have data for this step
                    ai_corners = nutils.get_corners(ai_state, veh_att[ai])
                    ai_poly = Polygon(ai_corners)
                    poly_cache[(ai, s, t)] = ai_poly
                else:
                    ai_poly = poly_cache[(ai, s, t)]
                    if ai_poly is None:
                        continue
                if (aj, s, t) not in poly_cache:
                    aj_state = pred_future[aj, s, t, :]
                    if np.sum(np.isnan(aj_state)) > 0:
                        poly_cache[(aj, s, t)] = None
                        continue # don't have data for this step
                    aj_corners = nutils.get_corners(aj_state, veh_att[aj])
                    aj_poly = Polygon(aj_corners)
                    poly_cache[(aj, s, t)] = aj_poly
                else:
                    aj_poly = poly_cache[(aj, s, t)]
                    if aj_poly is None:
                        continue
                cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
                if cur_iou > VEH_COLL_THRESH:
                    coll_count += 1
                    veh_coll[ai, s] = True
                    break # don't need to check rest of sequence

    coll_dict = {
        'num_coll_veh' : float(coll_count),
        'num_traj_veh' : float(NS*NA),
        'did_collide' : veh_coll
    }

    return coll_dict