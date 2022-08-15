# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

from datasets import nuscenes_utils as nutils
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err

import datasets.nuscenes_utils as nutils

def detach_embed_info(embed_info_attached):
    embed_info = dict()
    for k, v in embed_info_attached.items():
        if isinstance(v, torch.Tensor):
            # everything else
            embed_info[k] = v.detach()
        elif isinstance(v, tuple):
            # posterior or prior output
            embed_info[k] = (v[0].detach(), v[1].detach())
    return embed_info

def determine_feasibility_nusc(samples, normalizer, feasibility_thresh, 
                                feasibility_time=0, feasibility_vel=0.0,
                                feasibility_infront_min=None,
                                check_non_drivable_separation=True,
                                map_env=None,
                                map_idx=None):
    '''
    Determine whether the given sequences are plausible to seed scenario generation
    by measuring distance between ego and other agents at each step of each sample. 

    NOTE: This assumes the samples are of a SINGLE scene graph, i.e. idx 0 is the ego trajectory
    and all other indices are other agents in the same scene.

    :param samples: NORMALIZED samples from the model for the future (NA x NS x FT x 4)
    :param feasibility_thresh: if agents are ever less than this distance (meters) a part in
                                any sampled timestep, it's considered to be a feasible "seed" past.
    :param feasibility_time: must be >= feasibility time to be considered feasible. 
    :param feasibility_vel: maximum velocity of sampled trajectory for an agent must be >= this thresh
                            to be considered feasible.
    :param feasibility_infront_min: in [-1, 1]. if not None, dot product between ego heading and vector from ego to
                                    potential attacker must be >= this threshold in order to be condiered
                                    feasible. i.e encourage attacker to be in front of ego.
    :param check_non_drivable_separation: if true, only feasible if samples of ego/others that result in
                                            minimum distance are not separated by non-drivable area.
    :param map_env: env
    :param map_idx: (1, )

    :return feasible: (NA-1, ) True if the agent comes within the feasibility thresh, False otherwise
    :return feasible_time_step: (NA-1, ) the timestep index at which each agent comes closest to the ego.
    :return feasible: (NA-1, ) how far agent is from planner.
    '''
    if samples.size(0) == 1:
        # we only have ego, so it's not feasible
        return None, None, None
    samples = normalizer.unnormalize(samples)
    ego_samples = samples[0:1, :, :, :]
    agent_samples = samples[1:, :, :, :]
    NA, NS, FT, _ = agent_samples.size()
    ego_agent_dists = torch.norm(ego_samples[:,:,:,:2] - agent_samples[:,:,:,:2], dim=-1) # (NA-1, NS, FT)
    ego_agent_dists = ego_agent_dists[:, :, feasibility_time:]

    if feasibility_infront_min is not None:
        # mask out steps where agents are behind ego up to some threshold
        assert(feasibility_infront_min >= -1)
        assert(feasibility_infront_min <= 1)
        ego_h = ego_samples[:, :, feasibility_time:, 2:4]
        ego_pos = ego_samples[:, :, feasibility_time:, :2]
        agent_pos = agent_samples[:, :, feasibility_time:, :2]

        ego2agent = agent_pos - ego_pos
        ego2agent = ego2agent / torch.norm(ego2agent, dim=-1, keepdim=True)
        cossim = torch.sum(ego2agent * ego_h, dim=-1) # (NA-1, NS, T')
        infront = cossim >= feasibility_infront_min
        ego_agent_dists[~infront] = float('inf') # make sure they are filtered out

    min_samp_dists, min_samp_inds = torch.min(ego_agent_dists, dim=1) # (NA-1, FT)
    feasible_dist, feasible_time_step = torch.min(min_samp_dists, dim=1) # (NA-1, )
    feasible_time_step = feasible_time_step + feasibility_time
    feasible = (ego_agent_dists < feasibility_thresh).sum(dim=[1, 2]) > 0

    if check_non_drivable_separation:
        min_samp_inds = min_samp_inds[torch.arange(NA), feasible_time_step-feasibility_time]
        # the trajectories corresponding to minimum distances when passing ego
        min_samp_agent_trajs = agent_samples[torch.arange(NA), min_samp_inds]
        min_samp_ego_trajs = ego_samples.expand(NA, NS, FT, 4)[torch.arange(NA), min_samp_inds]
        min_samp_agent_state = min_samp_agent_trajs[torch.arange(NA), feasible_time_step][:, :2]
        min_samp_ego_state = min_samp_ego_trajs[torch.arange(NA), feasible_time_step][:, :2]
        
        intersect_feasible = nutils.check_line_layer(map_env.nusc_raster[:, 0], map_env.nusc_dx, 
                                                        min_samp_agent_state, min_samp_ego_state, 
                                                        map_idx.expand(NA))
        feasible = torch.logical_and(feasible, ~intersect_feasible)

    agent_vels = torch.norm(agent_samples[:, :, 1:, :2] - agent_samples[:, :, :-1, :2], dim=-1) # (NA-1, NS, FT-1)
    max_vels = torch.max(torch.max(agent_vels, dim=1)[0], dim=1)[0] # (NA-1, )
    feasible = torch.logical_and(feasible, max_vels > feasibility_vel)

    return feasible, feasible_time_step, feasible_dist


METRIC_NAMES = ['planner_coll_atk', 'planner_coll_others', 'adv_success', 
                'planner_coll_h', 'planner_coll_ang', 'planner_coll_env',
                'veh_coll_rate', 'env_coll_atk', 'env_coll_others', 
                'match_ext_pos', 'match_ext_ang', 
                'z_ll_atk', 'z_ll_internal', 'z_ll_planner',
                'init_pos_diff_atk', 'init_pos_diff_others'] + \
                ['sol_coll_others', 'sol_coll_env', 'sol_success', 'sol_z_ll', \
                 'sol_vel_mean', 'sol_vel_max', 'sol_acc_mean', 'sol_acc_max', \
                 'sol_hdot_mean', 'sol_hdot_max', 'sol_hddot_mean', 'sol_hddot_max']

def log_metric(metric_dict, stat_str, metric_np):
    if stat_str not in metric_dict:
        metric_dict[stat_str] = []
    metric_dict[stat_str] += metric_np.tolist()
    return metric_dict

def log_freq_stat(freq_dict_cnt, freq_dict_total, stat_str, cnt_add, tot_add):
    if stat_str not in freq_dict_cnt:
        freq_dict_cnt[stat_str] = 0
        freq_dict_total[stat_str] = 0
    freq_dict_cnt[stat_str] += cnt_add
    freq_dict_total[stat_str] += tot_add
    return freq_dict_cnt, freq_dict_total

def print_metrics(metrics, freq_metrics_cnt, freq_metrics_total):
    for k, v in metrics.items():
        Logger.log('%s = %f' % (k, np.mean(v)))
    for k, v in freq_metrics_cnt.items():
        Logger.log('%s = %f' % (k, float(v) / freq_metrics_total[k]))

def wandb_log_metrics(metrics, freq_metrics_cnt, freq_metrics_total):
    import wandb
    wandb_metrics = {}
    for k, v in metrics.items():
        wandb_metrics[k] = np.mean(v)
    for k, v in freq_metrics_cnt.items():
        wandb_metrics[k] = float(v) / freq_metrics_total[k]
    wandb.log(wandb_metrics)

def viz_optim_results(out_path, scene_graph, map_idx, map_env,
                        model, future_pred, planner_name, attack_agt, crop_t,
                        viz_bounds=[-60.0, -60.0, 60.0, 60.0],
                        bidx=0,
                        ow_gt=None,
                        show_gt=False,
                        show_gt_idx=None):
    '''
    attack_agt is LOCAL - i.e. with respect to the subgraph at batch index bidx
    '''
    NA = scene_graph.ptr[bidx+1] - scene_graph.ptr[bidx]
    nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path,
                                model.get_normalizer(), model.get_att_normalizer(),
                                future_pred=future_pred,
                                viz_traj=True,
                                make_video=False,
                                show_gt=show_gt,
                                show_gt_idx=show_gt_idx, # since gt is the planner in this case
                                # traj_color_val=-sample_pred['z_mdist'].detach(),
                                # traj_color_bounds=[-7, 0]
                                viz_bounds=viz_bounds,
                                crop_t=crop_t,
                                center_viz=crop_t is None,
                                car_colors=nutils.get_adv_coloring(NA, attack_agt, 0),
                                ow_gt=ow_gt
                                )
    nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path + '_vid',
                                model.get_normalizer(), model.get_att_normalizer(),
                                future_pred=future_pred,
                                viz_traj=False,
                                make_video=True,
                                show_gt=show_gt,
                                show_gt_idx=show_gt_idx, # show gt just for ego planner
                                viz_bounds=viz_bounds,
                                crop_t=crop_t,
                                center_viz=crop_t is None,
                                car_colors=nutils.get_adv_coloring(NA, attack_agt, 0),
                                ow_gt=ow_gt
                                )

def prepare_output_dict(scene_graph, map_idx, map_env, dt, model,
                        init_fut_traj,
                        adv_fut_traj,
                        sol_fut_traj=None,
                        attack_agt=None,
                        attack_t=None,
                        adv_z=None,
                        sol_z=None,
                        prior_distrib=None,
                        attack_bike_params=None,
                        internal_ego_traj=None):
    out_dict = {'N' :  int(init_fut_traj.size(0)), 'dt' : dt}
    map_name = map_env.map_list[map_idx]
    out_dict['map'] = map_name

    # unnormalize trajectories and lw
    normalizer = model.get_normalizer()
    past = normalizer.unnormalize(scene_graph.past_gt)
    init_fut_traj = normalizer.unnormalize(init_fut_traj)
    adv_fut_traj = normalizer.unnormalize(adv_fut_traj)
    lw = model.get_att_normalizer().unnormalize(scene_graph.lw)

    # vehicle attributes
    lw_out = lw.cpu().numpy()
    out_dict['lw'] = lw_out.tolist()
    sem_out = scene_graph.sem.cpu().numpy()
    out_dict['sem'] = sem_out.tolist()

    # past motion (shared among all trajectories)
    past_out = past.cpu().numpy()
    out_dict['past'] = past_out.tolist()
    # initialization
    init_out = init_fut_traj.cpu().numpy()
    out_dict['fut_init'] = init_out.tolist()
    # adversarial scene
    adv_out = adv_fut_traj.cpu().numpy()
    out_dict['fut_adv'] = adv_out.tolist()
    if internal_ego_traj is not None:
        internal_ego_traj = normalizer.unnormalize(internal_ego_traj)
        internal_ego_out = internal_ego_traj.cpu().numpy()
        out_dict['fut_internal_ego'] = internal_ego_out.tolist()
    # solution
    if sol_fut_traj is not None:
        sol_fut_traj = normalizer.unnormalize(sol_fut_traj)
        sol_out = sol_fut_traj.cpu().numpy()
        out_dict['fut_sol'] = sol_out.tolist()
    # attackers and t
    if attack_agt is not None:
        out_dict['attack_agt'] = int(attack_agt)
    if attack_t is not None:
        out_dict['attack_t'] = int(attack_t)
    # latents
    if adv_z is not None:
        out_dict['z_adv'] = adv_z.detach().cpu().numpy().tolist()
    if sol_z is not None:
        out_dict['z_sol'] = sol_z.detach().cpu().numpy().tolist()
    if prior_distrib is not None:
        prior_mean = prior_distrib[0].cpu().numpy()
        prior_var = prior_distrib[1].cpu().numpy()
        out_dict['z_prior'] = {'mean' : prior_mean.tolist(), 'var' : prior_var.tolist()}
    
    # bicycle acceleration profile for baselines
    if attack_bike_params is not None:
        out_dict['attack_bike_prof'] = attack_bike_params.cpu().numpy().tolist()

    return out_dict
