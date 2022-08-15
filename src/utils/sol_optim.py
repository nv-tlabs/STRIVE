# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from losses.traffic_model import compute_coll_rate_env
from datasets import nuscenes_utils as nutils

from utils.logger import Logger, throw_err
from utils.adv_gen_optim import collate_tgt_other_z
from utils.scenario_gen import log_metric, log_freq_stat

def run_find_solution_optim(cur_z, final_result_traj, future_len, lr, loss_weights, model, 
                            scene_graph, map_env, map_idx,
                            num_iters, embed_info,
                            tgt_prior_distrib, other_prior_distrib):
    '''
    :param cur_z: final latent features from previous optim.
    :param final_result_traj: output from initial adv gen optimzation (NA x 1 x FT x 4)
    '''
    B = map_idx.size(0)
    NA = final_result_traj.size(0)

    # set up optimization
    other_match_traj = final_result_traj[:, 0]
    tgt_mask = torch.zeros((NA), dtype=torch.bool).to(other_match_traj.device)
    tgt_mask[scene_graph.ptr[:-1]] = True # target is always at index 0 of each scene graph
    other_match_traj = model.get_normalizer().unnormalize(other_match_traj[~tgt_mask])
    other_match_traj = other_match_traj.view(other_match_traj.size(0), 1, other_match_traj.size(1), 4)

    # initialize latents
    tgt_z = tgt_prior_distrib[0]
    tgt_z = tgt_z.view(B, 1, -1).clone().detach() 
    tgt_z.requires_grad = True

    other_z_all = cur_z[~tgt_mask]
    other_z_all = other_z_all.view(NA-B, 1, -1).clone().detach()
    other_z_all.requires_grad = True

    optim_z = [tgt_z, other_z_all]
    sol_optim = optim.Adam(optim_z, lr=lr)

    # create loss functions
    from losses.adv_gen_nusc import AvoidCollLoss, TgtMatchingLoss
    sol_loss_weights = {k[4:] : v for k, v in loss_weights.items() if k[:4] == 'sol_'}
    cur_att = model.get_att_normalizer().unnormalize(scene_graph.lw)
    cur_map_idx = map_idx[scene_graph.batch]
    tgt_init_z = tgt_z.clone().detach()
    tgt_ptr = scene_graph.ptr
    avoid_loss = AvoidCollLoss(sol_loss_weights, cur_att,
                                cur_map_idx,
                                map_env,
                                tgt_init_z,
                                veh_coll_buffer=0.5,
                                single_veh_idx=0, # only want losses on planner node
                                ptr=tgt_ptr)
    match_loss = TgtMatchingLoss(sol_loss_weights)

    # run optim
    num_optim_iter = num_iters
    pbar_optim = tqdm.tqdm(range(num_optim_iter))
    for oidx in pbar_optim:
        def closure():
            sol_optim.zero_grad()

            # decode to get current future
            tgt_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all.detach())
            tgt_decoder_out = model.decode_embedding(tgt_loss_input_z, embed_info, scene_graph, map_idx, map_env,
                                                     nfuture=future_len)
            match_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z.detach(), other_z_all)
            match_decoder_out = model.decode_embedding(match_loss_input_z, embed_info, scene_graph, map_idx, map_env)

            # compute loss for target agent
            tgt_future_pred = model.get_normalizer().unnormalize(tgt_decoder_out['future_pred']) # (NA, 1, T, 4)
            tgt_future_pred = tgt_future_pred.transpose(0, 1).reshape(NA, future_len, 4)
            tgt_avoid_loss_dict = avoid_loss(tgt_future_pred,
                                            tgt_z,
                                            tgt_prior_distrib)
            loss_dict = {'tgt_' + k : v for k, v in tgt_avoid_loss_dict.items()}

            # compute loss for other agents
            other_future_pred = model.get_normalizer().unnormalize(match_decoder_out['future_pred'])
            other_future_pred = other_future_pred[~tgt_mask]
            match_loss_dict = match_loss(other_future_pred,
                                        other_match_traj,
                                        other_z_all,
                                        other_prior_distrib)
            match_loss_dict = {'other_' + k : v for k, v in match_loss_dict.items()}
            loss_dict = {**loss_dict, **match_loss_dict}

            loss = loss_dict['tgt_loss'] + loss_dict['other_loss']

            progress_bar_metrics = {}
            for k, v in loss_dict.items():
                if v is None:
                    continue
                progress_bar_metrics[k] = torch.mean(v).item()
                print('%s = %f' % (k, progress_bar_metrics[k]))
            pbar_optim.set_postfix(progress_bar_metrics)

            # backprop
            loss.backward()
            return loss

        # update
        closure()
        sol_optim.step()

    # get final results
    cur_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all)
    with torch.no_grad():
        sol_decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env)
    sol_result_traj = sol_decoder_out['future_pred'].clone().detach()
    # replace the other adversaries with direct output of adv gen optim
    sol_result_traj[~tgt_mask] = model.get_normalizer().normalize(other_match_traj)

    return cur_z, sol_result_traj, sol_decoder_out


def compute_sol_success(final_result_traj, model, scene_graph, map_env, map_idx,
                        use_map_coll=True):
    '''
    Whether solution optim succeeded
    All inputs assumed NORMALIZED.
    :param final_result_traj: (NA, 1, FT, 4) the final scenario where the agent at idx=0 is the solution.
    '''
    from losses.adv_gen_nusc import check_single_veh_coll

    # final result trajectories
    sol_fut = model.get_normalizer().unnormalize(final_result_traj[0, 0])
    other_fut = model.get_normalizer().unnormalize(final_result_traj[1:, 0])
    # agent attribs
    planner_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[0])
    other_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[1:])

    #
    # Collision between solution and any other agents
    #
    planner_coll_all, planner_coll_time = check_single_veh_coll(sol_fut, planner_lw, other_fut, other_lw)
    planner_coll_others = np.sum(planner_coll_all) > 0

    #
    # If impossible scenario (failed to find solution)
    #
    sol_impossible = planner_coll_others

    if use_map_coll:
        #
        # Collisions with environment
        #
        fin_coll_env_dict = compute_coll_rate_env(scene_graph, map_idx, final_result_traj.contiguous(),
                                            map_env, model.get_normalizer(), model.get_att_normalizer(),
                                            ego_only=True)
        fin_coll_env = fin_coll_env_dict['did_collide'].cpu().numpy()[:, 0] # NA

        sol_impossible = sol_impossible or fin_coll_env[0].item()

    
    return not sol_impossible
