# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import tqdm

import numpy as np
import torch
import torch.optim as optim

def run_init_optim(cur_z, init_traj, traj_vis, lr, loss_weights, model, 
                    scene_graph, map_env, map_idx, num_iters, embed_info, prior_distrib):
    B = map_idx.size(0)
    NA = init_traj.size(0)

    # set up optimization
    init_traj = model.get_normalizer().unnormalize(init_traj)[traj_vis == 1.0]
    cur_z = cur_z.clone().detach()
    cur_z.requires_grad = True
    optim_z = [cur_z]
    init_optim = optim.Adam(optim_z, lr=lr)

    # create loss functions
    from losses.adv_gen_nusc import TgtMatchingLoss
    init_loss_weights = {k[5:] : v for k, v in loss_weights.items() if k[:5] == 'init_'}
    match_loss = TgtMatchingLoss(init_loss_weights)

    # run optim
    pbar_optim = tqdm.tqdm(range(num_iters))
    for oidx in pbar_optim:
        def closure():
            init_optim.zero_grad()

            # decode to get current future
            decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env)
            # compute matching loss
            future_pred = model.get_normalizer().unnormalize(decoder_out['future_pred'])
            # only want to compute loss for timesteps we have GT data
            future_pred = future_pred[traj_vis == 1.0]

            loss_dict = match_loss(future_pred,
                                    init_traj,
                                    cur_z,
                                    prior_distrib)
            loss = loss_dict['loss']

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
        closure() # ADAM
        init_optim.step()

    # get final results
    with torch.no_grad():
        init_decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env)
    init_result_traj = init_decoder_out['future_pred'].clone().detach()

    return cur_z, init_result_traj, init_decoder_out
