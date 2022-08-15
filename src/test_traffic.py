# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

'''
Runs through nuScenes dataset and runs various evaluations on the traffic model.

By default, computes the same losses/errors as during training.
'''

import os, shutil
import time
import tqdm
import torch
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from losses.traffic_model import TrafficModelLoss, compute_disp_err, compute_coll_rate_env
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from losses.traffic_model import compute_coll_rate_veh
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, load_state
from utils.config import get_parser, add_base_args

def parse_cfg():
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    parser = get_parser('Test motion model')
    parser = add_base_args(parser)

    # additional data args
    parser.add_argument('--test_on_val', dest='test_on_val', action='store_true',
                        help="If given, uses the validation dataset rather than test set for evaluation.")
    parser.set_defaults(test_on_val=False)
    parser.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                        help="If given, shuffles test dataset.")
    parser.set_defaults(shuffle_test=False)

    #
    # test options
    #

    # reconstruct (use posterior)
    parser.add_argument('--test_recon_viz_multi', dest='test_recon_viz_multi', action='store_true',
                        help="Save all-agent visualization for reconstructing all test sequences.")
    parser.set_defaults(test_recon_viz_multi=False)
    parser.add_argument('--test_recon_coll_rate', dest='test_recon_coll_rate', action='store_true',
                        help="Computes collision rate of reconstructed test trajectories")
    parser.set_defaults(test_recon_coll_rate=False)
    # sample (use prior)
    parser.add_argument('--test_sample_viz_multi', dest='test_sample_viz_multi', action='store_true',
                        help="Save all-agent visualization for samplin all test sequences.")
    parser.set_defaults(test_sample_viz_multi=False)
    parser.add_argument('--test_sample_viz_rollout', dest='test_sample_viz_rollout', action='store_true',
                        help="Create videos of multiple sampled futures individually.")
    parser.set_defaults(test_sample_viz_rollout=False)

    parser.add_argument('--test_sample_disp_err', dest='test_sample_disp_err', action='store_true',
                        help="Computes min displacement errors (ADE, FDE, and angle-based version) based on multiple samples.")
    parser.set_defaults(test_sample_disp_err=False)
    parser.add_argument('--test_sample_coll_rate', dest='test_sample_coll_rate', action='store_true',
                        help="Computes collision rate of N random samples.")
    parser.set_defaults(test_sample_coll_rate=False)

    parser.add_argument('--test_sample_num', type=int, default=3, help='Number of future traj to sample')
    parser.add_argument('--test_sample_future_len', type=int, default=None, help='If not None, samples this many steps into the future rather than future_len')

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict


def run_one_epoch(data_loader, model, map_env, loss_fn, device, out_path,
                  test_recon_viz_multi=False,
                  test_recon_coll_rate=False,
                  test_sample_viz_multi=False,
                  test_sample_viz_rollout=False,
                  test_sample_disp_err=False,
                  test_sample_coll_rate=False,
                  test_sample_num=3, # how many futures to sample from the prior
                  test_sample_future_len=None,
                  use_challenge_splits=False
                  ):
    '''
    Run through test dataset and perform various desired evaluations.
    '''    
    pbar = tqdm.tqdm(data_loader)

    if test_recon_viz_multi:
        recon_multi_agent_out_path = os.path.join(out_path, 'viz_recon_multi')
        mkdir(recon_multi_agent_out_path)
    if test_sample_viz_multi:
        sample_multi_agent_out_path = os.path.join(out_path, 'viz_sample_multi')
        mkdir(sample_multi_agent_out_path)
    if test_sample_viz_rollout:
        sample_rollout_out_path = os.path.join(out_path, 'viz_sample_rollout')
        mkdir(sample_rollout_out_path)

    metrics = {} # regular error metrics, want to compute mean
    freq_metrics = {} # frequency metrics where we sum occurences of something.
    data_idx = 0
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        # print(scene_graph)
        # print(map_idx)
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)

        # uses mean of posterior to compute recon errors
        pred = model(scene_graph, map_idx, map_env, use_post_mean=True)
        loss_dict = loss_fn(scene_graph, pred)
        loss = loss_dict['loss'][0]
        # compute interpretable errors
        err_dict = loss_fn.compute_err(scene_graph, pred,
                                        model.get_normalizer())
        # metrics to save
        batch_metrics = {**loss_dict, **err_dict}
        batch_freq_metrics = {}
        #
        # Reconstruction-based Evaluations
        #
        recon_pred = None
        if test_recon_viz_multi or test_recon_coll_rate:
            recon_pred = model.reconstruct(scene_graph, map_idx, map_env)
            
        if test_recon_viz_multi:
            # Visualize all results for each agent jointly
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx
                pred_prefix = 'test_recon_multi_%08d_pred' % (multi_agt_data_idx)
                pred_out_path = os.path.join(recon_multi_agent_out_path, pred_prefix)
                nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, pred_out_path,
                                            model.get_normalizer(), model.get_att_normalizer(),
                                            future_pred=recon_pred['future_pred'],
                                            viz_traj=True,
                                            make_video=False,
                                            show_gt=True)

        if test_recon_coll_rate:
            coll_pred = {k : v for k, v in recon_pred.items()}
            coll_pred['future_pred'] = coll_pred['future_pred'].unsqueeze(1) # 1 "sample"

            # first environment collisions with veh at idx 0
            coll_rate_dict = compute_coll_rate_env(scene_graph, map_idx, coll_pred, map_env,
                                                       model.get_normalizer(), model.get_att_normalizer(),
                                                       ego_only=True)
            coll_rate_dict = {'recon_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

            # now vehicle collisions
            coll_rate_dict = compute_coll_rate_veh(scene_graph, coll_pred,
                                                       model.get_normalizer(), model.get_att_normalizer())
            coll_rate_dict = {'recon_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

            
        #
        # Sampling-based Evaluations
        #
        sample_pred = None
        if test_sample_disp_err or test_sample_viz_multi or test_sample_coll_rate or test_sample_viz_rollout:
            # sample_pred = model.sample(scene_graph, map_idx, map_env, test_sample_num,
                                        # include_mean=False, nfuture=test_sample_future_len)
            # batched version uses more memory, slightly faster
            sample_pred = model.sample_batched(scene_graph, map_idx, map_env, test_sample_num,
                                        include_mean=False, nfuture=test_sample_future_len)

        if test_sample_viz_multi:
            # Visualize all results for all agents at once
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx
                pred_prefix = 'test_sample_multi_%08d_pred' % (multi_agt_data_idx)
                pred_out_path = os.path.join(sample_multi_agent_out_path, pred_prefix)
                # pred 
                nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, pred_out_path,
                                            model.get_normalizer(), model.get_att_normalizer(),
                                            future_pred=sample_pred['future_pred'],
                                            viz_traj=True,
                                            make_video=False,
                                            show_gt=False)

        if test_sample_viz_rollout:
             # Visualize all results for all agents at once
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx
                pred_prefix = 'test_sample_rollout_%08d_pred' % (multi_agt_data_idx)
                pred_out_path = os.path.join(sample_rollout_out_path, pred_prefix)
                # pred
                nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, pred_out_path,
                                            model.get_normalizer(), model.get_att_normalizer(),
                                            future_pred=sample_pred['future_pred'],
                                            viz_traj=False,
                                            make_video=True,
                                            viz_bounds=[-45.0, -45.0, 45.0, 45.0],
                                            center_viz=True)

        if test_sample_disp_err:
            # compute minADE and minFDE for veh at idx 0 of graph 
            disp_err_dict = compute_disp_err(scene_graph, sample_pred, model.get_normalizer())
            batch_metrics = {**batch_metrics, **disp_err_dict}

        if test_sample_coll_rate:
            # compute collision frequency for idx 0 vehicle and env
            coll_rate_dict = compute_coll_rate_env(scene_graph, map_idx, sample_pred, map_env,
                                                       model.get_normalizer(), model.get_att_normalizer(),
                                                       ego_only=True)
            coll_rate_dict = {'sample_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

            # now between all vehicles
            coll_rate_dict = compute_coll_rate_veh(scene_graph, sample_pred,
                                                       model.get_normalizer(), model.get_att_normalizer())
            coll_rate_dict = {'sample_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

        data_idx += B

        # Metrics to log to the tqdm progress bar
        progress_bar_metrics = {}
        # Keep track of metrics over whole epoch
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
            progress_bar_metrics[k] = torch.mean(v).item()
            # print('%s = %f' % (k, torch.mean(torch.cat(metrics[k])).item()))

        freq_prefixes = ['recon_', 'sample_']
        freq_postfixes = ['_map', '_veh']
        freq_expts = [test_recon_coll_rate, test_sample_coll_rate]
        used_prefixes = [pref for pid, pref in enumerate(freq_prefixes) if freq_expts[pid]]
        used_postfixes = freq_postfixes #[post for pid, post in enumerate(freq_postfixes) if freq_expts[pid]]
        for freq_pref in used_prefixes:
            for freq_post in used_postfixes:
                num_coll = freq_pref + 'num_coll' + freq_post
                num_traj = freq_pref + 'num_traj' + freq_post
                if num_coll not in freq_metrics:
                    freq_metrics[num_coll] = 0
                    freq_metrics[num_traj] = 0
                freq_metrics[num_coll] += float(batch_freq_metrics[num_coll])
                freq_metrics[num_traj] += float(batch_freq_metrics[num_traj])
                progress_bar_metrics[freq_pref + 'coll_freq' + freq_post] = float(batch_freq_metrics[num_coll]) / batch_freq_metrics[num_traj]

        # Log the loss to the tqdm progress bar
        pbar.set_postfix(progress_bar_metrics)

    epoch_metrics = {}
    for k, v in metrics.items():
        metrics[k] = torch.cat(metrics[k])
        epoch_metrics["Test Mean " + k] = torch.mean(metrics[k]).item()

    for freq_pref in used_prefixes:
        for freq_post in used_postfixes:
            num_coll = freq_pref + 'num_coll' + freq_post
            num_traj = freq_pref + 'num_traj' + freq_post
            test_coll_freq = freq_metrics[num_coll]  / freq_metrics[num_traj]
            epoch_metrics["Test (%s, %s) Collision Freq" % (freq_pref, freq_post)] = test_coll_freq

    Logger.log('Final ===================================== ')
    if len(epoch_metrics) > 0:
        for k, v in epoch_metrics.items():
            Logger.log('%s = %f' % (k, v))


def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'test_log.txt')
    Logger.init(log_path)
    # save arguments used
    Logger.log('Args: ' + str(cfg_dict))

    # device setup
    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    # load dataset
    test_dataset = map_env = None
    # first create map environment
    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(data_path,
                            bounds=cfg.map_obs_bounds,
                            L=cfg.map_obs_size_pix,
                            W=cfg.map_obs_size_pix,
                            layers=cfg.map_layers,
                            device=device,
                            )
    test_dataset = NuScenesDataset(data_path, map_env,
                            version=cfg.data_version,
                            split='test' if not cfg.test_on_val else 'val',
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            reduce_cats=cfg.reduce_cats,
                            use_challenge_splits=cfg.use_challenge_splits
                            )

    # create loaders    
    test_loader = GraphDataLoader(test_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=cfg.shuffle_test,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    # create model
    model = TrafficModel(cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(test_dataset.categories),
                     map_feat_size=cfg.map_feat_size,
                     past_feat_size=cfg.past_feat_size,
                     future_feat_size=cfg.future_feat_size,
                     latent_size=cfg.latent_size,
                     output_bicycle=cfg.model_output_bicycle,
                     conv_channel_in=map_env.num_layers,
                     conv_kernel_list=cfg.conv_kernel_list,
                     conv_stride_list=cfg.conv_stride_list,
                     conv_filter_list=cfg.conv_filter_list
                     ).to(device)
    print(model)
    loss_weights = {
        'recon' : 1.0,
        'kl' : 1.0,
        'coll_veh_prior' : 0.0,
        'coll_env_prior' : 0.0
    }
    loss_fn = TrafficModelLoss(loss_weights).to(device)

    # load model weights
    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate a trained model!')

    Logger.log('Num model params: %d' % (count_params(model)))

    # so can unnormalize as needed
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # run evaluations on test data
    model.eval()
    with torch.no_grad():
        start_t = time.time()
        run_one_epoch(test_loader, model, map_env, loss_fn, device, cfg.out,
                    test_recon_viz_multi=cfg.test_recon_viz_multi,
                    test_recon_coll_rate=cfg.test_recon_coll_rate,
                    test_sample_viz_multi=cfg.test_sample_viz_multi,
                    test_sample_viz_rollout=cfg.test_sample_viz_rollout,
                    test_sample_disp_err=cfg.test_sample_disp_err,
                    test_sample_coll_rate=cfg.test_sample_coll_rate,
                    test_sample_num=cfg.test_sample_num,
                    test_sample_future_len=cfg.test_sample_future_len,
                    use_challenge_splits=cfg.use_challenge_splits
                    )
        Logger.log('Test time: %f s' % (time.time() - start_t))


if __name__ == "__main__":
    main()
