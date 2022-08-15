# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, time
import gc
import tqdm
import torch
import torch.optim as optim
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch as GraphBatch

from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from losses.traffic_model import compute_coll_rate_env
from losses.adv_gen_nusc import check_pairwise_veh_coll, AvoidCollLoss

from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, load_state
from utils.scenario_gen import detach_embed_info
from utils.scenario_gen import log_metric, log_freq_stat, print_metrics, wandb_log_metrics, prepare_output_dict
from utils.config import get_parser, add_base_args

def parse_cfg():
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    parser = get_parser('Sample+optim motion model')
    parser = add_base_args(parser)

    # dataset options
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'train'],
                        help='Which split of the dataset to sample traffic from')
    parser.add_argument('--seq_interval', type=int, default=10, help='skips ahead this many steps from start of current sequence to get the next sequence.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help="Shuffle data")
    parser.set_defaults(shuffle=False)

    # optim options
    parser.add_argument('--optim_use_lbfgs', dest='optim_use_adam', action='store_false',
                        help="If given, uses LBFGS optimizer rather than ADAM")
    parser.set_defaults(optim_use_adam=True)

    parser.add_argument('--feasibility_num', type=int, default=1, help='Min number of agents that need to be in a scene to sample/optimize.')

    parser.add_argument('--samp_future_len', type=int, default=16, help='The number of timesteps to roll out for the samples and optimization scenario.')
    parser.add_argument('--save_future_len', type=int, default=12, help='The number of future timesteps to actually visualize/save.')

    parser.add_argument('--loss_coll_veh', type=float, default=100.0, help='Loss to AVOID vehicle-vehicle collisions in the scene.')
    parser.add_argument('--loss_coll_env', type=float, default=100.0, help='Loss to AVOID vehicle-environment collisions in the scene.')
    parser.add_argument('--loss_init_z', type=float, default=0.01, help='Loss to keep latent z near initialization.')
    parser.add_argument('--loss_motion_prior', type=float, default=1.0, help='Loss to keep latent z likely under the motion prior VAE.')

    parser.add_argument('--num_iters', type=int, default=10, help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate for optimizer.')

    parser.add_argument('--viz', dest='viz', action='store_true',
                        help="If given, saves visualization before and after optimization.")
    parser.set_defaults(viz=False)
    parser.add_argument('--save', dest='save', action='store_true',
                        help="If given, saves the scenarios so they can be used later.")
    parser.set_defaults(save=False)

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict

def compute_metrics(metrics, freq_metrics_cnt, freq_metrics_total, cur_z,
                            final_result_traj,
                            model, scene_graph, map_env, map_idx, prior_distrib):
    '''
    All inputs assumed NORMALIZED.
    :param final_result_traj: (NA, 1, FT, 4) the final scenario where the agent at idx=0 is the true planner
                                            reaction to the scenario (NOT the model's prediction of the planner)
    '''
    seq_metrics = dict()

    # final result trajectories
    pred_fut = model.get_normalizer().unnormalize(final_result_traj[:,0])
    # agent attribs
    lw = model.get_att_normalizer().unnormalize(scene_graph.lw)

    #
    # Collisions between any controlled agents
    #
    controlled_coll_dict = check_pairwise_veh_coll(pred_fut, lw)
    freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                        'veh_coll', int(controlled_coll_dict['num_coll_veh']),
                                                        int(controlled_coll_dict['num_traj_veh']))
    seq_metrics['veh_coll'] = int(controlled_coll_dict['num_coll_veh'])

    #
    # Collisions with environment
    #
    fin_coll_env_dict = compute_coll_rate_env(scene_graph, map_idx, final_result_traj.contiguous(),
                                        map_env, model.get_normalizer(), model.get_att_normalizer(),
                                        ego_only=False)
    fin_coll_env = fin_coll_env_dict['did_collide'].cpu().numpy()[:, 0] # NA

    env_coll = np.sum(fin_coll_env)
    freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                            'env_coll', int(env_coll), fin_coll_env.shape[0])
    seq_metrics['env_coll'] = int(env_coll)

    return metrics, freq_metrics_cnt, freq_metrics_total, seq_metrics

def viz_optim_results(out_path, scene_graph, map_idx, map_env,
                        model, future_pred,
                        viz_bounds=[-60.0, -60.0, 60.0, 60.0],
                        bidx=0):
    nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path,
                                model.get_normalizer(), model.get_att_normalizer(),
                                future_pred=future_pred,
                                viz_traj=True,
                                make_video=False,
                                show_gt=False,
                                viz_bounds=viz_bounds,
                                center_viz=True
                                )
    nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path + '_vid',
                                model.get_normalizer(), model.get_att_normalizer(),
                                future_pred=future_pred,
                                viz_traj=False,
                                make_video=True,
                                show_gt=False,
                                viz_bounds=viz_bounds,
                                center_viz=True
                                )

def refine_traffic_optim(scene_graph, map_idx, map_env, model,
                         loss_weights, num_iters, samp_future_len,
                          save_future_len, optim_use_adam, lr):
    '''
    Randomly sample future with traffic model, and run optimization to remove any vehicle/environment collisions.
    '''
    with torch.no_grad():
        # Get sample to start from
        sample_pred = model.sample_batched(scene_graph, map_idx, map_env, 1, include_mean=False)
        # sample_pred = model.sample(scene_graph, map_idx, map_env, 1, include_mean=False)
        # embed past and map to get inputs to decoder used during optim
        embed_info_attached = model.embed(scene_graph, map_idx, map_env)

    # need to detach all the encoder outputs from current comp graph to be used in optimization
    embed_info = detach_embed_info(embed_info_attached)

    init_future_pred = sample_pred['future_pred'][:,0]
    z_init = sample_pred['z_samp'][:,0].clone().detach()

    cur_z = z_init
    cur_z.requires_grad = True
    if optim_use_adam:
        scene_optim = optim.Adam([cur_z], lr=lr)
    else:
        scene_optim = optim.LBFGS([cur_z],
                                max_iter=20,
                                lr=lr,
                                line_search_fn='strong_wolfe')

    # create loss function
    avoid_loss = AvoidCollLoss(loss_weights,
                                model.get_att_normalizer().unnormalize(scene_graph.lw),
                                map_idx[scene_graph.batch],
                                map_env,
                                cur_z.clone().detach(),
                                veh_coll_buffer=0.2)

    # run optim
    pbar_optim = tqdm.tqdm(range(num_iters))
    for oidx in pbar_optim:
        def closure():
            scene_optim.zero_grad()

            # decode to get current future
            decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env,
                                                    nfuture=samp_future_len)

            # compute loss
            cur_future_pred = model.get_normalizer().unnormalize(decoder_out['future_pred'])
            loss_dict = avoid_loss(cur_future_pred,
                                    cur_z,
                                    embed_info['prior_out'])
            loss = loss_dict['loss']

            # Metrics to log to the tqdm progress bar
            progress_bar_metrics = {}
            # Keep track of metrics over whole epoch
            for k, v in loss_dict.items():
                if v is None:
                    continue
                progress_bar_metrics[k] = torch.mean(v).item()
                print('%s = %f' % (k, progress_bar_metrics[k]))
            # Log the loss to the tqdm progress bar
            pbar_optim.set_postfix(progress_bar_metrics)

            # backprop
            loss.backward()
            return loss

        # update
        if optim_use_adam:
            closure() # ADAM
            scene_optim.step()
        else:
            scene_optim.step(closure)

    optim_decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env,
                                                    nfuture=save_future_len)
    optim_result_traj = optim_decoder_out['future_pred'].unsqueeze(1).clone().detach()

    return init_future_pred, cur_z, optim_result_traj, embed_info

def run_one_epoch(data_loader, batch_size, model, map_env, device, out_path, loss_weights,
                  feasibility_NA=1,
                  samp_future_len=12,
                  save_future_len=12,
                  optim_use_adam=False,
                  num_iters=10,
                  lr=1.0,
                  viz=True,
                  save=True,
                  use_wandb=False
                  ):
    '''
    Run through dataset, sample futures, and refine sample to avoid collisions.
    '''
    pbar_data = tqdm.tqdm(data_loader)

    gen_out_path = out_path
    mkdir(gen_out_path)
    if viz:
        gen_out_path_viz = os.path.join(gen_out_path, 'viz_results')
        mkdir(gen_out_path_viz)
    if save:
        gen_out_path_scenes = os.path.join(gen_out_path, 'scenario_results')
        mkdir(gen_out_path_scenes)

    metrics = {}
    freq_metrics_cnt = {}
    freq_metrics_total = {}
    data_idx = 0
    empty_cache = False
    batch_i = []
    batch_scene_graph = []
    batch_map_idx = []
    batch_total_NA = 0
    for i, data in enumerate(pbar_data):
        start_t = time.time()
        scene_graph, map_idx = data
        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()
        try:
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            print('scene_%d' % (i))
            print(scene_graph)
            is_last_batch = i == (len(data_loader)-1)

            is_feas = True
            if scene_graph.future_gt.size(0) < feasibility_NA:
                Logger.log('Only %d agents in scene, skipping...' % (scene_graph.future_gt.size(0)))
                is_feas = False
                if not is_last_batch:
                    continue

            if is_feas:
                Logger.log('Feasible. Adding to batch...')
                batch_scene_graph += scene_graph.to_data_list()
                batch_map_idx.append(map_idx)
                batch_i.append(i)
                batch_total_NA += scene_graph.future_gt.size(0)
                Logger.log('Batch NA: %d' % (batch_total_NA))

            if batch_total_NA < batch_size and not is_last_batch:
                # collect more before performing optim
                continue
            else:
                if len(batch_scene_graph) == 0:
                    # this is the last seq in dataset, and we have no other seqs queued
                    continue
                # create the batch
                scene_graph = GraphBatch.from_data_list(batch_scene_graph)
                map_idx = torch.cat(batch_map_idx, dim=0)
                cur_batch_i = batch_i

                Logger.log('Formed batch! Starting optimization...')
                Logger.log(scene_graph)

                # reset
                batch_scene_graph = []
                batch_map_idx = []
                batch_i = []
                batch_total_NA = 0

            B = map_idx.size(0)
            NA = scene_graph.past.size(0)

            #
            # Sample traffic model and optimize
            #
            init_future_pred, cur_z, optim_result_traj, embed_info = refine_traffic_optim(scene_graph, map_idx, map_env, model,
                                                                                            loss_weights, num_iters, samp_future_len,
                                                                                            save_future_len, optim_use_adam, lr)

            # compute evaluation metrics to determine success
            success_list = []
            for b in range(B):
                metric_res = compute_metrics(metrics, freq_metrics_cnt, freq_metrics_total, 
                                                cur_z[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                optim_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                model,
                                                GraphBatch.from_data_list([scene_graph.to_data_list()[b]]),
                                                map_env,
                                                map_idx[b:b+1],
                                                (embed_info['prior_out'][0][scene_graph.ptr[b]:scene_graph.ptr[b+1]], embed_info['prior_out'][1][scene_graph.ptr[b]:scene_graph.ptr[b+1]])
                                                )
                metrics, freq_metrics_cnt, freq_metrics_total, seq_metrics = metric_res
                success_list.append(seq_metrics['veh_coll'] == 0 and seq_metrics['env_coll'] == 0)

            Logger.log('Optimized sequence in %f sec!' % (time.time() - start_t))

            # output scenario and viz
            scene_graph_list = scene_graph.to_data_list()
            for b in range(B):
                result_dir = None
                if success_list[b]:
                    result_dir = 'success'
                else:
                    result_dir = 'failed'
                
                if save:
                    import json
                    # save scenario
                    cur_scene_out_path = os.path.join(gen_out_path_scenes, result_dir)
                    mkdir(cur_scene_out_path)

                    scene_out_dict = prepare_output_dict(scene_graph_list[b], map_idx[b].item(), map_env, data_loader.dataset.dt, model,
                                                          init_future_pred[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          optim_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]][:,0]
                                                          )

                    fout_path = os.path.join(cur_scene_out_path, 'scene_%04d.json' % cur_batch_i[b])
                    Logger.log('Saving scene to %s' % (fout_path))
                    with open(fout_path, 'w') as writer:
                        json.dump(scene_out_dict, writer)

                if viz:
                    cur_viz_out_path = os.path.join(gen_out_path_viz, result_dir)
                    mkdir(cur_viz_out_path)

                    # save before viz
                    pred_prefix = 'test_sample_%d_before' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        init_future_pred,
                                        bidx=b)

                    # save before viz
                    pred_prefix = 'test_sample_%d_after' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        optim_result_traj[:,0],
                                        bidx=b)
        except RuntimeError as e:
            Logger.log('Caught error in training batch %s!' % (str(e)))
            Logger.log('Skipping')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            empty_cache = True
            continue

    print_metrics(metrics, freq_metrics_cnt, freq_metrics_total)
    if use_wandb:
        wandb_log_metrics(metrics, freq_metrics_cnt, freq_metrics_total)

def main():
    cfg, cfg_dict = parse_cfg()

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        # wandb setup
        wandb.init(project=cfg.wandb_project, config=cfg_dict,
                    mode='offline' if cfg.wandb_offline else 'online',
                    name=cfg.wandb_name)
        cfg_dict = wandb.config
        cfg_dict = {k : v for k, v in cfg_dict.items()} # copy so can be manually updated
        cfg = dict2obj(cfg_dict)

    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'refine_traffic_log.txt')
    Logger.init(log_path)
    # save arguments used
    Logger.log('Args: ' + str(cfg_dict))

    # device setup
    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    # load dataset
    # first create map environment
    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(data_path,
                            bounds=cfg.map_obs_bounds,
                            L=cfg.map_obs_size_pix,
                            W=cfg.map_obs_size_pix,
                            layers=cfg.map_layers,
                            device=device
                            )
    test_dataset = NuScenesDataset(data_path, map_env,
                            version=cfg.data_version,
                            split=cfg.split,
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            reduce_cats=cfg.reduce_cats,
                            seq_interval=cfg.seq_interval,
                            use_challenge_splits=cfg.use_challenge_splits
                            )

    # create loaders    
    test_loader = GraphDataLoader(test_dataset,
                                    batch_size=1, # will collect batches on the fly to get more deterministic size
                                    shuffle=cfg.shuffle,
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

    # load model weights
    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to sample traffic!')

    Logger.log('Num model params: %d' % (count_params(model)))

    # so can unnormalize as needed
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    loss_weights = {
        'coll_veh' : cfg.loss_coll_veh,
        'coll_env' : cfg.loss_coll_env,
        'motion_prior' : cfg.loss_motion_prior,
        'init_z' : cfg.loss_init_z
    }

    # run through dataset once and generate possible scenarios
    model.train()
    run_one_epoch(test_loader, cfg.batch_size, model, map_env, device, cfg.out, loss_weights,
                  feasibility_NA=cfg.feasibility_num,
                  samp_future_len=cfg.samp_future_len,
                  save_future_len=cfg.save_future_len,
                  optim_use_adam=cfg.optim_use_adam,
                  num_iters=cfg.num_iters,
                  lr=cfg.lr,
                  viz=cfg.viz,
                  save=cfg.save,
                  use_wandb=use_wandb)

    if use_wandb:
        # save full log
        wandb.save(log_path)


if __name__ == "__main__":
    main()
