# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, argparse, time

import gc
import tqdm
import torch
import torch.optim as optim
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader

from models.traffic_model import TrafficModel
from losses.traffic_model import TrafficModelLoss
from nuscenes.nuscenes import NuScenes
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, save_state, load_state, compute_kl_weight, c2c
from utils.config import get_parser, add_base_args

def parse_cfg():
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    parser = get_parser('Train motion model')
    parser = add_base_args(parser)

    # additional dataset options
    parser.add_argument('--scenario_dir', type=str, default=None,
                        help='additional adv gen scenarios to train on')
    parser.add_argument('--data_noise_std', type=float, default=0.0, help='std of noise to add to model input data.')

    # Training options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--val_every', type=int, default=3, help='Number of epochs between validations.')
    parser.add_argument('--save_every', type=int, default=3, help='Number of epochs between saving model checkpoint.')
    parser.add_argument('--print_every', type=int, default=10, help='Number of batches between printing stats.')

    # Optimizer options
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for ADAM')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay on params.')

    # Losses
    parser.add_argument('--loss_kl', type=float, default=0.004, help='KL loss weight')
    parser.add_argument('--kl_anneal_end', type=int, default=20, help='If given, uses KL loss annealing and will reach full weight at this epoch.')
    parser.add_argument('--loss_recon', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--loss_veh_coll_prior', type=float, default=0.05, help='Vehicle collision loss weight for sample from prior')
    parser.add_argument('--loss_env_coll_prior', type=float, default=0.1, help='Map collision loss weight for sample from prior')

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict


def run_one_epoch(data_loader, model, map_env, loss_fn, device, out_path,
                  train=True,
                  optimizer=None,
                  step_counter=0,
                  use_wandb=False):
    '''
    Run through dataset and for a single epoch. Trains if desired.
    '''
    if use_wandb:
        import wandb
    if train and optimizer is None:
        throw_err('Must give optimizer to train!')
    prefix = "Train" if train else "Eval"
    
    pbar = tqdm.tqdm(data_loader)

    if train:
        assert optimizer is not None

    # Vector of all losses and IoU values for the batch
    metrics = {}

    empty_cache = False
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        pred = loss_dict = None
        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()
        try: 
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            # print(scene_graph)
            # print(map_idx)
            B = map_idx.size(0)

            do_sample = loss_fn.loss_weights['coll_veh_prior'] > 0.0 or \
                        loss_fn.loss_weights['coll_env_prior'] > 0.0
            pred = model(scene_graph, map_idx, map_env, future_sample=do_sample)

            loss_dict = loss_fn(scene_graph, pred,
                                map_idx=map_idx,
                                map_env=map_env
                                )
            loss = loss_dict['loss'][0]

            if train:
                # training step for generator
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute interpretable errors
            err_dict = loss_fn.compute_err(scene_graph, pred,
                                            model.get_normalizer())
        except RuntimeError as e:
            Logger.log('Caught error in training batch %s!' % (str(e)))
            Logger.log('Skipping')
            if pred is not None:
                del pred
            if loss_dict is not None:
                del loss_dict
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            empty_cache = True
            continue

        # metrics to save
        batch_metrics = {**loss_dict, **err_dict}

        if use_wandb:
            # per-batch wandb metrics
            wandb_batch_metrics = {}
            if train:
                step_counter += B
                for k, v in batch_metrics.items():
                    if v is not None:
                        wandb_batch_metrics[prefix + " Batch Mean " + k] = torch.mean(v).item()

            if len(wandb_batch_metrics) > 0:
                wandb.log(wandb_batch_metrics, step=step_counter)

        # Metrics to log to the tqdm progress bar
        progress_bar_metrics = {}
        # Keep track of metrics over whole epoch
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(c2c(v))
            progress_bar_metrics[k] = torch.mean(v).item()
        # Log the loss to the tqdm progress bar
        pbar.set_postfix(progress_bar_metrics)

    wandb_epoch_metrics = {}
    for k, v in metrics.items():
        metrics[k] = np.concatenate(metrics[k])
        wandb_epoch_metrics[prefix + " Epoch Mean " + k] = np.mean(metrics[k])

    mean_epoch_loss = wandb_epoch_metrics[prefix + " Epoch Mean loss"]

    if use_wandb and len(wandb_epoch_metrics) > 0:
        wandb.log(wandb_epoch_metrics, step=step_counter)

    return step_counter, mean_epoch_loss


def main():
    cfg, cfg_dict = parse_cfg()

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        # wandb setup
        wandb.init(project=cfg.wandb_project, config=cfg_dict,
                    mode='offline' if cfg.wandb_offline else 'online',
                    name=cfg.wandb_name)

    # create output directory and logging
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'train_log.txt')
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
                            device=device,
                            )

    # create nuscenes object out here and pass into dataset to save memory
    Logger.log('Creating nuscenes data object...')
    nusc_obj = NuScenes(version='v1.0-{}'.format(cfg.data_version),
                        dataroot=data_path,
                        verbose=False)
    train_dataset = NuScenesDataset(data_path, map_env,
                            version=cfg.data_version,
                            split='train',
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            nusc=nusc_obj,
                            noise_std=cfg.data_noise_std,
                            scenario_path=cfg.scenario_dir,
                            use_challenge_splits=cfg.use_challenge_splits,
                            reduce_cats=cfg.reduce_cats
                            )
    val_dataset = NuScenesDataset(data_path, map_env,
                            version=cfg.data_version,
                            split='val',
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            nusc=nusc_obj,
                            use_challenge_splits=cfg.use_challenge_splits,
                            reduce_cats=cfg.reduce_cats
                            )

    # create loaders
    train_loader = GraphDataLoader(train_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=True,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug
    val_loader = GraphDataLoader(val_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=False,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    # create model
    model = TrafficModel(cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(train_dataset.categories),
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
        'recon' : cfg.loss_recon,
        'kl' : cfg.loss_kl,
        'coll_veh_prior' : cfg.loss_veh_coll_prior,
        'coll_env_prior' : cfg.loss_env_coll_prior
    }
    loss_fn = TrafficModelLoss(loss_weights,
                            train_dataset.get_state_normalizer(),
                            train_dataset.get_att_normalizer()).to(device)

    Logger.log('Num model params: %d' % (count_params(model)))

    # create optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    # load model weights & optimizer to start from, if given
    ckpt_epoch = 0
    ckpt_eval_loss = float('inf')
    if cfg.ckpt is not None:
        ckpt_epoch, ckpt_eval_loss = load_state(cfg.ckpt, model,
                                                optimizer=optimizer,
                                                map_location=device)
        Logger.log('Loaded checkpoint from epoch %d with validation loss %f...' % (ckpt_epoch, ckpt_eval_loss))

    # so can unnormalize as needed
    model.set_normalizer(train_dataset.get_state_normalizer())
    model.set_att_normalizer(train_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # KL loss annealing
    use_kl_anneal = cfg.kl_anneal_end is not None
    if use_kl_anneal:
        assert cfg.kl_anneal_end > 0
        Logger.log('Using KL annealing...')

    # run training
    ckpts_path = os.path.join(cfg.out, 'checkpoints')
    mkdir(ckpts_path)
    step_counter = 0
    min_eval_loss = ckpt_eval_loss
    for epoch in range(ckpt_epoch, cfg.epochs):
        Logger.log('Starting epoch %d...' % (epoch))

        # compute loss weights with KL annealing
        if use_kl_anneal:
            cur_beta = compute_kl_weight(epoch, cfg.kl_anneal_end, cfg.loss_kl)
            loss_fn.loss_weights['kl'] = cur_beta
            Logger.log('KL weight %f...' % (loss_fn.loss_weights['kl']))
            if use_wandb:
                wandb.log({'kl_weight' : loss_fn.loss_weights['kl']}, step=step_counter)
            if epoch == cfg.kl_anneal_end:
                Logger.log('KL ANNEALING FINISHED: resetting val loss tracking...')
                min_eval_loss = float('inf')

        # train for one epoch
        start_t = time.time()
        model.train()                           
        step_counter, _ = run_one_epoch(train_loader, model, map_env, loss_fn, device, cfg.out,
                                                        train=True,
                                                        optimizer=optimizer,
                                                        step_counter=step_counter,
                                                        use_wandb=use_wandb)
        # lot of excess memory used by pygeometric
        torch.cuda.empty_cache()
        Logger.log('Epoch time: %f' % (time.time() - start_t))
        # validate if desired
        if epoch % cfg.val_every == 0:
            Logger.log('Validating...')
            with torch.no_grad():
                model.eval()
                step_counter, mean_eval_loss = run_one_epoch(val_loader, model, map_env, loss_fn, device, cfg.out,
                                                            train=False,
                                                            step_counter=step_counter,
                                                            use_wandb=use_wandb)
                if mean_eval_loss < min_eval_loss:
                    Logger.log('Lowest eval loss so far! Saving checkpoint...')
                    min_eval_loss = mean_eval_loss
                    save_file = os.path.join(ckpts_path, 'best_eval_model.pth')
                    save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
                    if use_wandb:
                        wandb.save(save_file)
                torch.cuda.empty_cache()

        # save checkpoint if desired
        if epoch % cfg.save_every == 0:
            Logger.log('Saving checkpoint...')
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model.pth' % (epoch))
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
            save_file = os.path.join(ckpts_path, 'latest_model.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
            if use_wandb:
                wandb.save(save_file)

    if use_wandb:
        # save full log after training
        wandb.save(log_path)

if __name__ == "__main__":
    main()
