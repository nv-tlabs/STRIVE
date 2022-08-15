# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, time
import gc
import tqdm
import torch
import torch.optim as optim
from torch import nn
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch as GraphBatch

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from losses.traffic_model import compute_coll_rate_env, compute_coll_rate_veh

from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, load_state
from utils.scenario_gen import determine_feasibility_nusc, detach_embed_info
from utils.scenario_gen import viz_optim_results, prepare_output_dict
from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
from utils.sol_optim import run_find_solution_optim, compute_sol_success
from utils.init_optim import run_init_optim
from planners.planner import PlannerConfig
from utils.config import get_parser, add_base_args

def parse_cfg():
    parser = get_parser('Adversarial scenario generation')
    parser = add_base_args(parser)

    # data
    parser.add_argument('--split', type=str, default='val',
                        choices=['test', 'val', 'train'],
                        help='Which split of the dataset to find scnarios in')
    parser.add_argument('--val_size', type=int, default=400, help='The size of the validation set used to split the trainval version of the dataset.')
    parser.add_argument('--seq_interval', type=int, default=10, help='skips ahead this many steps from start of current sequence to get the next sequence.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help="Shuffle data")
    parser.set_defaults(shuffle=False)

    parser.add_argument('--adv_attack_with', type=str, default=None,
                        choices=['pedestrian', 'cyclist', 'motorcycle', 'car', 'truck'],
                        help='what to attack with (optional - by default will use any kind of agent)')

    # which planner to attack
    parser.add_argument('--planner', type=str, default='ego',
                        choices=['ego', 'hardcode'],
                        help='Which planner to attack. ego is will use ego motion from nuscenes dataset (i.e. the replay planner).')
    parser.add_argument('--planner_cfg', type=str, default='default',
                        help='hyperparameter configuration to use for the planner (if relevant)')

    # determining feasibility
    parser.add_argument('--feasibility_thresh', type=float, default=10.0, help='Future samples for target must be within this many meters from another agent for the initialization scenario to be feasible.')
    parser.add_argument('--feasibility_time', type=int, default=4, help='For feasibility, only consider timesteps >= feasibility_time, i.e., do not try to crash at timestep 0.')
    parser.add_argument('--feasibility_vel', type=float, default=0.5, help='maximum velocity (delta position of one timestep) of sampled trajectory for an agent must be >= this thresh to be considered feasible')
    parser.add_argument('--feasibility_infront_min', type=float, default=0.0, help='threshold for how in-front-of the ego vehicle the attacker is (measured by cosine similarity).')
    parser.add_argument('--feasibility_check_sep', dest='feasibility_check_sep', action='store_true',
                        help="If given, ensures attacker and target on not separated by non-drivable area.")
    parser.set_defaults(feasibility_check_sep=False)

    # optimizer & losses
    # initialization optimization
    parser.add_argument('--init_loss_match_ext', type=float, default=10.0, help='Match initial trajectory from nuScenes data.')
    parser.add_argument('--init_loss_motion_prior_ext', type=float, default=0.1, help='Keep latent z likely under the traffic model prior.')
    # adversarial optimization
    parser.add_argument('--loss_coll_veh', type=float, default=20.0, help='Loss to avoid vehicle-vehicle collisions between non-planner agents.')
    parser.add_argument('--loss_coll_veh_plan', type=float, default=20.0, help='Loss to avoid collisions between the planner and unlikely adversaries.')
    parser.add_argument('--loss_coll_env', type=float, default=20.0, help='Loss to avoid vehicle-environment collisions for non-planner agents.')
    parser.add_argument('--loss_init_z', type=float, default=0.5, help='Loss to keep latent z near init for unlikely adversaries (i.e. the MAX weight of init loss).')
    parser.add_argument('--loss_init_z_atk', type=float, default=0.05, help='Loss to keep latent z near init for likely adversaries (i.e. the MIN weight of init loss).')
    parser.add_argument('--loss_motion_prior', type=float, default=1.0, help='Loss to keep latent z likely under motion prior for unlikely adversaries (i.e. the MAX weight of prior loss).')
    parser.add_argument('--loss_motion_prior_atk', type=float, default=0.005, help='Loss to keep latent z likely under motion prior for likely adversaries (i.e. the MIN weight of prior loss).')
    parser.add_argument('--loss_motion_prior_ext', type=float, default=0.0001, help='Loss to keep latent z likely under motion prior for the planner.')
    parser.add_argument('--loss_match_ext', type=float, default=10.0, help='Match predicted planner trajectory to true planner rollout.')
    parser.add_argument('--loss_adv_crash', type=float, default=2.0, help='Minimize distance between planner and adversaries.')
    # solution optimization
    parser.add_argument('--sol_future_len', type=int, default=16, help='The number of timesteps to roll out to compute collision losses for solution. If > model/data future len, will avoid irrecoverable final states for solution.')
    parser.add_argument('--sol_loss_coll_veh', type=float, default=10.0, help='Loss to avoid planner-vehicle collisions.')
    parser.add_argument('--sol_loss_coll_env', type=float, default=10.0, help='Loss to avoid planner-environment collisions.')
    parser.add_argument('--sol_loss_motion_prior', type=float, default=0.005, help='Loss to keep planner z likely under the motion prior.')
    parser.add_argument('--sol_loss_init_z', type=float, default=0.0, help='Loss to keep planner z near output of adv optim.')
    parser.add_argument('--sol_loss_motion_prior_ext', type=float, default=0.001, help='Loss to keep non-planner z near output of adv optim.')
    parser.add_argument('--sol_loss_match_ext', type=float, default=10.0, help='Match trajectories from output of adv optim for non-planner agents.')

    parser.add_argument('--num_iters', type=int, default=300, help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for adam.')

    parser.add_argument('--viz', dest='viz', action='store_true',
                        help="If given, saves low-quality visualization before and after optimization.")
    parser.set_defaults(viz=False)
    parser.add_argument('--save', dest='save', action='store_true',
                        help="If given, saves the scenarios as json so they can be used later.")
    parser.set_defaults(save=False)

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict

def run_one_epoch(data_loader, batch_size, model, map_env, device, out_path, loss_weights,
                  planner_name=None,
                  planner_cfg='default',
                  feasibility_thresh=10.0,
                  feasibility_time=4,
                  feasibility_vel=0.5,
                  feasibility_infront_min=0.0,
                  feasibility_check_sep=True,
                  sol_future_len=16,
                  num_iters=300,
                  lr=0.05,
                  viz=True,
                  save=True,
                  adv_attack_with=None
                  ):
    '''
    Run through dataset and find possible scenarios.
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

    data_idx = 0
    empty_cache = False
    batch_i = []
    batch_scene_graph = []
    batch_map_idx = []
    batch_total_NA = 0
    for i, data in enumerate(pbar_data):
        start_t = time.time()
        sample_pred = None
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
            print([map_env.map_list[map_idx[b]] for b in range(map_idx.size(0))])
            is_last_batch = i == (len(data_loader)-1)

            # First sample prior to get possible futures
            with torch.no_grad():
                sample_pred = model.sample_batched(scene_graph, map_idx, map_env, 20, include_mean=True)
                # sample_pred = model.sample(scene_graph, map_idx, map_env, 20, include_mean=True)

            empty_cache = True
            # determine if this sequence is feasible for scenario generation
            feasible, feasible_time, feasible_dist = determine_feasibility_nusc(sample_pred['future_pred'],
                                                                                model.get_normalizer(),
                                                                                feasibility_thresh,
                                                                                feasibility_time,
                                                                                0.0,
                                                                                feasibility_infront_min=feasibility_infront_min,
                                                                                check_non_drivable_separation=feasibility_check_sep,
                                                                                map_env=map_env,
                                                                                map_idx=map_idx)

            if planner_name == 'ego':
                # make sure the ego "planner" has max velocity above some thresh for
                # it to be considered an interesting scenario
                ego_gt = model.get_normalizer().unnormalize(scene_graph.future_gt[0])
                ego_vels = torch.norm(ego_gt[1:, :2] - ego_gt[:-1, :2], dim=-1)
                max_vel = torch.max(ego_vels).cpu().item()
                if max_vel < feasibility_vel:
                    Logger.log('Ego vehicle not moving more than velocity threshold, skipping...')
                    if not is_last_batch:
                        continue
            elif planner_name == 'hardcode':
                # make sure some sample of the ego went over the velocity thresh
                #   so with some confidence it will be an interesting scenario
                ego_samps = model.get_normalizer().unnormalize(sample_pred['future_pred'][0].detach()) # NS x FT x 4
                ego_vels = torch.norm(ego_samps[:, 1:, :2] - ego_samps[:, :-1, :2], dim=-1) # NS x FT-1
                max_vel = torch.max(ego_vels).cpu().item()
                if max_vel < feasibility_vel:
                    Logger.log('Ego samples not moving more than velocity threshold, skipping...')
                    if not is_last_batch:
                        continue

            if feasible is None:
                Logger.log('Only ego vehicle in scene, skipping...')
                if not is_last_batch:
                    continue
            elif torch.sum(feasible).item() == 0:
                Logger.log('Infeasible, no vehicles near ego, skipping...')
                if not is_last_batch:
                    continue

            is_feas = False
            if feasible is not None and torch.sum(feasible).item() > 0:
                is_feas = True
                if adv_attack_with is not None:
                    # only attack with a specific category
                    feas_sem = scene_graph.sem[1:]
                    veclist = [tuple(feas_sem[aidx].to(int).cpu().numpy().tolist()) for aidx in range(feas_sem.size(0))]
                    is_adv_atk = [data_loader.dataset.vec2cat[curvec] == adv_attack_with for curvec in veclist]
                    adv_atk_feas = torch.zeros_like(feasible)
                    adv_atk_feas[is_adv_atk] = True
                    feasible = torch.logical_and(feasible, adv_atk_feas)
                    if torch.sum(feasible) == 0:
                        Logger.log('No feasible attackers of requested category, skipping...')
                        is_feas = False

                if is_feas:
                    # print which vehicle is the best candidate (for info/debug purposes)
                    feasible_dist[~feasible] = float('inf')
                    temp_attack_agt = torch.min(feasible_dist, dim=0)[1] + 1
                    print('Heuristic attack agt is %d' % (temp_attack_agt))
                    print('Heuristic attack time is %d' % (feasible_time[temp_attack_agt-1]))  

            # This is a feasible seed, add it to the batch
            if is_feas:
                Logger.log('Feasible. Adding to batch...')
                batch_scene_graph += scene_graph.to_data_list()
                batch_map_idx.append(map_idx)
                batch_i.append(i)
                batch_total_NA += scene_graph.future_gt.size(0)
                Logger.log('Current batch NA: %d' % (batch_total_NA))

            if batch_total_NA < batch_size and not is_last_batch:
                # collect more before performing optim
                continue
            else:
                if len(batch_scene_graph) == 0:
                    # this is the last seq in dataset, and we have no other seqs queueued
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
            ego_inds = scene_graph.ptr[:-1]
            ego_mask = torch.zeros((NA), dtype=torch.bool)
            ego_mask[ego_inds] = True
            
            #
            # Initialize optimization
            #
            # embed past and map to get inputs to decoder used during optim
            with torch.no_grad():
                embed_info_attached = model.embed(scene_graph, map_idx, map_env)
            # need to detach all the encoder outputs from current comp graph to be used in optimization
            embed_info = detach_embed_info(embed_info_attached)

            init_future_pred = init_traj = z_init = init_coll_env = None

            planner = plan_out_path = None
            if planner_name == 'hardcode':
                from planners.hardcode_goalcond_nusc import HardcodeNuscPlanner, CONFIG_DICT
                assert(planner_cfg in CONFIG_DICT)
                Logger.log('Using planner config:')
                Logger.log(CONFIG_DICT[planner_cfg])
                planner = HardcodeNuscPlanner(map_env, PlannerConfig(**CONFIG_DICT[planner_cfg])) 

            # start from GT scene future (reconstructed with motion model)
            z_init = embed_info_attached['posterior_out'][0].detach()
            init_traj = scene_graph.future_gt[:, :, :4].clone().detach()
            Logger.log('Running initialization optimization...')

            # run initial optimization to closely fit nuscenes scene
            z_init, init_fit_traj, _ = run_init_optim(z_init, init_traj, scene_graph.future_vis, 0.1, loss_weights, model,
                                                      scene_graph, map_env, map_idx, 75, embed_info, embed_info['prior_out'])
            # if we're using a specific planner, replace ego with planner rollout
            if planner_name == 'hardcode':
                # reset planner
                all_init_state = model.get_normalizer().unnormalize(scene_graph.past_gt[:, -1, :])
                all_init_veh_att = model.get_att_normalizer().unnormalize(scene_graph.lw)
                planner.reset(all_init_state, all_init_veh_att, scene_graph.batch, B, map_idx)
                # rollout
                init_non_ego = model.normalizer.unnormalize(init_fit_traj[~ego_mask]).cpu().numpy()
                plan_t = np.linspace(model.dt, model.dt*model.FT, model.FT)
                init_agt_ptr = scene_graph.ptr - torch.arange(B+1)
                planner_init = planner.rollout(init_non_ego, plan_t, init_agt_ptr.cpu().numpy(), plan_t,
                                                control_all=False).to(scene_graph.future_gt)
                planner_init = model.get_normalizer().normalize(planner_init)
                # replace init traj ego's with planner traj
                init_traj[ego_mask] = planner_init

                # and optim a bit more, now to match the planner traj
                Logger.log('Fine-tune init with planner rollout...')
                z_init, init_fit_traj, _ = run_init_optim(z_init, init_traj, scene_graph.future_vis, lr, loss_weights, model,
                                                            scene_graph, map_env, map_idx, 100, embed_info, embed_info['prior_out'])

                # check if planner collides with scene trajectories already. if so, not worth continuing
                from losses.adv_gen_nusc import check_single_veh_coll
                bvalid = []
                for b in range(B):
                    init_hardcode_coll, _ = check_single_veh_coll(model.get_normalizer().unnormalize(init_fit_traj[scene_graph.ptr[b]]),
                                                                    model.get_att_normalizer().unnormalize(scene_graph.lw[scene_graph.ptr[b]]),
                                                                    model.get_normalizer().unnormalize(init_fit_traj[(scene_graph.ptr[b]+1):scene_graph.ptr[b+1]]),
                                                                    model.get_att_normalizer().unnormalize(scene_graph.lw[(scene_graph.ptr[b]+1):scene_graph.ptr[b+1]])
                                                                    )
                    bvalid.append(np.sum(init_hardcode_coll) == 0)

                bvalid = np.array(bvalid, dtype=np.bool)
                if np.sum(bvalid) < B:
                    Logger.log('Planner already caused collision after init, removing from batch...')
                    if np.sum(bvalid) == 0:
                        Logger.log('No valid sequences left in batch! Skipping...')
                        continue
                    # need to remove invalid scenarios from batch
                    # rebuild and reset all necessary variables
                    map_idx = map_idx[bvalid]
                    cur_batch_i = [bi for b, bi in enumerate(cur_batch_i) if bvalid[b]]

                    avalid = np.zeros((NA), dtype=np.bool) # which agents are part of new graphs
                    for b in range(B):
                        if bvalid[b]:
                            avalid[scene_graph.ptr[b]:scene_graph.ptr[b+1]] = True

                    z_init = z_init[avalid]
                    init_traj = init_traj[avalid]

                    init_batch_data_list = scene_graph.to_data_list()
                    init_batch_data_list = [g for b, g, in enumerate(init_batch_data_list) if bvalid[b]]
                    scene_graph = GraphBatch.from_data_list(init_batch_data_list)

                    B = map_idx.size(0)
                    NA = scene_graph.past.size(0)
                    ego_inds = scene_graph.ptr[:-1]
                    ego_mask = torch.zeros((NA), dtype=torch.bool)
                    ego_mask[ego_inds] = True

                    with torch.no_grad():
                        embed_info_attached = model.embed(scene_graph, map_idx, map_env)
                    embed_info = detach_embed_info(embed_info_attached)

                    Logger.log(scene_graph)

            with torch.no_grad():
                init_future_pred = model.decode_embedding(z_init, embed_info_attached, scene_graph, map_idx, map_env)['future_pred'].detach()
                init_coll_env_dict = compute_coll_rate_env(scene_graph, map_idx, init_future_pred.unsqueeze(1).contiguous(),
                                                    map_env, model.get_normalizer(), model.get_att_normalizer(),
                                                    ego_only=False)
                init_coll_env = init_coll_env_dict['did_collide'].cpu().numpy()[:, 0] # NA

                # make sure ego is actual data or planner rollout - not our initial fitting
                init_future_pred[ego_mask] = init_traj[ego_mask]

            if planner_name == 'hardcode':
                plan_out_path = None
                if viz:
                    plan_out_path = os.path.join(gen_out_path_viz, 'planner_out')
                    cur_seq_str = 'sample_' + '_'.join(['%03d' for b in range(len(cur_batch_i))]) % tuple([cur_batch_i[b] for b in range(len(cur_batch_i))])
                    plan_out_path = os.path.join(plan_out_path, cur_seq_str)
                    mkdir(plan_out_path)

            # adversarial optimization
            cur_z = z_init.clone().detach()
            tgt_prior_distrib = (embed_info['prior_out'][0][ego_mask], embed_info['prior_out'][1][ego_mask])
            other_prior_distrib = (embed_info['prior_out'][0][~ego_mask], embed_info['prior_out'][1][~ego_mask])

            adv_gen_out = run_adv_gen_optim(cur_z, lr, loss_weights, model, scene_graph, map_env, map_idx,
                                            num_iters, embed_info, 
                                            planner_name, tgt_prior_distrib, other_prior_distrib,
                                            feasibility_time, feasibility_infront_min,
                                            planner=planner,
                                            planner_viz_out=plan_out_path)
            cur_z, final_result_traj, final_decoder_out, cur_min_agt, cur_min_t = adv_gen_out
            attack_agt = cur_min_agt
            attack_t = cur_min_t

            adv_succeeded = []
            other_ptr = scene_graph.ptr - torch.arange(len(scene_graph.ptr))
            for b in range(B):
                cur_adv_succeeded = compute_adv_gen_success(final_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                    model,
                                                    GraphBatch.from_data_list([scene_graph.to_data_list()[b]]),
                                                    attack_agt[b] - scene_graph.ptr[b].item())
                adv_succeeded.append(cur_adv_succeeded)

            # build the solution optimization batch (only indices that succeeded)
            print(adv_succeeded)
            batch_graph_list = scene_graph.to_data_list()
            sol_graph_list = [batch_graph_list[b] for b in range(B) if adv_succeeded[b]]
            sol_succeeded = []
            if len(sol_graph_list) > 0:
                Logger.log('Batch adv optim successes:')
                Logger.log(adv_succeeded)

                sol_scene_graph = GraphBatch.from_data_list(sol_graph_list)
                Logger.log('Solution scene graph:')
                Logger.log(sol_scene_graph)

                sol_amask = torch.zeros((NA), dtype=torch.bool) # which agents are part of solution graphs
                sol_bmask = torch.zeros((B), dtype=torch.bool) # which batch indices need a solution
                for b in range(B):
                    if adv_succeeded[b]:
                        sol_amask[scene_graph.ptr[b]:scene_graph.ptr[b+1]] = True
                        sol_bmask[b] = True

                Logger.log('Adv gen succeeded! Finding solution...')
                # collect info for just the batch indices that need solution
                sol_in_final_result_traj = final_result_traj[sol_amask]
                sol_NA = sol_in_final_result_traj.size(0)
                sol_ego_inds = sol_scene_graph.ptr[:-1]
                sol_ego_mask = torch.zeros((sol_NA), dtype=torch.bool)
                sol_ego_mask[sol_ego_inds] = True
                sol_in_cur_z = cur_z.clone().detach()[sol_amask]
                sol_map_idx = map_idx[sol_bmask]
                sol_embed_info = dict()
                for k, v in embed_info.items():
                    if isinstance(v, torch.Tensor):
                        # everything else
                        sol_embed_info[k] = v[sol_amask]
                    elif isinstance(v, tuple):
                        # posterior or prior output
                        sol_embed_info[k] = (v[0][sol_amask], v[1][sol_amask])
                sol_tgt_prior_distrib = (tgt_prior_distrib[0][sol_bmask], tgt_prior_distrib[1][sol_bmask])
                sol_other_prior_distrib = (sol_embed_info['prior_out'][0][~sol_ego_mask], sol_embed_info['prior_out'][1][~sol_ego_mask])

                sol_z, sol_result_traj, sol_decoder_out = run_find_solution_optim(sol_in_cur_z, sol_in_final_result_traj, sol_future_len, 
                                                                                    lr, loss_weights, model, sol_scene_graph, map_env, sol_map_idx,
                                                                                    num_iters, sol_embed_info,
                                                                                    sol_tgt_prior_distrib, sol_other_prior_distrib)

                for b in range(sol_map_idx.size(0)):
                    cur_sol_succeeded = compute_sol_success(sol_result_traj[sol_scene_graph.ptr[b]:sol_scene_graph.ptr[b+1]][:,0:1],
                                                        model,
                                                        GraphBatch.from_data_list([sol_scene_graph.to_data_list()[b]]),
                                                        map_env,
                                                        sol_map_idx[b:b+1])
                    sol_succeeded.append(cur_sol_succeeded)

                cur_sidx = 0
                final_sol_suceeded = [False]*B
                for b in range(B):
                    if adv_succeeded[b]:
                        final_sol_suceeded[b] = sol_succeeded[cur_sidx]
                        cur_sidx += 1
                sol_succeeded = final_sol_suceeded

            print(sol_succeeded)

            Logger.log('Optimized sequence in %f sec!' % (time.time() - start_t))

            # output scenario and viz
            cur_sidx = 0
            scene_graph_list = scene_graph.to_data_list()
            for b in range(B):
                result_dir = None
                if not adv_succeeded[b]:
                    result_dir = 'adv_failed'
                else:
                    if sol_succeeded[b]:
                        result_dir = 'adv_sol_success'
                    else:
                        result_dir = 'sol_failed'

                cur_attack_agt = attack_agt[b] - scene_graph.ptr[b].item() # make index local to each batch idx
                if save:
                    import json
                    # save scenario
                    cur_scene_out_path = os.path.join(gen_out_path_scenes, result_dir)
                    mkdir(cur_scene_out_path)

                    out_sol_traj = sol_result_traj[sol_scene_graph.ptr[cur_sidx]:sol_scene_graph.ptr[cur_sidx+1]][:,0] if adv_succeeded[b] else None
                    out_sol_z = sol_z[sol_scene_graph.ptr[cur_sidx]:sol_scene_graph.ptr[cur_sidx+1]] if out_sol_traj is not None else None
                    scene_out_dict = prepare_output_dict(scene_graph_list[b], map_idx[b].item(), map_env, data_loader.dataset.dt, model,
                                                          init_future_pred[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          final_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]][:,0],
                                                          out_sol_traj,
                                                          cur_attack_agt,
                                                          attack_t[b],
                                                          cur_z[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          out_sol_z,
                                                          (embed_info['prior_out'][0][scene_graph.ptr[b]:scene_graph.ptr[b+1]], embed_info['prior_out'][1][scene_graph.ptr[b]:scene_graph.ptr[b+1]]),
                                                          internal_ego_traj=final_decoder_out['future_pred'][scene_graph.ptr[b]].detach()
                                                          )

                    fout_path = os.path.join(cur_scene_out_path, 'scene_%04d.json' % cur_batch_i[b])
                    Logger.log('Saving scene to %s' % (fout_path))
                    with open(fout_path, 'w') as writer:
                        json.dump(scene_out_dict, writer)

                if viz:
                    cur_viz_out_path = os.path.join(gen_out_path_viz, result_dir)
                    mkdir(cur_viz_out_path)

                    # save before viz
                    cur_crop_t = attack_t[b]
                    pred_prefix = 'test_sample_%d_before' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        init_future_pred, planner_name, cur_attack_agt,
                                        cur_crop_t,
                                        bidx=b,
                                        show_gt=True, # show entire nuscenes scene
                                        ow_gt=init_traj)

                    # save after optimization viz
                    pred_prefix = 'test_sample_%d_after' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        final_result_traj, planner_name, cur_attack_agt, cur_crop_t,
                                        bidx=b,
                                        show_gt_idx=0,
                                        ow_gt=final_decoder_out['future_pred'].clone().detach()) # show our internal pred of planner as "gt" since final_result_traj is actual planner traj

                    if adv_succeeded[b]:
                        pred_prefix = 'test_sample_%d_sol' % (cur_batch_i[b])
                        pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                        viz_optim_results(pred_out_path, sol_scene_graph, sol_map_idx, map_env, model,
                                        sol_result_traj, planner_name, cur_attack_agt, cur_crop_t,
                                        bidx=cur_sidx,
                                        show_gt_idx=0,
                                        ow_gt=final_result_traj[sol_amask][:, 0]) # show the failed planner path pre-sol as "GT"
                
                if adv_succeeded[b]:
                    cur_sidx += 1

        except RuntimeError as e:
            Logger.log('Caught error in optim batch %s!' % (str(e)))
            Logger.log('Skipping')
            raise e
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            empty_cache = True
            continue

def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'adv_gen_log.txt')
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
                            load_lanegraph=(cfg.planner=='hardcode'),
                            lanegraph_res_meters=1.0
                            )
    test_dataset = NuScenesDataset(data_path, map_env,
                            version=cfg.data_version,
                            split=cfg.split,
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            seq_interval=cfg.seq_interval,
                            randomize_val=True,
                            val_size=cfg.val_size,
                            reduce_cats=cfg.reduce_cats
                            )

    # create loaders    
    test_loader = GraphDataLoader(test_dataset,
                                    batch_size=1, # will collect batches on the fly after determining feasibility
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

    # load model weights
    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to do scenario generation!')

    # so can unnormalize as needed
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    loss_weights = {
        'coll_veh' : cfg.loss_coll_veh,
        'coll_veh_plan' : cfg.loss_coll_veh_plan,
        'coll_env' : cfg.loss_coll_env,
        'motion_prior' : cfg.loss_motion_prior,
        'motion_prior_atk' : cfg.loss_motion_prior_atk,
        'init_z' : cfg.loss_init_z,
        'init_z_atk': cfg.loss_init_z_atk,
        'motion_prior_ext' : cfg.loss_motion_prior_ext,
        'match_ext' : cfg.loss_match_ext,
        'adv_crash' : cfg.loss_adv_crash,
        'sol_coll_veh' : cfg.sol_loss_coll_veh,
        'sol_coll_env' : cfg.sol_loss_coll_env,
        'sol_motion_prior' : cfg.sol_loss_motion_prior,
        'sol_init_z' : cfg.sol_loss_init_z,
        'sol_motion_prior_ext' : cfg.sol_loss_motion_prior_ext,
        'sol_match_ext' : cfg.sol_loss_match_ext,
        'init_match_ext' : cfg.init_loss_match_ext,
        'init_motion_prior_ext' : cfg.init_loss_motion_prior_ext
    }

    # run through dataset once and generate possible scenarios
    model.train()
    run_one_epoch(test_loader, cfg.batch_size, model, map_env, device, cfg.out, loss_weights,
                  planner_name=cfg.planner, 
                  planner_cfg=cfg.planner_cfg,
                  feasibility_thresh=cfg.feasibility_thresh,
                  feasibility_time=cfg.feasibility_time,
                  feasibility_vel=cfg.feasibility_vel,
                  feasibility_infront_min=cfg.feasibility_infront_min,
                  feasibility_check_sep=cfg.feasibility_check_sep,
                  sol_future_len=cfg.sol_future_len,
                  num_iters=cfg.num_iters,
                  lr=cfg.lr,
                  viz=cfg.viz,
                  save=cfg.save,
                  adv_attack_with=cfg.adv_attack_with)


if __name__ == "__main__":
    main()
