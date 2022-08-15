# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, argparse, time, glob
import json
import configargparse
import pickle
import csv
import torch
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from datasets.map_env import NuScenesMapEnv
import datasets.nuscenes_utils as nutils
from utils.common import dict2obj, mkdir
from utils.logger import Logger
from utils.scenario_gen import log_metric, log_freq_stat
from losses.adv_gen_nusc import interp_traj, check_single_veh_coll
from utils.transforms import transform2frame


def parse_cfg():
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      config_file_parser_class=configargparse.YAMLConfigFileParser,
                                      description='Test motion model')
    # logging
    parser.add_argument('--out', type=str, default='./out/eval_adv_gen_out',
                        help='Directory to save model weights and logs to.')

    # scenarios to cluster/predict
    parser.add_argument('--scenarios', type=str, required=True,
                        help='Directory to load scenarios from, should contain adv_failed, adv_sol_success, and sol_failed')
    # which clustering to use for labeling
    parser.add_argument('--cluster_path', default='./data/clustering/cluster.pkl', type=str,
                        help="path to cluster object. If given, uses this clustering to label the given scenarios, does NOT redo the clustering.")
    parser.add_argument('--cluster_labels', default='./data/clustering/cluster_labels.txt', type=str,
                        help="if loading in cluster_path, can provide semantic labels for each cluster from a txt file. This is just a sequence of labels separated by commas.")

    parser.add_argument('--data_dir', type=str, default='./data/nuscenes',
                        help='Directory to load data from.')
    parser.add_argument('--data_version', type=str, default='mini',
                        choices=['trainval', 'mini'], help='Whether to use full nuscenes or mini.')

    # which evaluations to perform
    parser.add_argument('--eval_quant', dest='eval_quant', action='store_true', help="Quantitative evaluation")
    parser.set_defaults(eval_quant=False)
    parser.add_argument('--eval_qual', dest='eval_qual', action='store_true', help="Qualitative evaluation")
    parser.set_defaults(eval_qual=False)

    # viz options
    parser.add_argument('--viz_res', type=str, nargs='+', default=['adv_sol_success'], 
                        choices=['adv_sol_success', 'sol_failed', 'adv_failed'], help='Which results to visualize.')
    parser.add_argument('--viz_stage', type=str, nargs='+', default=['adv'], 
                        choices=['adv', 'init', 'sol'], help='Which stages of the pipeline to visualize.')
    parser.add_argument('--viz_video', dest='viz_video', action='store_true', help="whether to viz video")
    parser.set_defaults(viz_video=False)

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict

def read_adv_scenes(scene_path):
    scene_flist = sorted(glob.glob(os.path.join(scene_path, '*.json')))    
    scene_list = []
    for scene_fpath in scene_flist:
        scene_name = scene_fpath.split('/')[-1][:-5]
        jdict = None
        with open(scene_fpath, 'r') as f:
            jdict = json.load(f)
        if jdict is None:
            print('Failed to load! Skipping')
            continue
        
        cur_scene = {
            'name' : scene_name,
            'map' : jdict['map'],
            'dt' : jdict['dt']
        }
        cur_scene['sem'] = torch.tensor(jdict['sem'])
        cur_scene['veh_att'] = torch.tensor(jdict['lw'])
        cur_scene['past'] = torch.tensor(jdict['past'])
        cur_scene['fut_adv'] = torch.tensor(jdict['fut_adv'])
        cur_scene['fut_init'] = torch.tensor(jdict['fut_init'])
        if 'fut_sol' in jdict:
            cur_scene['fut_sol'] = torch.tensor(jdict['fut_sol'])
        if 'fut_internal_ego' in jdict:
            cur_scene['fut_internal_ego'] = torch.tensor(jdict['fut_internal_ego'])
        if 'attack_t' in jdict:
            cur_scene['attack_t'] = jdict['attack_t']
        if 'attack_agt' in jdict:
            cur_scene['attack_agt'] = jdict['attack_agt']
        if 'z_adv' in jdict:
            cur_scene['z_adv'] = torch.tensor(jdict['z_adv']) # latents of the adversarial scene rep in motion model
        if 'z_sol' in jdict:
            cur_scene['z_sol'] = torch.tensor(jdict['z_sol']) # latents of the solution scene rep in motion model
        if 'z_prior' in jdict:
            cur_scene['z_prior_mean'] = torch.tensor(jdict['z_prior']['mean'])
            cur_scene['z_prior_var'] = torch.tensor(jdict['z_prior']['var'])

        scene_list.append(cur_scene)

    return scene_list

def compute_coll_feat(lw, scene_traj, dt):
    planner_traj = scene_traj[0]
    other_traj = scene_traj[1:]
    planner_lw = lw[0]
    other_lw = lw[1:]

    # interpolate to find better approx of exact time/state of collision
    interp_scale = 5
    interp_dt = dt / float(interp_scale)
    planner_fut_interp = interp_traj(planner_traj.unsqueeze(0), scale_factor=interp_scale)[0]
    other_fut_interp = interp_traj(other_traj, scale_factor=interp_scale)
    planner_coll_all_interp, planner_coll_time_interp = check_single_veh_coll(planner_fut_interp,
                                                                                planner_lw,
                                                                                other_fut_interp,
                                                                                other_lw)

    # get attacker in local frame of target at collision times
    planner_coll_time_interp = planner_coll_time_interp[planner_coll_all_interp]
    planner_coll_agt_interp = np.nonzero(planner_coll_all_interp)[0]
    plan_coll_states = planner_fut_interp[planner_coll_time_interp] # (NC, 4)
    atk_coll_states = other_fut_interp[planner_coll_all_interp, planner_coll_time_interp] # (NC, 4)

    min_coll_t = np.amin(planner_coll_time_interp)
    min_coll_idx = np.argmin(planner_coll_time_interp) # idx of the collision that happened first
    min_coll_agt = planner_coll_agt_interp[min_coll_idx]
    # for earliest collision
    local_atk_states = transform2frame(plan_coll_states, atk_coll_states.unsqueeze(1))[min_coll_idx,0] # (4)
    plan_coll_states = plan_coll_states[min_coll_idx]

    # relative heading and position (converted to angle in local frame)
    coll_h = torch.atan2(local_atk_states[3], local_atk_states[2]).item()
    coll_hvec = [local_atk_states[2].item(), local_atk_states[3].item()]
    coll_pos = local_atk_states[:2] / torch.norm(local_atk_states[:2], dim=0)
    coll_ang = torch.atan2(coll_pos[1], coll_pos[0]).item()

    # relative vel
    lr_coll_t = int((min_coll_t*interp_dt) / dt) # collision time in OG data
    if lr_coll_t > 0:
        atk_vel = (other_traj[min_coll_agt, lr_coll_t, :2] - other_traj[min_coll_agt, lr_coll_t-1, :2]) / dt
        plan_vel = (planner_traj[lr_coll_t, :2] - planner_traj[lr_coll_t-1, :2]) / dt
    else:
        atk_vel = (other_traj[min_coll_agt, lr_coll_t+1, :2] - other_traj[min_coll_agt, lr_coll_t, :2]) / dt
        plan_vel = (planner_traj[lr_coll_t+1, :2] - planner_traj[lr_coll_t, :2]) / dt
    rel_vel = plan_vel - atk_vel
    coll_rel_s = torch.norm(rel_vel).item()

    feat = {
        'hvec': coll_hvec,
        'angvec' : coll_pos.numpy().tolist(),
        'rel_s' : coll_rel_s,
    }

    return feat

def plot_scenario_distrib(clustering, cluster_labels, adv_sol_success, sol_failed, out_path):
    all_labels = np.unique(clustering.labels_)
    sort_inds = np.argsort(np.array(cluster_labels))
    cluster_labels = np.array(cluster_labels)[sort_inds]

    fig = plt.figure(dpi=200)
    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)
    dir_sidx = 0
    clust_cnt_list = []
    total_clust_count = np.zeros_like(all_labels, dtype=int)
    for cur_scenes in [adv_sol_success, sol_failed]: # plot each directory independently
        cur_labels = np.array([cur_scenes[si]['label_idx'] for si in range(len(cur_scenes))])
        clust_count = np.zeros_like(all_labels, dtype=int)
        for li in range(clust_count.shape[0]):
            cnt = np.sum(cur_labels == all_labels[li])
            clust_count[li] = cnt

        clust_count = clust_count[sort_inds]
        total_clust_count += clust_count
        clust_cnt_list.append(clust_count)

    y_pos = np.arange(len(total_clust_count))
    y_off = np.linspace(-0.2, 0.2, len(clust_cnt_list))[::-1]
    w = y_off[1] - y_off[0]
    legend_list = ['Solution Found', 'No Solution']
    color_list = ['orange', 'red']
    for ci, clust_count in enumerate(clust_cnt_list): # plot each directory independently
        plt.barh(y_pos+y_off[ci], clust_count, w, color=color_list[ci], align='center', label=legend_list[ci])

    plt.yticks(y_pos, tuple(cluster_labels))
    plt.legend()
    plt.xlabel('Count')
    plt.title('Collision Scenario Distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def assign_cluster(scene_list, clustering, cluster_labels, csv_out_path=None):
    feat_list = []
    print('Collecting scene features...')
    for si, scene in enumerate(scene_list):
        lw = scene['veh_att']
        traj = scene['fut_adv']
        coll_feats = compute_coll_feat(lw, traj, scene['dt'])
        feat_list.append(coll_feats)

    # form full feature array (NC, D) and assign
    angvec = np.array([feat['angvec'] for feat in feat_list])
    hvec = np.array([feat['hvec'] for feat in feat_list])
    scene_feats = np.concatenate([angvec, hvec], axis=1)
    scene_labels = clustering.predict(scene_feats)

    for si, scene in enumerate(scene_list):
        scene['label'] = cluster_labels[scene_labels[si]]
        scene['label_idx'] = scene_labels[si]
    
    # write to csv file
    if csv_out_path is not None:
        with open(csv_out_path, 'w') as f:
            csvwrite = csv.writer(f)
            csvwrite.writerow(['scene', 'cluster_idx', 'cluster_name'])
            for sidx, scene in enumerate(scene_list):
                currow = [scene['name'], scene_labels[sidx], scene['label']]
                csvwrite.writerow(currow)

    return feat_list

def quant_eval(scenarios, cluster_path, cluster_labels, map_env, out_path):

    # assign collision scenarios to cluster
    # load in already fitted cluster object
    with open(cluster_path, 'rb') as f:
        clustering = pickle.load(f)
    with open(cluster_labels, 'r') as f:
        line = f.readlines()[0]
    cluster_labels = line.split(',')
    cluster_labels = [label.strip() for label in cluster_labels]

    coll_feat_list = []
    for coll_sname in ['adv_sol_success', 'sol_failed']:
        if len(scenarios[coll_sname]) == 0:
            continue
        csv_out_path = os.path.join(out_path, coll_sname + '_labels.csv')
        # updates scenes in place
        coll_feat_list += assign_cluster(scenarios[coll_sname], clustering, cluster_labels, csv_out_path)

    # plot label distribution comparion success and failed
    plot_scenario_distrib(clustering, cluster_labels, scenarios['adv_sol_success'], scenarios['sol_failed'],
                            os.path.join(out_path, 'scene_distrib.png'))

    adv_success_rate, sol_success_rate = compute_success_rates(scenarios)
    tot_success_rate = adv_success_rate * sol_success_rate

    metrics = {}
    freq_metrics_cnt = {}
    freq_metrics_total = {}
    # accumulate scenarios starting from successful only, then solution failed, then no-collision scenarios
    coll_sidx = 0
    resnames = ['adv_sol_success', 'sol_failed', 'adv_failed']
    for residx, resname in enumerate(resnames):
        for scene in scenarios[resname]:
            map_idx = map_env.map_list.index(scene['map'])
            metric_res = compute_metrics(scene, map_env, map_idx, metrics, freq_metrics_cnt, freq_metrics_total)
            metrics, freq_metrics_cnt, freq_metrics_total, seq_metrics = metric_res
            if resname in ['adv_sol_success', 'sol_failed']:
                sol_success = resname == 'adv_sol_success'
                freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                                    'sol_success', int(sol_success), 1)
                seq_metrics['sol_success'] = int(sol_success)

                coll_sidx += 1
            else:
                seq_metrics['sol_success'] = np.nan

            scene['eval_metrics'] = seq_metrics

        eval_name = None
        if resname == 'adv_sol_success':
            eval_name = 'adv_sol'
        elif resname == 'sol_failed':
            eval_name = 'all_adv'
        elif resname == 'adv_failed':
            eval_name = 'all_scenes'
        # per-seq csv
        per_seq_metrics = sorted(list(scenarios['adv_sol_success'][0]['eval_metrics'].keys()))
        per_seq_out_path = os.path.join(out_path, 'eval_per_seq_' + eval_name + '.csv')
        with open(per_seq_out_path, 'w') as f:
            csvwrite = csv.writer(f)
            header = ['name'] + per_seq_metrics
            csvwrite.writerow(header)
            for ridx in range(residx+1):
                for scene in scenarios[resnames[ridx]]:
                    data = [scene['name']]
                    data += [scene['eval_metrics'][k] for k in per_seq_metrics]
                    csvwrite.writerow(data)
        # total csv
        total_out_path = os.path.join(out_path, 'eval_total_' + eval_name + '.csv')
        with open(total_out_path, 'w') as f:
            csvwrite = csv.writer(f)
            header = ['adv_success', 'sol_success', 'tot_success'] + per_seq_metrics
            csvwrite.writerow(header)
            data = [adv_success_rate, sol_success_rate, tot_success_rate]
            for k in per_seq_metrics:
                if k in metrics:
                    data.append(np.mean(metrics[k]))
                if k in freq_metrics_cnt:
                    data.append(float(freq_metrics_cnt[k]) / freq_metrics_total[k])
                if k not in metrics and k not in freq_metrics_cnt:
                    data.append('')
            csvwrite.writerow(data)


def compute_accels(pos_traj, h_traj, dt):
    pre_crash_pos = pos_traj
    pre_crash_head = h_traj
    planner_vel = (pre_crash_pos[1:] - pre_crash_pos[:-1]) / dt
    planner_s = torch.norm(planner_vel, dim=-1)
    unit_head = pre_crash_head / torch.norm(pre_crash_head, dim=-1, keepdim=True)
    planner_vel = planner_s.unsqueeze(1) * unit_head[:-1]
    plan_fwd_accel = torch.abs((planner_s[1:] - planner_s[:-1]) / dt) 
    planner_accel = (planner_vel[1:] - planner_vel[:-1]) / dt
    lat_dir = torch.cat([-unit_head[:-2, 1:2], unit_head[:-2, 0:1]], dim=1)
    plan_lat_accel = torch.sum(planner_accel * lat_dir, dim=-1) # project accel onto lateral dir
    plan_lat_accel = torch.abs(plan_lat_accel)
    planner_accel = torch.norm(planner_accel, dim=-1)

    return planner_accel, plan_fwd_accel, plan_lat_accel

def compute_metrics(scene, map_env, map_idx, metrics, freq_metrics_cnt, freq_metrics_total):
    from losses.adv_gen_nusc import check_single_veh_coll, check_pairwise_veh_coll
    from losses.common import log_normal
    from losses.traffic_model import compute_coll_rate_env_from_traj

    seq_metrics = dict()

    veh_att = scene['veh_att']
    fut_adv = scene['fut_adv']
    atk_agt = scene['attack_agt'] # the agent being "most controlled", not necessarily the one that ends up colliding though...

    fut_sol = None
    if 'fut_sol' in scene:
        fut_sol = scene['fut_sol']

    planner_gt_fut = fut_adv[0]
    planner_pred_fut = None
    if 'fut_internal_ego' in scene:
        planner_pred_fut = scene['fut_internal_ego']
    other_fut = fut_adv[1:]
    planner_lw = veh_att[0]
    other_lw = veh_att[1:]

    atk_mask = torch.zeros((fut_adv.size(0)), dtype=torch.bool)
    atk_mask[atk_agt] = True
    other_mask = ~atk_mask.clone()
    other_mask[0] = False # leave out ego in addition to attacker

    # get time of collision
    planner_coll_all, planner_coll_time = check_single_veh_coll(planner_gt_fut, planner_lw, other_fut, other_lw)
    did_collide = np.sum(planner_coll_all) > 0
    coll_t = np.amin(planner_coll_time)
    coll_agt = np.argmin(planner_coll_time) + 1 # indexes into full scene NA

    if did_collide:
        # update to be the agent that collides
        atk_agt = coll_agt
        atk_mask = torch.zeros((fut_adv.size(0)), dtype=torch.bool)
        atk_mask[atk_agt] = True
        other_mask = ~atk_mask.clone()
        other_mask[0] = False # leave out ego in addition to attacker

    freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                            'adv_collide', int(did_collide), 1)
    seq_metrics['adv_collide'] = int(did_collide)

    CT = coll_t

    if CT > 0:
        # Collisions between any non-planner agents
        controlled_coll_dict = check_pairwise_veh_coll(other_fut[:, :CT], other_lw)
        freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                            'veh_coll_rate', int(controlled_coll_dict['num_coll_veh']),
                                                            int(controlled_coll_dict['num_traj_veh']))
        seq_metrics['veh_coll_rate'] = float(controlled_coll_dict['num_coll_veh']) / float(controlled_coll_dict['num_traj_veh'])

        #
        # Collisions with environment for the attacker/others
        #
        mapixes = map_idx*torch.ones((fut_adv.size(0)), dtype=torch.long)
        coll_env_dict = compute_coll_rate_env_from_traj(fut_adv[:,:CT].unsqueeze(1), veh_att, mapixes, map_env)
        coll_env = coll_env_dict['did_collide'].cpu().numpy()[:, 0] # NA

        # attacker
        atk_env_coll = coll_env[atk_agt]
        freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                                'env_coll_atk', int(atk_env_coll), 1)
        seq_metrics['env_coll_atk'] = int(atk_env_coll)
        # others
        if torch.sum(other_mask) > 0:
            atk_env_others = np.sum(coll_env[other_mask])
            freq_metrics_cnt, freq_metrics_total = log_freq_stat(freq_metrics_cnt, freq_metrics_total, 
                                                                    'env_coll_others', int(atk_env_others), 
                                                                    torch.sum(other_mask).item())
            seq_metrics['env_coll_others'] = float(atk_env_others) / torch.sum(other_mask).item()
        else:
            seq_metrics['env_coll_others'] = np.nan
    else:
        seq_metrics['veh_coll_rate'] = np.nan
        seq_metrics['env_coll_atk'] = np.nan
        seq_metrics['env_coll_others'] = np.nan

    #
    # Comfort (accelerations)
    #

    # attacker
    atk_pre_pos = fut_adv[atk_agt, :CT, :2]
    atk_pre_h = fut_adv[atk_agt, :CT, 2:4]
    if atk_pre_pos.size(0) > 2:
        atk_accel, atk_accel_fwd, atk_accel_lat = compute_accels(atk_pre_pos, atk_pre_h, scene['dt'])
        metrics = log_metric(metrics, 'adv_atk_accel', atk_accel.numpy())
        seq_metrics['adv_atk_accel'] = atk_accel.mean().item()
        metrics = log_metric(metrics, 'adv_atk_accel_fwd', atk_accel_fwd.numpy())
        seq_metrics['adv_atk_accel_fwd'] = atk_accel_fwd.mean().item()
        metrics = log_metric(metrics, 'adv_atk_accel_lat', atk_accel_lat.numpy())
        seq_metrics['adv_atk_accel_lat'] = atk_accel_lat.mean().item()
    else:
        seq_metrics['adv_atk_accel'] = np.nan
        seq_metrics['adv_atk_accel_fwd'] = np.nan
        seq_metrics['adv_atk_accel_lat'] = np.nan

    # others (non-attackers)
    others_pre_pos = fut_adv[other_mask, :CT, :2]
    others_pre_h = fut_adv[other_mask, :CT, 2:4]
    if torch.sum(other_mask) > 0 and others_pre_pos.size(1) > 2:
        others_accel = []
        others_accel_fwd = []
        others_accel_lat = []
        for oi in range(others_pre_pos.size(0)):
            res_accel = compute_accels(others_pre_pos[oi], others_pre_h[oi], scene['dt'])
            others_accel.append(res_accel[0])
            others_accel_fwd.append(res_accel[1])
            others_accel_lat.append(res_accel[2])
        others_accel = torch.cat(others_accel, dim=0)
        others_accel_fwd = torch.cat(others_accel_fwd, dim=0)
        others_accel_lat = torch.cat(others_accel_lat, dim=0)
        metrics = log_metric(metrics, 'adv_other_accel', others_accel.numpy())
        seq_metrics['adv_other_accel'] = others_accel.mean().item()
        metrics = log_metric(metrics, 'adv_other_accel_fwd', others_accel_fwd.numpy())
        seq_metrics['adv_other_accel_fwd'] = others_accel_fwd.mean().item()
        metrics = log_metric(metrics, 'adv_other_accel_lat', others_accel_lat.numpy())
        seq_metrics['adv_other_accel_lat'] = others_accel_lat.mean().item()
    else:
        seq_metrics['adv_other_accel'] = np.nan
        seq_metrics['adv_other_accel_fwd'] = np.nan
        seq_metrics['adv_other_accel_lat'] = np.nan

    #
    # Likelihood of latents under motion model prior
    #
    z_adv = scene['z_adv']
    z_prior_mean = scene['z_prior_mean']
    z_prior_var = scene['z_prior_var']
    z_sol = None
    if 'z_sol' in scene:
        z_sol = scene['z_sol']

    # LL of just attacker_agt (how plausible is the primary controlled agent's motion which causes a collision?)
    atk_z = z_adv[atk_agt:atk_agt+1]
    atk_z_ll = log_normal(atk_z, z_prior_mean[atk_agt:atk_agt+1], z_prior_var[atk_agt:atk_agt+1])
    metrics = log_metric(metrics, 'adv_z_ll_atk', atk_z_ll.flatten().numpy())
    seq_metrics['adv_z_ll_atk'] = atk_z_ll.item()
    if torch.sum(other_mask) > 0:
        # LL of all other agts (how plausible is the overall scene, especially those not involved directly in attack)
        other_z = z_adv[other_mask]
        other_z_ll = log_normal(other_z, z_prior_mean[other_mask], z_prior_var[other_mask])
        metrics = log_metric(metrics, 'adv_z_ll_other', other_z_ll.flatten().numpy())
        seq_metrics['adv_z_ll_other'] = other_z_ll.mean().item()
    else:
        seq_metrics['adv_z_ll_other'] = np.nan

    # Accuracy of fitting planner internally (up to collision)
    if planner_pred_fut is not None:
        # positional error
        pos_err = torch.norm(planner_gt_fut[:CT, :2] - planner_pred_fut[:CT, :2], dim=-1)
        metrics = log_metric(metrics, 'match_plan_pos', pos_err.numpy())
        seq_metrics['match_plan_pos'] = pos_err.mean().item()
        # angular error
        gt_h = planner_gt_fut[:CT, 2:4] / torch.norm(planner_gt_fut[:CT, 2:4], dim=-1, keepdim=True)
        pred_h = planner_pred_fut[:CT, 2:4] / torch.norm(planner_pred_fut[:CT, 2:4], dim=-1, keepdim=True)
        dotprod = torch.sum(gt_h * pred_h, dim=-1)
        dotprod = dotprod.clamp(-1, 1)
        ang_err_rad = torch.acos(dotprod)
        ang_err = torch.rad2deg(ang_err_rad)
        metrics = log_metric(metrics, 'match_plan_ang', ang_err.numpy())
        seq_metrics['match_plan_ang'] = ang_err.mean().item()
        metrics = log_metric(metrics, 'match_plan_ang_rad', ang_err_rad.numpy())
        seq_metrics['match_plan_ang_rad'] = ang_err_rad.mean().item()
    else:
        seq_metrics['match_plan_pos'] = np.nan
        seq_metrics['match_plan_ang'] = np.nan
        seq_metrics['match_plan_ang_rad'] = np.nan

    return metrics, freq_metrics_cnt, freq_metrics_total, seq_metrics

def compute_success_rates(scenarios):
    tot_scenes = float(sum([len(cur_scenes) for _, cur_scenes in scenarios.items()]))
    n_adv_succ = len(scenarios['adv_sol_success']) + len(scenarios['sol_failed'])
    adv_success_rate = n_adv_succ / tot_scenes
    sol_success_rate = len(scenarios['adv_sol_success']) / float(n_adv_succ)
    return adv_success_rate, sol_success_rate


def qual_eval(scenarios, viz_res, viz_stage, map_env, out_path,
                viz_video=False):
    for resname in viz_res:
        cur_viz_path = os.path.join(out_path, resname)
        mkdir(cur_viz_path)
        for scene in scenarios[resname]:
            viz_scene_path = os.path.join(cur_viz_path, scene['name'])
            mkdir(viz_scene_path)
            for stage in viz_stage:
                if resname == 'adv_failed' and stage == 'sol':
                    continue # can't viz this
                viz_scenario(scene, stage, map_env, viz_scene_path, video=viz_video)


def viz_scenario(scene, stage, map_env, out_path,
                L=720,
                W=720,
                video=False):
    assert(stage in ['init', 'adv', 'sol'])
    viz_out_path = os.path.join(out_path, 'viz_' + stage)

    scene_past = scene['past'][:,:,:4]
    scene_fut = None
    if stage == 'init':
        scene_fut = scene['fut_init']
    elif stage == 'adv':
        scene_fut = scene['fut_adv']
    elif stage == 'sol':
        if 'fut_sol' not in scene:
            return
        scene_fut = scene['fut_sol']

    lw = scene['veh_att']
    if 'attack_agt' in scene:
        atk_agt = scene['attack_agt']
    else:
        atk_agt = 0
    if 'attack_t' in scene:
        atk_t = scene['attack_t']
    else:
        atk_t = scene_fut.size(1) // 2

    # crop consistently across different stages
    # centroid of coll pos, and init for attacker and planner
    atk_past_idx = torch.nonzero(~torch.isnan(scene_past[atk_agt, :, 0]), as_tuple=True)[0][0]
    crop_pos = (scene['fut_adv'][0, atk_t, :2] + scene_past[0, 0, :2] + scene_past[atk_agt, atk_past_idx, :2]) / 3
    bound_diffs = torch.cat([scene['fut_adv'][0, atk_t, :2] - crop_pos, 
                            scene_past[0, 0, :2] - crop_pos, 
                            scene_past[atk_agt, atk_past_idx, :2] - crop_pos], dim=0)
    bound_max = torch.amax(torch.abs(bound_diffs)).item() + 5.0
    bound_max = max(30.0, bound_max)
    bounds = [-bound_max, -bound_max, bound_max, bound_max]
    crop_pos = crop_pos.unsqueeze(0)

    scene_traj = torch.cat([scene_past, scene_fut], dim=1)
    print(scene_traj.size())

    NA, NT, _ = scene_traj.size()

    map_name = scene['map']
    map_idx = map_env.map_list.index(map_name)
    crop_h = torch.Tensor([[1.0, 0.0]])
    crop_kin = torch.cat([crop_pos, crop_h], dim=1)

    # render local map crop
    map_rend = map_env.get_map_crop_pos(crop_kin,
                                        torch.tensor([map_idx], dtype=torch.long),
                                        bounds=bounds,
                                        L=L,
                                        W=W)[0]
    # transform trajectory into this cropped frame
    crop_traj, crop_lw = map_env.objs2crop(crop_kin[0],
                                            scene_traj.reshape(NA*NT, 4),
                                            lw,
                                            None,
                                            bounds=bounds,
                                            L=L,
                                            W=W)
    crop_traj = crop_traj.reshape(NA, NT, 4)

    car_colors, car_alpha, traj_linewidth, traj_marksize = nutils.get_adv_coloring(NA, atk_agt if stage != 'init' else None, 0,
                                                                    alpha=True,
                                                                    linewidth=True,
                                                                    markersize=True)
    traj_markers = [False]*NA
    draw_order = np.arange(NA)
    draw_order = draw_order[draw_order != atk_agt] # take out attacker
    draw_order = draw_order[1:] # take out ego
    draw_order = np.concatenate([draw_order, [0, atk_agt]])
    crop_traj = crop_traj[draw_order]
    crop_lw = crop_lw[draw_order]
    car_colors = np.array(car_colors)[draw_order].tolist()
    car_alpha = np.array(car_alpha)[draw_order].tolist()
    traj_linewidth = np.array(traj_linewidth)[draw_order].tolist()
    traj_markers = np.array(traj_markers)[draw_order].tolist()
    traj_marksize = np.array(traj_marksize)[draw_order].tolist()
    if stage == 'init':
        nutils.viz_map_crop(map_rend, viz_out_path + '_map.png', indiv=False)
    nutils.viz_map_crop(map_rend, viz_out_path + '.png',
                        crop_traj,
                        crop_lw,
                        viz_traj=True,
                        indiv=False,
                        car_colors=car_colors,
                        car_alpha=car_alpha,
                        traj_linewidth=traj_linewidth,
                        traj_markers=traj_markers,
                        traj_markersize=traj_marksize
                        )
    if video:
        nutils.viz_map_crop_video(map_rend, viz_out_path + '_vid',
                                crop_traj,
                                crop_lw,
                                viz_traj=False,
                                indiv=False,
                                car_colors=car_colors,
                                fps=2
                                )

def eval_adv_gen(cfg):
    '''
    eval_loader is assumed to be using batch size 1.
    '''
    scenario_dir = cfg.scenarios
    out_path = cfg.out
    cluster_path = cfg.cluster_path
    cluster_labels = cfg.cluster_labels

    if not os.path.exists(scenario_dir):
        print('Could not find scenario_dir %s!' % (scenario_dir))
        exit()
    mkdir(out_path)

    # read in all scenarios for this generation
    res_names = ['adv_failed', 'adv_sol_success', 'sol_failed']
    scenarios = {k : [] for k in res_names}
    for resname in res_names:
        res_path = os.path.join(scenario_dir, resname)
        if os.path.exists(res_path):
            # read in and run through adversarial dataset
            print('Reading in adversarial scenarios from %s...' % (res_path))
            # list of scenes: each a dict with map (str), init_state (N, 6), veh_att (N, 2), adv_fut (N-1, FT, 4)
            cur_scenes = read_adv_scenes(res_path)
            scenarios[resname] += cur_scenes

    for k, v in scenarios.items():
        print('%s : %d' % (k, len(v)))

    # create map envrionment for collision checking and visualization
    device = torch.device('cpu')
    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(data_path,
                             bounds=[-17.0, -38.5, 60.0, 38.5],
                             L=256,
                             W=256,
                            layers=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                            device=device,
                            flip_singapore=True,
                            load_lanegraph=False,
                            lanegraph_res_meters=1.0,
                            pix_per_m=8 if cfg.eval_qual else 4
                            )

    if cfg.eval_quant:
        quant_out = os.path.join(out_path, 'eval_quant')
        mkdir(quant_out)
        quant_eval(scenarios, cluster_path, cluster_labels, map_env, quant_out)

    if cfg.eval_qual:
        qual_out = os.path.join(out_path, 'eval_qual')
        mkdir(qual_out)
        qual_eval(scenarios, cfg.viz_res, cfg.viz_stage, map_env, qual_out, cfg.viz_video)
    

def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'eval_adv_gen_log.txt')
    Logger.init(log_path)
    # save arguments used
    Logger.log('Args: ' + str(cfg_dict))

    # device setup
    device = torch.device('cpu')
    Logger.log('Using device %s...' % (str(device)))

    eval_adv_gen(cfg)

if __name__ == "__main__":
    main()