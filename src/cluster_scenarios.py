# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, time, glob, shutil
import configargparse
import pickle

import torch
import numpy as np

from sklearn.cluster import KMeans

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import datasets.nuscenes_utils as nutils
from datasets.utils import read_adv_scenes
from utils.common import dict2obj, mkdir
from losses.adv_gen_nusc import interp_traj, check_single_veh_coll
from utils.transforms import transform2frame

def parse_cfg():
    '''
    Parse given config file into a config object.
    Returns: config object and config dict
    '''
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      config_file_parser_class=configargparse.YAMLConfigFileParser,
                                      description='Collision scenario clustering')
    # logging
    parser.add_argument('--out', type=str, default='./out/clustering_out',
                        help='Directory to save model weights and logs to.')
    # scenarios to cluster
    parser.add_argument('--scenario_dirs', nargs='+', type=str, default=None,
                        help='Directories to load scenarios from.')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters to use for k-means')
    # viz
    parser.add_argument('--viz', dest='viz', action='store_true',
                        help="If given, visualizes each collision and places them in directories based on clustering.")
    parser.set_defaults(viz=False)
    

    args = parser.parse_args()
    config_dict = vars(args)
    config = dict2obj(config_dict)
    
    return config, config_dict

def viz_scenario(veh_att, scene_traj, out_path, view_size=[50, 50]):
    center = torch.mean(scene_traj[0], dim=0)
    NA, T, _ = scene_traj.size()
    for t in range(T):
        fig = plt.figure(figsize=(6,6))
        nutils.render_obj_observation(scene_traj[:, t:t+1], veh_att,
                                      viz_traj=False,
                                      color_traj=None,
                                      color_traj_bounds=None,
                                      color=nutils.get_adv_coloring(NA, None, 0))
        plt.grid(b=None)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(center[0] - (view_size[0] / 2), center[0] + (view_size[0] / 2))
        plt.ylim(center[1] - (view_size[1] / 2), center[1] + (view_size[1] / 2))
        plt.tight_layout()
        tout_path = os.path.join(out_path, 'frame_%04d.jpg' % (t))
        print('saving', tout_path)
        plt.savefig(tout_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    nutils.create_video(os.path.join(out_path, 'frame_%04d.jpg'),
                        os.path.join(out_path + '.mp4'),
                        4)
    shutil.rmtree(out_path)

def compute_coll_feat(lw, scene_traj, dt):
    planner_traj = scene_traj[0]
    other_traj = scene_traj[1:]
    planner_lw = lw[0]
    other_lw = lw[1:]

    # interpolate to find better approx of exact time of collision
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

    feat = {
        'h' : coll_h,
        'hvec': coll_hvec,
        'ang' : coll_ang,
        'angvec' : coll_pos.numpy().tolist()
    }
    return feat

def cluster_scenarios(scenario_dirs, out_path, k, viz):
    # read in scenarios
    scene_list = []
    for scene_dir in scenario_dirs:
        cur_prefix = scene_dir.split('/')[-1]
        print('Reading in adversarial scenarios from %s...' % (cur_prefix))
        # list of scenes: each a dict with map (str), init_state (N, 6), veh_att (N, 2), adv_fut (N-1, FT, 4)
        cur_scenes = read_adv_scenes(scene_dir)
        scene_list += cur_scenes

    print('Collecting scene features...')
    feat_list = []
    name_list = []
    for si, scene in enumerate(scene_list):
        lw = scene['veh_att']
        traj = scene['scene_fut']
        cur_name = ('%04d_'%si) + scene['name']
        name_list.append(cur_name)
        # viz
        if viz:
            cur_out_path = os.path.join(out_path, cur_name)
            mkdir(cur_out_path)
            viz_scenario(lw, traj, cur_out_path)

        scene_feat = compute_coll_feat(lw, traj, scene['dt'])
        feat_list.append(scene_feat)

    # form full feature array (NC, D)
    angvec = np.array([feat['angvec'] for feat in feat_list])
    hvec = np.array([feat['hvec'] for feat in feat_list])
    scene_feats = np.concatenate([angvec, hvec], axis=1)
    print(scene_feats.shape)

    # perform clustering
    print('Clustering using k=%d clusters...' % (k))
    clustering = KMeans(n_clusters=k, random_state=0).fit(scene_feats)
    labels = clustering.labels_
    centroids = clustering.cluster_centers_

    # save clustering object to label scenarios later
    with open(os.path.join(out_path, 'cluster.pkl'), 'wb') as f:
        pickle.dump(clustering, f)

    fig, axs = plt.subplots(1, 2)
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    axs[0].plot(x, y, '--b', alpha=0.15)
    axs[0].title.set_text('collision direction')
    axs[1].plot(x, y, '--b', alpha=0.15)
    axs[1].title.set_text('adversary heading')
    axs[0].axis('equal')
    axs[1].axis('equal')
    for ki in np.unique(labels):
        axs[0].plot(angvec[:,0][labels == ki], angvec[:,1][labels == ki], 'o', markersize=4, label='%d'%ki)
        axs[1].plot(hvec[:,0][labels == ki], hvec[:,1][labels == ki], 'o', markersize=4, label='%d'%ki)
        # move all videos from this cluster to corresponding folder
        if viz:
            cur_clust_out = os.path.join(out_path, 'viz_clust%02d' % (ki))
            mkdir(cur_clust_out)
            for si in np.nonzero(labels == ki)[0]:
                shutil.move(os.path.join(out_path, name_list[si] + '.mp4'), 
                            os.path.join(cur_clust_out, name_list[si] + '.mp4'))

    # axs[0].legend()
    # axs[1].legend()
    plt.savefig(os.path.join(out_path, 'cluster_k%d.jpg' % (k)))
    plt.close(fig)

def main():
    cfg, cfg_dict = parse_cfg()
    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    cluster_scenarios(cfg.scenario_dirs, cfg.out, cfg.k, cfg.viz)

if __name__ == "__main__":
    main()