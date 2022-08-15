# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

'''
Given a directory of scenarios (.json files), visualizes them.
'''

import os, time
import configargparse
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from datasets.map_env import NuScenesMapEnv
from datasets.utils import read_adv_scenes
import datasets.nuscenes_utils as nutils
from utils.common import dict2obj, mkdir

def parse_cfg():
    '''
    Parse given config file into a config object.
    Returns: config object and config dict
    '''
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      config_file_parser_class=configargparse.YAMLConfigFileParser,
                                      description='Viz sceanrios')
    # logging
    parser.add_argument('--out', type=str, default='./out/viz_scenarios_out',
                        help='Directory to save visualizations to.')

    # scenarios
    parser.add_argument('--scenarios', type=str, required=True,
                        help='Directory to load scenarios from, should contain json files')

    # dir to load map data in from
    parser.add_argument('--data_dir', type=str, default='./data/nuscenes',
                        help='Directory to load data from.')

    # viz options
    parser.add_argument('--viz_video', dest='viz_video', action='store_true', help="Whether to save video or just trajectories")
    parser.set_defaults(viz_video=False)

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict

def viz_scenario(scene, map_env, out_path,
                L=720,
                W=720,
                video=False):
    viz_out_path = out_path

    scene_past = scene['scene_past'][:,:,:4]
    scene_fut = scene['scene_fut']
    lw = scene['veh_att']

    T = scene_past.size(1)
    crop_pos = scene_fut[0:1, T // 2, :2] # crop around ego
    centroid_states = scene_fut[:, T // 2, :2] - crop_pos
    bound_max = torch.amax(torch.abs(centroid_states)).item()# + 5.0
    bounds = [-bound_max, -bound_max, bound_max, bound_max]

    scene_traj = torch.cat([scene_past, scene_fut], dim=1)

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

    nutils.viz_map_crop(map_rend, viz_out_path + '.png',
                        crop_traj,
                        crop_lw,
                        viz_traj=True,
                        traj_markers=[False]*NA,
                        indiv=False
                        )
    if video:
        nutils.viz_map_crop_video(map_rend, viz_out_path + '_vid',
                                crop_traj,
                                crop_lw,
                                viz_traj=False,
                                indiv=False
                                )

def viz_scenario_dir(cfg):
    scenario_dir = cfg.scenarios
    out_path = cfg.out

    if not os.path.exists(scenario_dir):
        print('Could not find scenario_dir %s!' % (scenario_dir))
        exit()
    mkdir(out_path)

    scenes = read_adv_scenes(scenario_dir)
    print('Loaded:')
    print([s['name'] for s in scenes])

    # create map envrionment for visualization
    device = torch.device('cpu')
    data_path = os.path.join(cfg.data_dir, 'trainval')
    map_env = NuScenesMapEnv(data_path,
                             bounds=[-17.0, -38.5, 60.0, 38.5],
                             L=256,
                             W=256,
                            layers=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                            device=device,
                            pix_per_m=8
                            )

    # visualize all scenes
    for cur_scene in scenes:
        viz_scene_path = os.path.join(out_path, cur_scene['name'])
        mkdir(viz_scene_path)
        viz_scenario(cur_scene, map_env, viz_scene_path, video=cfg.viz_video)

def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    # save arguments used
    print('Args: ' + str(cfg_dict))

    viz_scenario_dir(cfg)

if __name__ == "__main__":
    main()