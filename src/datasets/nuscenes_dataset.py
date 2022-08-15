# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, itertools

import time
from tqdm import tqdm
import numpy as np
from itertools import chain
import json

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as Graph
from torch_geometric.data import DataLoader as GraphDataLoader

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.splits import create_splits_scenes

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
from datasets.map_env import NUSC_MAP_SIZES
import datasets.nuscenes_utils as nutils
from datasets.utils import MeanStdNormalizer, normalize_scene_graph, read_adv_scenes, NUSC_NORM_STATS, NUSC_VAL_SPLIT_200, NUSC_VAL_SPLIT_400

# This function is based on https://github.com/Khrylx/AgentFormer/blob/main/data/process_nuscenes.py#L20
# Copyright 2021 Carnegie Mellon University
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
NUM_IN_TRAIN_VAL_CHALLENGE = 200
def get_prediction_challenge_split(split, dataroot):
    '''
    Gets a list of {instance_token}_{sample_token} strings for each split.
    :param split: One of 'mini_train', 'mini_val', 'train', 'val'.
    :param dataroot: Path to the nuScenes dataset.
    :return: List of tokens belonging to the split. Format {instance_token}_{sample_token}.
    '''
    if split not in {'mini_train', 'mini_val', 'train', 'train_val', 'val'}:
        raise ValueError("split must be one of (mini_train, mini_val, train, train_val, val)")
    
    if split == 'train_val':
        split_name = 'train'
    else:
        split_name = split

    path_to_file = os.path.join(dataroot, "maps", "prediction", "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]
    
    if split == 'train':
        scenes_for_split = scenes_for_split[NUM_IN_TRAIN_VAL_CHALLENGE:]
    if split == 'train_val':
        scenes_for_split = scenes_for_split[:NUM_IN_TRAIN_VAL_CHALLENGE]

    token_list_for_scenes = map(lambda scene: prediction_scenes.get(scene, []), scenes_for_split)

    return prediction_scenes, scenes_for_split, list(chain.from_iterable(token_list_for_scenes))


class NuScenesDataset(Dataset):
    def __init__(self, data_path,
                 map_env,
                 version='mini',
                 split='train',
                 categories=['car', 'truck'],
                 npast=4,
                 nfuture=12,
                 nusc=None,
                 noise_std=0.0,
                 flip_singapore=True,
                 seq_interval=1,
                 randomize_val=False,
                 val_size=200,
                 scenario_path=None,
                 require_full_past=False,
                 use_challenge_splits=False,
                 reduce_cats=False
                ):
        '''
         - data_path : root directory of nuscenes version to be used
         - map_env : map environment to base map indices on
         - version: mini or trainval
         - split : train, val, or test. By defaults, splits by maps.
         - categories : which types of agents to return from
                        ['car', 'truck', 'bus', 'motorcycle', 'trailer', 'cyclist', 'pedestrian']
         - npast : the number of input (past) steps
         - nfuture : the number of output (future) steps
         - nusc : pre-loaded NuScenes object to use rather than loading in again.
         - noise_std: standard dev of gaussian noise to add to state vector.
         - flip_singapore: if true, flips singapore trajectories about the 
         - seq_interval: number of steps between sequences in the dataset. default is 1, i.e. each 
                        subsequence sequence return from the dataset will be shifted one timestep later
                        than the previous within the same scene.
         - randomize_val: if true, uses a random subset of train split for val, rather than the nusc predict val scenes.
         - val_size: size of the validation split (default: 200 which is the nuscenes prediction challenge). Does not apply to mini version.
         - scenario_path: (optional) path to additional adversarial scenarios to load in as part of the dataset. These will simply be appended to the end of the split. If this is given, data_path is allowed to be None.
         - require_full_past: if true, past history must be fully available to return an agent seq.
         - use_challenge_splits: only loads nuScenes prediction challenge data
         - reduce_cats: maps the agent category to be one of 'car', 'truck', or 'pedestrian'
        '''
        super(NuScenesDataset, self).__init__()
        assert version in ['mini', 'trainval']
        assert split in ['train', 'val', 'test']
        self.version = version
        self.data_path = data_path
        self.use_nusc = self.data_path is not None
        self.split = split
        self.map_env = map_env
        self.map_list = self.map_env.map_list
        self.dt = 0.5 # 2 Hz
        self.noise_std = noise_std
        self.npast = npast
        self.nfuture = nfuture
        self.seq_len = npast + nfuture
        self.flip_singapore = flip_singapore
        if self.flip_singapore:
            print('Flipping singapore trajectories data about x axis...')
        self.seq_interval = seq_interval
        self.randomize_val = randomize_val
        self.val_size = val_size
        self.scenario_path = scenario_path
        if self.scenario_path is None:
            assert self.use_nusc

        self.require_full_past = require_full_past
        if self.require_full_past:
            print('require_full_past activated...no agents will have nan past data')
        self.use_challenge_splits = use_challenge_splits
        if self.use_challenge_splits:
            print('Using official nuscenes pred challenge splits...')

        # high-level categories for the model that uses this data
        all_cats = ['car', 'truck', 'bus', 'motorcycle', 'trailer', 'cyclist', 'pedestrian', 'emergency', 'construction']
        all_cat2key = {
            'car' : ['vehicle.car'],
            'truck' : ['vehicle.truck'],
            'bus' : ['vehicle.bus'],
            'motorcycle' : ['vehicle.motorcycle'],
            'trailer' : ['vehicle.trailer'],
            'cyclist' : ['vehicle.bicycle'],
            'pedestrian' : ['human.pedestrian'],
            'emergency' : ['vehicle.emergency'],
            'construction' : ['vehicle.construction']
        }
        self.categories = categories
        self.key2cat = {}
        for cat in self.categories:
            if cat not in all_cats:
                print('Unrecognized category %s!' % (cat))
                exit()
            for k in all_cat2key[cat]:
                self.key2cat[k] = cat

        if reduce_cats:
            reduce_map = {
                'vehicle.car' : 'car',
                'vehicle.truck' : 'truck',
                'vehicle.bus' : 'truck',
                'vehicle.motorcycle' : 'motorcycle',
                'vehicle.trailer' : 'truck',
                'vehicle.bicycle' : 'cyclist',
                'human.pedestrian' : 'pedestrian',
                'vehicle.emergency' : 'car',
                'vehicle.construction' : 'truck'
            }
            self.key2cat = {k : reduce_map[k] for k in self.key2cat.keys()}
            self.categories = sorted(list(set([v for k, v in self.key2cat.items()])))

        iden = torch.eye(len(self.categories), dtype=torch.int)
        self.cat2vec = {self.categories[cat_idx] : iden[cat_idx] for cat_idx in range(len(self.categories))}
        self.vec2cat = {tuple(iden[cat_idx].tolist()) : self.categories[cat_idx]  for cat_idx in range(len(self.categories))}

        self.nusc = nusc
        if self.use_nusc and self.nusc is None:
            print('Creating nuscenes data object...')
            self.nusc = NuScenes(version='v1.0-{}'.format(version),
                                dataroot=self.data_path,
                                verbose=False)

        # tally number of frames in each class
        self.data = {}
        self.seq_map = []
        self.scene2map = {}
        if self.use_nusc:
            # list of scene names in this split
            self.scenes, self.pred_challenge_scenes = self.get_scenes()
            # maps {scene_name -> map_name}
            self.scene2map = self.get_scene2map()
            # load in all the data from these scenes
            self.data, self.seq_map = self.compile_data()            

        if self.scenario_path is not None:
            print('Num regular subseq: %d' % (len(self.seq_map)))
            print('Loading in additional scenario data...')
            scenario_data, scenario_seq_map, scenario_scene2map = self.compile_scenarios(self.scenario_path)
            self.data = {**self.data, **scenario_data}
            self.seq_map = self.seq_map + scenario_seq_map
            self.scene2map = {**self.scene2map, **scenario_scene2map}
            print('Num adversarial subseq: %d' % (len(scenario_seq_map)))

        self.data_len = len(self.seq_map)

        print('Num scenes: %d' % (len(self.data)))
        print('Num subseq: %d' % (self.data_len))

        # build normalization info objects
        # state normalizer. states of (x, y, hx, hy, s, hdot)
        ninfo = NUSC_NORM_STATS[tuple(sorted(self.categories))]
        norm_mean = [ninfo['lscale'][0], ninfo['lscale'][0], ninfo['h'][0], ninfo['h'][0], ninfo['s'][0], ninfo['hdot'][0]]
        norm_std = [ninfo['lscale'][1], ninfo['lscale'][1], ninfo['h'][1], ninfo['h'][1], ninfo['s'][1], ninfo['hdot'][1]]
        self.normalizer = MeanStdNormalizer(torch.Tensor(norm_mean),
                                           torch.Tensor(norm_std))
        # vehicle attribute normalizer of (l, w)
        att_norm_mean = [ninfo['l'][0], ninfo['w'][0]]
        att_norm_std = [ninfo['l'][1], ninfo['w'][1]]
        self.veh_att_normalizer = MeanStdNormalizer(torch.Tensor(att_norm_mean),
                                                  torch.Tensor(att_norm_std))
        self.norm_info = ninfo
    
    def get_state_normalizer(self):
        return self.normalizer

    def get_att_normalizer(self):
        return self.veh_att_normalizer

    def compile_scenarios(self, scenario_path):
        adv_scenes = read_adv_scenes(scenario_path)
        scene2info = {}
        scene2map = {}
        seq_map = []
        for scene in adv_scenes:
            sname = scene['name']
            scene2info[sname] = {}

            lw = scene['veh_att'].numpy()
            NA = lw.shape[0]
            k = ['car' for _ in range(NA)] # by default assume all cars
            if 'sem' in scene:
                k = [self.vec2cat[tuple(sem)] for sem in scene['sem'].numpy().tolist()]

            past = scene['scene_past'].numpy()
            fut = scene['scene_fut'].numpy()

            attack_t = scene['attack_t']

            # need to compute velocities for future (already have for past)
            fut_traj = np.concatenate([past[:, -1:, :4], fut], axis=1)
            t = np.arange(fut_traj.shape[1])*self.dt
            vels = [np.linalg.norm(nutils.velocity(fut_traj[aidx, :, :2], t)[1:], axis=1) for aidx in range(NA)]
            fut_h = np.arctan2(fut_traj[:, :, 3], fut_traj[:, :, 2])
            hdots = [nutils.heading_change_rate(fut_h[aidx], t)[1:] for aidx in range(NA)]
            
            # ego
            ego_fut = np.concatenate([fut[0], vels[0][:, np.newaxis], hdots[0][:, np.newaxis]], axis=1)
            ego_traj = np.concatenate([past[0], ego_fut], axis=0)
            ego_lw = lw[0]
            ego_is_vis = np.ones((ego_traj.shape[0]), dtype=int)
            ego_info = {
                'traj' : ego_traj,
                'lw' :  ego_lw,
                'is_vis' : ego_is_vis,
                'k' : 'ego',
            }
            scene2info[sname]['ego'] = ego_info

            # all others
            for aidx in range(1, NA):
                cur_fut = np.concatenate([fut[aidx], vels[aidx][:, np.newaxis], hdots[aidx][:, np.newaxis]], axis=1)
                cur_traj = np.concatenate([past[aidx], cur_fut], axis=0)
                is_vis = np.logical_not(np.isnan(np.sum(cur_traj, axis=1))).astype(int)
                info = {
                    'traj' : cur_traj,
                    'lw' :  lw[aidx],
                    'is_vis' : is_vis,
                    'k' : k[aidx]
                }
                scene2info[sname]['agt%03d' % (aidx)] = info

            # update data map
            scene2map[sname] = (scene['map'], self.map_list.index(scene['map']))
            T = ego_traj.shape[0]
            scene_seq = [(sname, start_idx) for start_idx in range(0, T - self.seq_len, 1)]
            seq_map.extend(scene_seq)

        return scene2info, seq_map, scene2map

    def get_scenes(self):
        # filter by scene split
        # use val for testing and split up train for validation
        cur_split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.split == 'train' or self.split == 'val']
        scenes = create_splits_scenes()[cur_split]       

        # use scenes based on the prediction challenge, so val is actually part of the
        #   training set and we use the true val split for testing
        num_in_train_val = self.val_size if self.version == 'trainval' else 2 # mini is just hardcoded
        scenes = np.array(scenes)
        print('num total scenes before split: %d' % (len(scenes)))
        val_mask = np.zeros((scenes.shape[0]), dtype=bool)
        if self.split in ['train', 'val']:
            if self.randomize_val:
                print('Using validation split that is a random subset of train!')
                # pre-computed random splits
                if num_in_train_val == 200:
                    val_inds = NUSC_VAL_SPLIT_200
                elif num_in_train_val == 400:
                    val_inds = NUSC_VAL_SPLIT_400
                elif self.version == 'mini':
                    val_inds = [0,1]
                else:
                    print('Val size not supported! Please use 200, 400')
                val_inds = np.array(val_inds)
                val_mask[val_inds] = True
            else:
                val_mask[:num_in_train_val] = True

        if self.split == 'train':
            scenes = scenes[np.logical_not(val_mask)]
        if self.split == 'val':
            scenes = scenes[val_mask]

        scenes = sorted(scenes.tolist())

        pred_challenge_scenes = None # dict mapping scene_name -> list of instance-tok_sample-tok for each challenge split data
        if self.use_challenge_splits:
            from nuscenes.prediction import PredictHelper
            chall_split_map = {
                'train' : 'train',
                'val' : 'train_val',
                'test' : 'val'
            }
            pred_challenge_scenes, scenes, split_data = get_prediction_challenge_split(chall_split_map[self.split], dataroot=self.data_path)

        return scenes, pred_challenge_scenes

    def get_scene2map(self):
        scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            scene2map[rec['name']] = (log['location'], self.map_list.index(log['location']))
        return scene2map

    def compile_data(self):
        # collect all samples (keyframes) in our split scenss and sort by timestamp
        recs = [rec for rec in self.nusc.sample]
        for rec in recs:
            rec['scene_name'] = self.nusc.get('scene', rec['scene_token'])['name']
        recs = [rec for rec in recs if rec['scene_name'] in self.scenes]
        recs = sorted(recs, key=lambda x: (x['scene_name'], x['timestamp']))
    
        scene2data = {}
        print('Loading in %s data...' % (self.split))
        for rec in tqdm(recs):
            samp_token = rec['token']
            scene = rec['scene_name']
            mname = self.scene2map[scene][0]
            mheight = NUSC_MAP_SIZES[mname][0]
            locname = mname.split('-')[0]
            if scene not in scene2data:
                scene2data[scene] = {}
                scene2data[scene]['ego'] = {'traj': [], 'w': 1.73,
                                            'l': 4.084, 'k': 'ego'}
            # add ego location always
            egopose = self.nusc.get('ego_pose',
                                    self.nusc.get('sample_data',
                                                  rec['data']['LIDAR_TOP'])
                                    ['ego_pose_token'])
            rot = Quaternion(egopose['rotation']).rotation_matrix
            rot = np.arctan2(rot[1, 0], rot[0, 0])
            scene2data[scene]['ego']['traj'].append({
                'x': egopose['translation'][0],
                'y': mheight - egopose['translation'][1] if self.flip_singapore and locname == 'singapore' else egopose['translation'][1],
                'h' : rot,
                'hcos': np.cos(rot),
                'hsin': -np.sin(rot) if self.flip_singapore and locname == 'singapore' else np.sin(rot),
                't': egopose['timestamp'],
                'samp_tok' : samp_token
            })                
            # add detection locations
            for ann in rec['anns']:
                instance = self.nusc.get('sample_annotation', ann)
                # only add if in the desired categories
                cur_key = '.'.join(instance['category_name'].split('.')[:2])
                if not cur_key in self.key2cat:
                    continue
                cur_cat = self.key2cat[cur_key]

                instance_name = instance['instance_token']
                rot = Quaternion(instance['rotation']).rotation_matrix
                rot = np.arctan2(rot[1, 0], rot[0, 0])
                if instance_name not in scene2data[scene]:
                    assert(instance_name != 'ego'), instance_name
                    scene2data[scene][instance_name] =\
                        {'traj': [], 'w': instance['size'][0],
                         'l': instance['size'][1],
                         'k': cur_cat}
                scene2data[scene][instance_name]['traj'].append({
                    'x': instance['translation'][0],
                    'y': mheight - instance['translation'][1] if self.flip_singapore and locname == 'singapore' else instance['translation'][1],
                    'h' : rot,
                    'hcos': np.cos(rot),
                    'hsin': -np.sin(rot) if self.flip_singapore and locname == 'singapore' else np.sin(rot),
                    't': rec['timestamp'],
                    'samp_tok' : samp_token
                })

        return self.post_process(scene2data)

    def post_process(self, data):
        scene2info = {}
        print('Post-processing data...')
        drivable_raster = self.map_env.nusc_raster[:, 0]
        carpark_raster = None
        if 'carpark_area' in self.map_env.layer_map:
            carpark_raster = self.map_env.nusc_raster[:, self.map_env.layer_map['carpark_area']]
        seq_map = [] # for deterministic iteration through data maps from data_idx -> (scene_name, start_idx)
        for scene in tqdm(data):
            scene2info[scene] = {}

            challenge_inst_samp_list = None
            if self.use_challenge_splits:
                challenge_inst_samp_list = self.pred_challenge_scenes.get(scene, [])
                challenge_inst_samp_list = {chall_inst_samp : True for chall_inst_samp in challenge_inst_samp_list}

            # first process ego info so we know all timestamps (since always available)
            ego_data = data[scene]['ego']
            # map timestamps -> frame idx to aggregate other incomplete agents
            ego_t_map = {row['t'] : ridx for ridx, row in enumerate(ego_data['traj'])}
            T = len(ego_t_map)
            # state with no vel
            ego_x = np.array([[row['x'], row['y'], row['hcos'], row['hsin']]
                                    for row in ego_data['traj']])
            ego_h = np.array([row['h'] for row in ego_data['traj']]) # heading angle
            ego_t = np.array([row['t']*1e-6 for row in ego_data['traj']])
            # compute speed
            ego_pos = ego_x[:, :2]
            ego_vel = nutils.velocity(ego_pos, ego_t)
            ego_s = np.linalg.norm(ego_vel, axis=1).reshape((-1, 1))
            ego_a = np.linalg.norm(nutils.velocity(ego_vel, ego_t), axis=1).reshape((-1, 1))
            # compute hdot
            ego_hdot = nutils.heading_change_rate(ego_h, ego_t).reshape((-1, 1))
            ego_ddh = nutils.heading_change_rate(ego_hdot.reshape((-1)), ego_t).reshape((-1, 1))
            # form state and record valid frames
            ego_traj = np.concatenate([ego_x, ego_s, ego_hdot], axis=1)
            ego_accel = np.concatenate([ego_a, ego_ddh], axis=1)
            ego_is_vis = np.logical_not(np.isnan(ego_s.flatten())).astype(np.int)
            # vehicle attributes
            ego_lw = np.array([ego_data['l'], ego_data['w']])

            samp_tok_seq = [row['samp_tok'] for row in ego_data['traj']]

            ego_info = None
            if self.use_challenge_splits:
                # have to add dummy data for a few steps at the beginning b/c they use data at first few steps
                nadd_steps = 2 # NOTE assumes at least 4 past steps are needed, won't work for > 4
                ego_info = {
                    'traj' : np.concatenate([np.ones((nadd_steps, 6))*np.nan, ego_traj], axis=0),
                    'accel' : np.concatenate([np.ones((nadd_steps, 2))*np.nan, ego_accel], axis=0),
                    'lw' :  ego_lw,
                    'is_vis' : np.concatenate([np.zeros((nadd_steps), dtype=np.int), ego_is_vis], axis=0),
                    'k' : 'ego',
                }
            else:
                ego_info = {
                    'traj' : ego_traj,
                    'accel' : ego_accel,
                    'lw' :  ego_lw,
                    'is_vis' : ego_is_vis,
                    'k' : 'ego',
                }
            scene2info[scene]['ego'] = ego_info

            # now process other agents
            # first map onto same timeline as ego with nans at other spots
            #   also compute velocities.
            for name in data[scene]:
                if name == 'ego':
                    continue
                info = {}
                inst_tok = name
                t_list = [row['t'] for row in data[scene][name]['traj']]
                x_list = np.array([[row['x'], row['y'], row['hcos'], row['hsin']]
                                    for row in data[scene][name]['traj']])
                h_list = np.array([row['h'] for row in data[scene][name]['traj']])
                lw = np.array([data[scene][name]['l'], data[scene][name]['w']]) # veh attribs

                #  check if in challenge split, if so need to add to data no matter what
                in_chall_split = np.zeros((T), dtype=bool)
                if self.use_challenge_splits:
                    # check which steps for this agent are in the challenge split
                    inst_samp_tok_list = [inst_tok + '_' + samp_tok_t for samp_tok_t in samp_tok_seq]
                    for tidx, inst_samp in zip(np.arange(T), inst_samp_tok_list):
                        in_chall_split[tidx] = inst_samp in challenge_inst_samp_list

                valid_frame = np.ones((x_list.shape[0]), dtype=bool) # all valid by default
                if not self.use_challenge_splits or np.sum(in_chall_split) == 0: # if using challenge split, need all frames of any vehicles that we need to make a pred for.
                    # check if on drivable area (i.e. layer 0 of maps) at each frame
                    torch_xlist = torch.from_numpy(x_list).to(drivable_raster.device)
                    torch_lw = torch.from_numpy(lw).to(drivable_raster.device).unsqueeze(0).expand(x_list.shape[0], 2)
                    mapixes =  torch.Tensor([self.scene2map[scene][1]]).to(drivable_raster.device).long().expand(x_list.shape[0])
                    drivable_frac = nutils.check_on_layer(drivable_raster,
                                                            self.map_env.nusc_dx,
                                                            torch_xlist,
                                                            torch_lw,
                                                            mapixes)
                    valid_frame = (drivable_frac >= 0.3).cpu().numpy()
                    # if available, check if in a carpark area
                    if carpark_raster is not None:
                        carpark_frac = nutils.check_on_layer(carpark_raster,
                                                            self.map_env.nusc_dx,
                                                            torch_xlist,
                                                            torch_lw,
                                                            mapixes)
                        not_on_carpark = (carpark_frac < 0.3).cpu().numpy()
                        valid_frame = np.logical_and(valid_frame, not_on_carpark)

                cur_x = np.ones_like(ego_x)*np.nan
                cur_h = np.ones_like(ego_h)*np.nan
                # only have values at observed frames
                for t, x, h, keep in zip(t_list, x_list, h_list, valid_frame):
                    if keep: # only keep frames on drivable and not in parking lot
                        cur_x[ego_t_map[t]] = x
                        cur_h[ego_t_map[t]] = h

                # if all frames nan (never on drivable surface), throw it out
                if np.sum(np.isnan(cur_x), axis=0)[0] == cur_x.shape[0]:
                    continue

                # compute speed
                pos = cur_x[:, :2]
                vel = nutils.velocity(pos, ego_t)
                s = np.linalg.norm(vel, axis=1).reshape((-1, 1))
                a = np.linalg.norm(nutils.velocity(vel, ego_t), axis=1).reshape((-1, 1))
                # compute hdot
                hdot = nutils.heading_change_rate(cur_h, ego_t).reshape((-1, 1))
                ddh = nutils.heading_change_rate(hdot.reshape((-1)), ego_t).reshape((-1, 1))
                # form state and record valid frames
                no_vis = np.isnan(s.flatten())
                # some position values might be available while vel is nan
                #       (when single random frame shows up)
                cur_x[no_vis] = np.nan
                is_vis = np.logical_not(no_vis).astype(np.float)
                traj = np.concatenate([cur_x, s, hdot], axis=1)
                accel = np.concatenate([a, ddh], axis=1)

                info = None
                nadd_steps = 2
                if self.use_challenge_splits:
                    info = {
                        'traj' : np.concatenate([np.ones((nadd_steps, 6))*np.nan, traj], axis=0),
                        'accel' : np.concatenate([np.ones((nadd_steps, 2))*np.nan, accel], axis=0),
                        'lw' :  lw,
                        'is_vis' : np.concatenate([np.zeros((nadd_steps), dtype=np.int), is_vis], axis=0),
                        'k' : data[scene][name]['k']
                    }
                else:
                    info = {
                        'traj' : traj,
                        'accel' : accel,
                        'lw' :  lw,
                        'is_vis' : is_vis,
                        'k' : data[scene][name]['k']
                    }
                scene2info[scene][name] = info

                if self.use_challenge_splits:
                    # check which steps for this agent are in the challenge split
                    inst_samp_tok_list = (['']*nadd_steps) + [inst_tok + '_' + samp_tok_t for samp_tok_t in samp_tok_seq]
                    # (scene, start_idx, inst_tok)
                    for tidx, inst_samp in zip(np.arange(T+nadd_steps), inst_samp_tok_list):
                        if inst_samp in challenge_inst_samp_list:
                            cur_sidx = tidx - (self.npast-1) # want last frame of past to be at the frame in the split
                            seq_map.append((scene, cur_sidx, inst_tok))
                            assert(cur_sidx >= 0)
                            challenge_inst_samp_list[inst_samp] = False

            if not self.use_challenge_splits:
                # update data map
                scene_seq = [(scene, start_idx) for start_idx in range(0, T - self.seq_len, self.seq_interval)]
                seq_map.extend(scene_seq)

        return scene2info, seq_map

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        idx_info = self.seq_map[idx]
        inst_tok = None
        if not self.use_challenge_splits:
            scene_name, sidx = idx_info
        else:
            scene_name, sidx, inst_tok = idx_info
        eidx = sidx + self.seq_len
        midx = sidx + self.npast
        _, map_idx = self.scene2map[scene_name]

        # NOTE only keep an agent in the sequence if it has an annotation
        #       at last frame of the past.
        #       This is not perfect since past/future-only agents will certainly affect traffic

        # always put ego at node 0
        ego_data = self.data[scene_name]['ego']
        past = [ego_data['traj'][sidx:midx, :]]
        future = [ego_data['traj'][midx:eidx, :]]
        sem = [self.cat2vec['car']] # one-hot vec
        lw = [ego_data['lw']]
        past_vis = [ego_data['is_vis'][sidx:midx]]
        fut_vis = [ego_data['is_vis'][midx:eidx]]

        if self.use_challenge_splits:
            # prepend data for the agent we're making a prediction for
            # so ego is not at 0, challenge data is
            agent_data = self.data[scene_name][inst_tok]
            assert(np.isnan(agent_data['traj'][midx-1]).astype(np.int).sum() == 0) # should not be nan if we're making a prediction for it
            past = [agent_data['traj'][sidx:midx, :]] + past
            future = [agent_data['traj'][midx:eidx, :]] + future
            sem = [self.cat2vec[agent_data['k']]] + sem # one-hot vec
            lw = [agent_data['lw']] + lw
            past_vis = [agent_data['is_vis'][sidx:midx]] + past_vis
            fut_vis = [agent_data['is_vis'][midx:eidx]] + fut_vis

        for agent in self.data[scene_name]:
            if agent == 'ego':
                continue
            if self.use_challenge_splits and agent == inst_tok:
                continue
            agent_data = self.data[scene_name][agent]
            if np.isnan(agent_data['traj'][midx-1]).astype(np.int).sum() > 0:
                continue
            if self.require_full_past and np.isnan(agent_data['traj'][:midx]).sum() > 0:
                # has some nan in past
                continue

            # have a valid agent, add info
            # may be nan at many frames, this must be dealt with in model
            past.append(agent_data['traj'][sidx:midx, :])
            future.append(agent_data['traj'][midx:eidx, :])
            sem.append(self.cat2vec[agent_data['k']])
            lw.append(agent_data['lw'])
            past_vis.append(agent_data['is_vis'][sidx:midx])
            fut_vis.append(agent_data['is_vis'][midx:eidx])

        past = torch.Tensor(np.stack(past, axis=0))
        future = torch.Tensor(np.stack(future, axis=0))
        sem = torch.Tensor(np.stack(sem, axis=0))
        lw = torch.Tensor(np.stack(lw, axis=0))
        past_vis = torch.Tensor(np.stack(past_vis, axis=0))
        fut_vis = torch.Tensor(np.stack(fut_vis, axis=0))

        # normalize
        past_gt = self.normalizer.normalize(past) # gt past (no noise)
        past = self.normalizer.normalize(past)
        future_gt = self.normalizer.normalize(future) # gt future (used to compute err/loss)
        future = self.normalizer.normalize(future) # observed future (input to net)
        lw = self.veh_att_normalizer.normalize(lw)

        # add noise if desired
        if self.noise_std > 0:
            past += torch.randn_like(past)*self.noise_std
            future += torch.randn_like(future)*self.noise_std
            # make sure heading is still a unit vector
            past[:, :, 2:4] = past[:, :, 2:4] / torch.norm(past[:, :, 2:4], dim=-1, keepdim=True)
            future[:, :, 2:4] = future[:, :, 2:4] / torch.norm(future[:, :, 2:4], dim=-1, keepdim=True)
            # make sure position is still positive
            past[:, :, :2] = torch.clamp(past[:, :, :2], min=0.0)
            future[:, :, :2] = torch.clamp(future[:, :, :2], min=0.0)
            # also for vehicle attributes
            lw += torch.randn_like(lw)*self.noise_std

        #  then build fully-connected scene graph
        NA = past.size(0)
        edge_index = None
        if NA > 1:
            node_list = range(NA)
            edge_index = list(itertools.product(node_list, node_list))
            edge_index_list = [(i, j) for i, j in edge_index if i != j]
            edge_index = torch.Tensor(edge_index_list).T.to(torch.long).contiguous()
        else:
            edge_index = torch.Tensor([[],[]]).long()

        graph_prop_dict = {
            'x' : torch.empty((NA,)),
            'pos' : torch.empty((NA,)),
            'edge_index' : edge_index,
            'past' : past,
            'past_gt' : past_gt,
            'future' : future,
            'future_gt' : future_gt,
            'sem' : sem,
            'lw' : lw,
            'past_vis' : past_vis,
            'future_vis' : fut_vis,
        }
        scene_graph = Graph(**graph_prop_dict)

        return scene_graph, map_idx
