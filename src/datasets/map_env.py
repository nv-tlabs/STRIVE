# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import numpy as np
import torch

import sys
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
import datasets.nuscenes_utils as nutils

NUSC_MAP_SIZES = { # in meters (H x W)
            'singapore-onenorth' : [2025.0, 1585.6],
            'singapore-hollandvillage' : [2922.9, 2808.3],
            'singapore-queenstown' : [3687.1, 3228.6],
            'boston-seaport' : [2118.1, 2979.5]
        }

class NuScenesMapEnv(object):
    def __init__(self, map_data_path,
                       bounds=[-17.0, -38.5, 60.0, 38.5],
                       layers=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                       L=256,
                       W=256,
                       device='cpu',
                       flip_singapore=True,
                       load_lanegraph=False,
                       lanegraph_res_meters=1.0,
                       lanegraph_eps=1e-6,
                       pix_per_m=4):
        '''
        :param map_data_path: path to the dataset e.g. /path/to/mini which contains the maps directory
        :param bounds: [low_l, low_w, high_l, high_w] distances (in meters) around location to crop map observations
        :param layers: name of the nusc layers to render
        :param L: number of pixels along length of vehicle to render crop with
        :param W: number of pixels along width of vehicle to render crop with
        :param device: the device to store the rasterized maps on. Note that for
                        5 pix/m resolution this required ~5GB of memory for road and divider layers.
        :param flip_singapore: if true, flips singapore maps about the x axis to change driving direction from 
                                left hand to right hand side
        :param load_lanegraph: if true, loads the lane graph as well
        :param lanegraph_res_meters: resolution at which to discretize lane graph
        :param lanegraph_eps:
        :param pix_per_m: resolution to discretize map layers
        '''
        super(NuScenesMapEnv, self).__init__()

        self.data_path = map_data_path
        self.nusc_maps = nutils.get_nusc_maps(map_data_path)
        if load_lanegraph:
            print('Loading lane graphs...')
            self.lane_graphs = {map_name: nutils.process_lanegraph(nmap, lanegraph_res_meters, lanegraph_eps) for map_name,nmap in self.nusc_maps.items()}
        self.map_list = list(self.nusc_maps.keys())
        self.layer_names = layers
        self.bounds = bounds
        self.L = L
        self.W = W
        self.device = torch.device(device)
        self.flip_singapore = flip_singapore

        self.road_list = ['drivable_area', 'road_segment', 'lane']
        self.num_layers = 0
        road_layers = [lay for lay in self.layer_names if lay in self.road_list]
        self.num_layers = 1 if len(road_layers) > 0 else 0
        nonroad_layers = [lay for lay in self.layer_names if lay not in self.road_list]
        self.num_layers += len(nonroad_layers)

        # map layer names to their channel index in returned crops
        self.layer_map = {}
        for lay in road_layers:
            self.layer_map[lay] = 0
        lay_idx = 1
        for lay in nonroad_layers:
            self.layer_map[lay] = lay_idx
            lay_idx += 1
        
        # binarize all the layers we need for all maps and cache for crop later
        print('Rasterizing nuscenes maps...')
        m_per_pix = 1.0 / pix_per_m
        self.nusc_raster = []
        self.nusc_dx = []
        max_H, max_W = -float('inf'), -float('inf')
        msize_list = []
        for midx, mname in enumerate(self.map_list):
            nmap = self.nusc_maps[mname]
            msize = np.array(NUSC_MAP_SIZES[mname])
            cur_msize = msize * pix_per_m
            cur_msize = np.round(cur_msize).astype(np.int32)
            cur_dx = msize / cur_msize
            self.nusc_dx.append(cur_dx)
            cur_msize = tuple(cur_msize)
            if cur_msize[0] > max_H:
                max_H = cur_msize[0]
            if cur_msize[1] > max_W:
                max_W = cur_msize[1]
            msize_list.append(cur_msize)

        for midx, mname in enumerate(self.map_list):
            print(mname)
            nmap = self.nusc_maps[mname]
            cur_msize = msize_list[midx]

            # get binarized rasterization of full map
            map_layers = []
            # first road
            # draw both road layers in one channel
            road_layers = [lay for lay in self.layer_names if lay in self.road_list]
            if len(road_layers) > 0:
                road_img = nmap.get_map_mask(None, 0.0, road_layers, cur_msize)
                # collapse to single layer
                road_img = np.clip(np.sum(road_img, axis=0), 0, 1).reshape((1, cur_msize[0], cur_msize[1])).astype(np.uint8)
                map_layers.append(road_img)

            # draw any other layers separately (e.g. walkway)
            other_layers = [lay for lay in self.layer_names if lay not in self.road_list]
            if len(other_layers) > 0:
                other_img = nmap.get_map_mask(None, 0.0, other_layers, cur_msize)
                map_layers.append(other_img)

            # Create single image
            map_img = np.concatenate(map_layers, axis=0)
            print(map_img.shape)

            # flip about x axis if desired (i.e. switch driving direction)
            if self.flip_singapore and mname.split('-')[0] == 'singapore':
                print('Flipping %s about x axis...' % (mname))
                map_img = np.flip(map_img, axis=1).copy()

                if load_lanegraph:
                    print('Flipping lane graph about x axis...')
                    # also need to flip lane graph
                    cur_lg = self.lane_graphs[mname]
                    # xys is (L, 2), reflect y
                    xys = cur_lg['xy']
                    mheight = NUSC_MAP_SIZES[mname][0]
                    xys[:,1] = mheight - xys[:,1]
                    self.lane_graphs[mname]['xy'] = xys
                    # negate diffy in edges [x0, y0, diff[0], diff[1], dist]
                    edges = cur_lg['edges']
                    edges[:,1] = mheight - edges[:,1]
                    edges[:,3] *= -1
                    self.lane_graphs[mname]['edges'] = edges

            # # NOTE: viz debug
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(6, 6))
            # plt.imshow(map_img[0], origin='lower', vmin=0, vmax=1)
            # # plt.imshow(map_img[0], vmin=0, vmax=1)
            # plt.gca().set_aspect('equal', adjustable='box')
            # imname = 'check_%04d_%s_nopad_flipped.jpg' % (0, mname)
            # print('saving', imname)
            # plt.savefig(imname)
            # plt.close(fig)

            pad_right = max_W - cur_msize[1]
            pad_bottom = max_H - cur_msize[0]
            pad = torch.nn.ZeroPad2d((0, pad_right, 0, pad_bottom))
            padded_map_img = pad(torch.from_numpy(map_img).unsqueeze(0))[0]

            self.nusc_raster.append(padded_map_img)

        # pad each map to max for efficient cropping
        self.nusc_raster = torch.stack(self.nusc_raster, dim=0).to(device)
        self.nusc_dx = torch.from_numpy(np.stack(self.nusc_dx, axis=0)).to(device)

    def get_map_crop(self, scene_graph, map_idx,
                    bounds=None,
                    L=None,
                    W=None):
        '''
        Render local crop for whole batch of agents represented as a scene graph.
        Assumes .pos is UNNORMALIZED in true scale map coordinate frame.

        :param scene_graph: batched scene graph with .pos size (N x 4) or (N x NS x 4) (x,y,hx,hy) in .lw (N x 2) (x,y)
                            will render each crop in the frame of .pos. The .batch attrib must be idx
                            for which agent is in which batch.
        :param map_idx: the map index of each batch in the scene graph (B,)
        :params bounds, L, W: overrides bounds, L, W set in constructor

        :returns map_crop: N x C x H x W
        '''
        device = scene_graph.pos.device
        B = map_idx.size(0)
        NA = scene_graph.pos.size(0)

        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W

        # map index for each agent in the whole scene graph. 
        mapixes = map_idx[scene_graph.batch]
        pos_in = scene_graph.pos
        if len(scene_graph.pos.size()) == 3:
            NS = scene_graph.pos.size(1)
            pos_in = pos_in.reshape(NA*NS, -1)
            mapixes = mapixes.unsqueeze(1).expand(NA, NS).reshape(-1)
        # render by indexing into pre-rasterized binary maps
        map_obs = nutils.get_map_obs(self.nusc_raster, self.nusc_dx, pos_in,
                                     mapixes, bounds, L=L, W=W).to(device)
        
        return map_obs

    def get_map_crop_pos(self, pos, mapixes,
                          bounds=None,
                          L=None,
                          W=None):
        '''
        Render local crops around given global positions (assumed UNNORMALIZED).

        :param pos: batched positions (N x 4) (x,y,hx,hy) in .lw (N x 2) (x,y)
        :param mapixes: the map index of each batch in the scene graph (N,)
        :params bounds, L, W: overrides bounds, L, W set in constructor

        :returns map_crop: N x C x H x W
        '''
        device = pos.device
        NA = pos.size(0)

        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W

        # render by indexing into pre-rasterized binary maps
        map_obs = nutils.get_map_obs(self.nusc_raster, self.nusc_dx, pos,
                                     mapixes, bounds, L=L, W=W).to(device)
        
        return map_obs

    def objs2crop(self, center, obj_center, obj_lw, map_idx, bounds=None, L=None, W=None):
        '''
        converts given objects N x 4 to the crop frame defined by the given center (x,y,hx,hy)
        '''
        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W
        local_objs = nutils.objects2frame(obj_center.cpu().numpy()[np.newaxis, :, :],
                                          center.cpu().numpy())[0]
        # [low_l, low_w, high_l, high_w]
        local_objs[:, 0] -= bounds[0]
        local_objs[:, 1] -= bounds[1]

        # convert to pix space
        pix2m_L = L / float(bounds[2] - bounds[0])
        pix2m_W = W / float(bounds[3] - bounds[1])
        local_objs[:, 0] *= pix2m_L
        local_objs[:, 1] *= pix2m_W
        pix_objl = obj_lw[:, 0]*pix2m_L
        pix_objw = obj_lw[:, 1]*pix2m_W
        pix_objlw = torch.stack([pix_objl, pix_objw], dim=1)
        local_objs = torch.from_numpy(local_objs)

        return local_objs, pix_objlw
