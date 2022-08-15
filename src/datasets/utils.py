# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import glob, json, os

import torch
import numpy as np

def read_adv_scenes(scene_path):
    scene_flist = sorted(glob.glob(os.path.join(scene_path, '*.json')))    
    scene_list = []
    for scene_fpath in scene_flist:
        scene_name = scene_fpath.split('/')[-1][:-5]
        # print('Loading %s...' % (scene_name))
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
        cur_scene['veh_att'] = torch.tensor(jdict['lw'])
        cur_scene['scene_past'] = torch.tensor(jdict['past'])
        cur_scene['scene_fut'] = torch.tensor(jdict['fut_adv'])
        if 'attack_t' in jdict:
            cur_scene['attack_t'] = jdict['attack_t']
        if 'sem' in jdict:
            cur_scene['sem'] = torch.tensor(jdict['sem'])

        scene_list.append(cur_scene)

    return scene_list

#
# Normalization Helpers
#

class MeanStdNormalizer(object):
    '''
    Util objects to normalize and unnormalize tensors of states by subtracting mean and dividing by std.
    (data - \mu) / \sigma
    '''
    def __init__(self, mean_vals, std_vals):
        '''
        :param mean_vals:  (D,) Torch tensor
        :param std_vals:  (D,) Torch tensor
        '''
        self.mean_vals = mean_vals.to(torch.float)
        self.std_vals = std_vals.to(torch.float)
        self.D = self.mean_vals.size(0)

    def normalize(self, state_data):
        '''
        Normalizes the given state data.

        :param state_data: (..., D) if last dim < D only normalizes the first D components

        :return: state_data normalized.
        '''
        cur_D = state_data.size(-1)
        new_size = torch.ones(len(state_data.size())).to(torch.int)
        new_size[-1] = cur_D
        new_size = tuple(new_size)
        cur_mean = self.mean_vals[:cur_D].reshape(new_size).to(state_data.device)
        cur_std = self.std_vals[:cur_D].reshape(new_size).to(state_data.device)
        return (state_data - cur_mean) / cur_std

    def normalize_single(self, state_data, state_idx):
        '''
        Normalizes the given single index of the state data

        :param state_data: (...)

        :return: state_data normalized.
        '''
        cur_mean = self.mean_vals[state_idx].to(state_data.device)
        cur_std = self.std_vals[state_idx].to(state_data.device)
        return (state_data - cur_mean) / cur_std
        

    def unnormalize(self, state_data):
        '''
        Unnormalizes the given state data.

        :param state_data: (..., D)

        :return: state_data unnormalized.
        '''
        cur_D = state_data.size(-1)
        new_size = torch.ones(len(state_data.size())).to(torch.int)
        new_size[-1] = cur_D
        new_size = tuple(new_size)
        cur_mean = self.mean_vals[:cur_D].reshape(new_size).to(state_data.device)
        cur_std = self.std_vals[:cur_D].reshape(new_size).to(state_data.device)
        return (state_data*cur_std) + cur_mean

    def unnormalize_single(self, state_data, state_idx):
        '''
        Unnormalizes the given state data.

        :param state_data: (..., D)

        :return: state_data unnormalized.
        '''
        cur_mean = self.mean_vals[state_idx].to(state_data.device)
        cur_std = self.std_vals[state_idx].to(state_data.device)
        return (state_data*cur_std) + cur_mean

#
# nuscenes normalization statistics
#
BIKE_MAXS = 50.0
BIKE_MAXHDOT = 2.0*np.pi

NUSC_BIKE_PARAMS = {
            'maxs' : BIKE_MAXS,
            'maxhdot' : BIKE_MAXHDOT,
            'dt' : 0.5,
            'a_stats' : (0.409074, 1.045530),
            'ddh_stats' : (0.000046, 0.075032)
}

# mean and std for each
NUSC_NORM_STATS = {
    ('car', 'truck') : {
        'l' : (4.844294, 1.084860),
        'w' : (2.021752, 0.299647),
        's' : (1.802009, 3.507907),
        'h' : (0.0, 1.0), # already a unit vector
        'hdot' : (-0.000037, 0.055684),
        'lscale' : (0.0, 15.0), # must have mean 0! Make heavy use of this assumption in model (when transforming between frames)
        'a' : (0.409074, 1.045530),
        'ddh' : (0.000046, 0.075032)
    },
    # for prediction challenge split - use same normalization as before
    ('bus', 'car', 'construction', 'emergency', 'truck') : {
        'l' : (4.844294, 1.084860),
        'w' : (2.021752, 0.299647),
        's' : (1.802009, 3.507907),
        'h' : (0.0, 1.0), # already a unit vector
        'hdot' : (-0.000037, 0.055684),
        'lscale' : (0.0, 15.0), # must have mean 0! Make heavy use of this assumption in model (when transforming between frames)
        'a' : (0.409074, 1.045530),
        'ddh' : (0.000046, 0.075032)
    },
    # ALL except trailers
    ('bus', 'car', 'construction', 'cyclist', 'emergency', 'motorcycle', 'pedestrian', 'truck') : {
        'l' : (4.844294, 1.084860),
        'w' : (2.021752, 0.299647),
        's' : (1.802009, 3.507907),
        'h' : (0.0, 1.0), # already a unit vector
        'hdot' : (-0.000037, 0.055684),
        'lscale' : (0.0, 15.0), # must have mean 0! Make heavy use of this assumption in model (when transforming between frames)
        'a' : (0.409074, 1.045530),
        'ddh' : (0.000046, 0.075032)
    },
    # for reduced categories
    ('car', 'cyclist', 'motorcycle', 'pedestrian', 'truck') : {
        'l' : (4.844294, 1.084860),
        'w' : (2.021752, 0.299647),
        's' : (1.802009, 3.507907),
        'h' : (0.0, 1.0), # already a unit vector
        'hdot' : (-0.000037, 0.055684),
        'lscale' : (0.0, 15.0), # must have mean 0! Make heavy use of this assumption in model (when transforming between frames)
        'a' : (0.409074, 1.045530),
        'ddh' : (0.000046, 0.075032)
    },
    ('bus', 'car', 'motorcycle', 'trailer', 'truck') : {
        'l' : (5.135896, 2.072248),
        'w' : (2.042160, 0.409259),
        's' : (1.789616, 3.480962),
        'h' : (0.0, 1.0), # already a unit vector
        'hdot' : (-0.000115, 0.058249),
        'lscale' : (0.0, 15.0) # must have mean 0.0! Make heavy use of this assumption in model (when transforming between frames)
    },
    # NOTE no normalization
    ('bus', 'car', 'construction', 'cyclist', 'emergency', 'motorcycle', 'pedestrian', 'trailer', 'truck') : {
        'l' : (0.0, 1.0),
        'w' : (0.0, 1.0),
        's' : (0.0, 1.0),
        'h' : (0.0, 1.0),
        'hdot' : (0.0, 1.0),
        'lscale' : (0.0, 1.0),
        'a' : (0.0, 1.0),
        'ddh' : (0.0, 1.0)
    }
}

#
# nuScenes splits
#

# NOTE: these are our own random train/val splits, not official nuScenes ones
NUSC_VAL_SPLIT_200 = [408,481,190,277,639,10,278,77,125,435,292,38,287,404,424,28,126,622,364,211,386,493,258,354,594,153,561,486,11,571,264,319,350,565,390,189,254,306,382,669,591,219,91,403,67,193,156,242,524,311,620,499,32,240,491,15,621,270,144,207,284,584,214,640,556,42,597,328,405,225,276,338,676,660,632,1,251,406,449,259,48,460,177,135,268,359,626,545,605,348,134,141,631,206,456,550,624,551,426,467,187,198,2,691,617,662,634,501,599,155,255,527,326,35,50,596,299,685,76,414,352,539,142,448,307,303,635,396,479,531,161,471,413,89,249,603,512,474,630,612,654,610,281,290,495,618,569,93,43,218,228,490,51,425,324,30,409,535,446,478,279,688,232,677,331,4,652,650,296,450,188,376,257,552,56,330,20,231,205,47,563,127,8,627,504,103,148,123,473,332,420,81,678,17,656,674,575,494,371,366]
NUSC_VAL_SPLIT_400 = [27,154,689,477,393,42,1,9,95,676,252,427,380,452,214,360,105,101,402,429,331,349,372,115,308,318,201,338,377,527,490,162,215,38,385,234,494,285,311,616,172,107,282,47,541,428,14,673,86,606,418,388,143,188,378,224,480,295,594,463,479,320,572,281,379,431,410,390,321,533,142,357,488,608,69,624,561,699,24,680,602,443,251,73,40,121,255,471,8,79,422,61,316,49,644,538,305,23,681,524,497,509,137,466,579,157,345,526,562,20,396,294,373,409,84,485,123,230,239,337,678,636,623,97,329,85,323,459,161,666,621,581,227,660,88,326,597,268,469,131,464,103,622,670,515,31,548,613,661,191,346,684,78,537,46,468,677,447,217,15,164,619,493,592,653,685,170,353,262,439,112,355,543,395,290,650,194,118,630,34,690,125,322,573,612,225,306,449,275,216,640,77,655,603,5,246,598,266,536,478,601,139,438,474,682,618,599,450,588,626,667,159,391,82,127,33,200,134,229,303,89,496,698,432,41,141,212,507,384,58,499,245,235,2,76,122,614,679,420,580,361,458,336,586,96,430,508,284,312,12,70,334,63,609,335,135,663,309,585,186,256,656,407,475,240,454,444,220,236,487,150,412,249,482,519,367,687,359,522,569,176,94,500,532,652,554,68,93,221,192,278,7,4,32,25,405,19,451,371,279,133,465,299,258,368,87,501,615,486,436,492,560,22,117,552,204,605,218,341,350,697,36,434,167,632,539,319,195,406,178,280,529,483,401,132,550,453,54,351,163,274,421,272,152,356,190,374,369,521,35,66,516,128,182,181,511,232,333,628,171,576,348,631,518,354,277,620,415,196,417,29,160,570,694,0,649,211,557,265,21,583,512,470,177,11,288,457,643,534,633,525,435,376,48,565,411,269,617,248,210]

#
# Scene graph operations
#

def normalize_scene_graph(scene_graph, state_normalizer, att_normalizer, unnorm=False):
    '''
    Norm/unnormalizers past, future, lw, and pos (if contained in the graph).

    WARNING: directly updates scene_graph attributes - does not make a copy.
    '''
    state_norm_func = state_normalizer.unnormalize if unnorm else state_normalizer.normalize
    att_norm_func = att_normalizer.unnormalize if unnorm else att_normalizer.normalize
    if 'past' in scene_graph and len(scene_graph.past.size()) > 1:
        scene_graph.past = state_norm_func(scene_graph.past)
    if 'past_gt' in scene_graph and len(scene_graph.past_gt.size()) > 1:
        scene_graph.past_gt = state_norm_func(scene_graph.past_gt)
    if 'future' in scene_graph and len(scene_graph.future.size()) > 1:
        scene_graph.future = state_norm_func(scene_graph.future)
    if 'future_gt' in scene_graph and len(scene_graph.future_gt.size()) > 1:
        scene_graph.future_gt = state_norm_func(scene_graph.future_gt)
    if 'pos' in scene_graph and len(scene_graph.pos.size()) > 1:
        scene_graph.pos = state_norm_func(scene_graph.pos)
    if 'lw' in scene_graph and len(scene_graph.lw.size()) > 1:
        scene_graph.lw = att_norm_func(scene_graph.lw)
    return scene_graph

def get_ego_inds(scene_graph):
    '''
    Returns mask of first agents in each sequence of a batched scene graph
    '''
    blist = scene_graph.batch.cpu().numpy()
    ego_inds = (blist[1:] - blist[:-1]) == 1
    ego_inds = np.append([True], ego_inds) # first index is always ego
    return ego_inds

def create_subgraph(batched_scene_graph, valid_mask):
    '''
    Removes nodes marked False in the valid mask along with their respective edges and updates
    various graph properties: past, past_gt, future, future_gt, x, pos, lw
    '''
    from torch_geometric.utils import subgraph
    from torch_geometric.data import Data as Graph
    from torch_geometric.data import Batch as GraphBatch

    batch_valid_masks = [valid_mask[batched_scene_graph.ptr[bi]:batched_scene_graph.ptr[bi+1]] for bi in range(batched_scene_graph.num_graphs)]
    sg_list = batched_scene_graph.to_data_list()
    subgraph_list = [] 
    for gi, scene_graph in enumerate(sg_list):
        sub_edge_idx, _ = subgraph(valid_mask, scene_graph.edge_index, relabel_nodes=True)
        cur_mask = batch_valid_masks[gi]
        assert(cur_mask[0].item() is True) # ego should NEVER be removed

        graph_prop_dict = {'edge_index' : sub_edge_idx}
        if hasattr(scene_graph, 'x'):
            graph_prop_dict['x'] = scene_graph.x[cur_mask]
        if hasattr(scene_graph, 'pos'):
            graph_prop_dict['pos'] = scene_graph.pos[cur_mask]
        if hasattr(scene_graph, 'past'):
            graph_prop_dict['past'] = scene_graph.past[cur_mask]
        if hasattr(scene_graph, 'past_gt'):
            graph_prop_dict['past_gt'] = scene_graph.past_gt[cur_mask]
        if hasattr(scene_graph, 'future'):
            graph_prop_dict['future'] = scene_graph.future[cur_mask]
        if hasattr(scene_graph, 'future_gt'):
            graph_prop_dict['future_gt'] = scene_graph.future_gt[cur_mask]
        if hasattr(scene_graph, 'sem'):
            graph_prop_dict['sem'] = scene_graph.sem[cur_mask]
        if hasattr(scene_graph, 'lw'):
            graph_prop_dict['lw'] = scene_graph.lw[cur_mask]
        if hasattr(scene_graph, 'past_vis'):
            graph_prop_dict['past_vis'] = scene_graph.past_vis[cur_mask]
        if hasattr(scene_graph, 'future_vis'):
            graph_prop_dict['future_vis'] = scene_graph.future_vis[cur_mask]

        cur_subgraph = Graph(**graph_prop_dict)
        subgraph_list.append(cur_subgraph)

    out_graph = GraphBatch.from_data_list(subgraph_list)

    return out_graph