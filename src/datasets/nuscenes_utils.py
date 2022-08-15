# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, shutil
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.arcline_path_utils import discretize_lane

from datasets.utils import normalize_scene_graph

def get_nusc_maps(data_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=data_folder,
                 map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

def remove_duplicates(xy, eps):
    """Ensures that returned array always have at least 2
    rows.
    """
    N,_ = xy.shape
    assert(N >= 2), N

    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    kept = np.ones(len(xy), dtype=bool)
    kept[:-1] = dist > eps
    assert(kept[0]), kept

    return xy[kept]

def check_duplicates(xy, eps):
    assert(xy.shape[0] >= 2), xy.shape
    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    assert(np.all(dist > eps)), dist


def process_lanegraph(nmap, res_meters, eps):
    """lanes are
    {
        xys: n x 2 array (xy locations)
        in_edges: n x X list of lists
        out_edges: n x X list of lists
        edges: m x 5 (x,y,hcos,hsin,l)
        edgeixes: m x 2 (v0, v1)
        ee2ix: dict (v0, v1) -> ei
    }
    """
    lane_graph = {}

    # discretize the lanes (each lane has at least 2 points)
    for lane in nmap.lane + nmap.lane_connector:
        my_lane = nmap.arcline_path_3.get(lane['token'], [])
        discretized = np.array(discretize_lane(my_lane, res_meters))[:, :2]
        discretized = remove_duplicates(discretized, eps)
        check_duplicates(discretized, eps) # make sure each point is at least eps away from neighboring points
        lane_graph[lane['token']] = discretized

    # make sure connections don't have duplicates (now each lane has at least 1 point)
    for intok,conn in nmap.connectivity.items():
        for outtok in conn['outgoing']:
            if outtok in lane_graph:
                dist = np.linalg.norm(lane_graph[outtok][0] - lane_graph[intok][-1])
                if dist <= eps:
                    lane_graph[intok] = lane_graph[intok][:-1]
                    assert(lane_graph[intok].shape[0] >= 1), lane_graph[intok]

    xys = []
    laneid2start = {}
    for lid,lane in lane_graph.items():
        laneid2start[lid] = len(xys)
        xys.extend(lane.tolist())

    in_edges = [[] for _ in range(len(xys))]
    out_edges = [[] for _ in range(len(xys))]
    for lid,lane in lane_graph.items():
        for ix in range(len(lane)-1):
            out_edges[laneid2start[lid]+ix].append(laneid2start[lid]+ix+1)
        for ix in range(1, len(lane)):
            in_edges[laneid2start[lid]+ix].append(laneid2start[lid]+ix-1)
        for outtok in nmap.connectivity[lid]['outgoing']:
            if outtok in lane_graph:
                out_edges[laneid2start[lid]+len(lane)-1].append(laneid2start[outtok])
        for intok in nmap.connectivity[lid]['incoming']:
            if intok in lane_graph:
                in_edges[laneid2start[lid]].append(laneid2start[intok]+len(lane_graph[intok])-1)

    # includes a check that the edges are all more than length eps
    edges, edgeixes, ee2ix = process_edges(xys, out_edges, eps)

    return {'xy': np.array(xys), 'in_edges': in_edges, 'out_edges': out_edges,
            'edges': edges, 'edgeixes': edgeixes, 'ee2ix': ee2ix}

def process_edges(xys, out_edges, eps):
    edges = []
    edgeixes = []
    ee2ix = {}
    for i in range(len(out_edges)):
        x0,y0 = xys[i]
        for e in out_edges[i]:
            x1,y1 = xys[e]
            diff = np.array([x1-x0, y1-y0])
            dist = np.linalg.norm(diff)
            assert(dist>eps), dist
            diff /= dist
            assert((i,e) not in ee2ix)
            ee2ix[i,e] = len(edges)
            edges.append([x0, y0, diff[0], diff[1], dist])
            edgeixes.append([i, e])
    return np.array(edges), np.array(edgeixes), ee2ix

def get_grid(point_cloud_range, voxel_size):
    lower = np.array(point_cloud_range[:(len(point_cloud_range) // 2)])
    upper = np.array(point_cloud_range[(len(point_cloud_range) // 2):])

    dx = np.array(voxel_size)
    bx = lower + dx/2.0
    nx = ((upper - lower) / dx).astype(int)

    return dx, bx, nx

def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (B)
    :param theta2: angle 2 (B)
    :return diff: smallest angle difference between angles (B)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

def heading_change_rate(h, t):
    '''
    Given heading angles may be nan. If so returns nans for these frames. Velocity are computeed using
    backward finite differences except for leading frames which use forward finite diff. Any single
    frames (i.e have no previous or future steps) are nan.

    :param h: heading angles (T)
    :param t: timestamps (T) in sec
    :return hdot: (T)
    ''' 
    hdiff = angle_diff(h[1:], h[:-1]) / (t[1:] - t[:-1])
    hdot = np.append(hdiff[0:1], hdiff) # for first frame use forward diff

    # for any nan -> value transition frames, want to use forward difference
    hnan = np.isnan(h).astype(np.int)
    if np.sum(hnan) == 0:
        return hdot
    lead_nans = (hnan[1:] - hnan[:-1]) == -1
    lead_nans = np.append([False], lead_nans)
    repl_idx = np.append([False], lead_nans[:-1])
    num_fill = np.sum(repl_idx.astype(np.int))
    if num_fill != 0:
        if num_fill != np.sum(lead_nans.astype(np.int)):
            # the last frame is a leading nan, have to ignore it
            lead_nans[-1] = False
        hdot[lead_nans] = hdot[repl_idx]
    return hdot

def velocity(pos, t):
    '''
    Given positions may be nan. If so returns nans for these frames. Velocity are computeed using
    backward finite differences except for leading frames which use forward finite diff. Any single
    frames (i.e have no previous or future steps) are nan.

    :param pos: positions (T x D)
    :param t: timestamps (T) in sec
    :return vel: (T x D)
    ''' 
    vel_diff = (pos[1:, :] - pos[:-1, :]) / (t[1:] - t[:-1]).reshape((-1, 1))
    vel = np.concatenate([vel_diff[0:1,:], vel_diff], axis=0) # for first frame use forward diff

    # for any nan -> value transition frames, want to use forward difference
    posnan = np.isnan(np.sum(pos, axis=1)).astype(np.int)
    if np.sum(posnan) == 0:
        return vel
    lead_nans = (posnan[1:] - posnan[:-1]) == -1
    lead_nans = np.append([False], lead_nans)
    repl_idx = np.append([False], lead_nans[:-1])
    num_fill = np.sum(repl_idx.astype(np.int))
    if num_fill != 0:
        if num_fill != np.sum(lead_nans.astype(np.int)):
            # the last frame is a leading nan, have to ignore it
            lead_nans[-1] = False
        vel[lead_nans] = vel[repl_idx]
    return vel

#
# Cropping maps around agent
# 

def gen_car_coords(xys, hs, C, L, W, bounds=None, ls=None, ws=None):
    """
    we want B x C x L x W x 2

    bounds: list of size 4 [low_l, low_w, high_l, high_w]
    ls/ws: B
    """
    B = hs.size(0)
    device = hs.device

    # build grid in local frame
    if bounds is not None:
        lwise = torch.linspace(bounds[0], bounds[2], L, device=device).view(1, 1, L, 1).expand(B, C, L, W)
        wwise = torch.linspace(bounds[1], bounds[3], W, device=device).view(1, 1, 1, W).expand(B, C, L, W)
    elif ls is not None and ws is not None:
        lwise = torch.linspace(-1.0, 1.0, L, device=device).view(1, 1, L, 1).expand(B, C, L, W) * ls.view(B,1,1,1)/2
        wwise = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, C, L, W) * ws.view(B,1,1,1)/2
    else:
        print('Must either passs in bound or ls and ws to get local coords')
        exit()

    # rotate to global frame
    hcos = hs[:,0].view(B, 1, 1, 1)
    hsin = hs[:,1].view(B, 1, 1, 1)
    xys = torch.stack((lwise * hcos - wwise * hsin,
                       lwise * hsin + wwise * hcos), 4) + xys.view(B, 1, 1, 1, 2)

    return xys

def get_map_obs(maps, dx, frame, mapixes, bounds, L=256, W=256):
    '''
    Get map crop in frame of car (centered at position of car).

    :param maps: binarized maps to crop from (M x C x H x W)
    :param dx: m / pix ratio for each dim of each map M x 2
    :param frame: B x 4 local frame to crop map in (x,y,hx,hy)
    :param mapixes: (B, ) the map index into maps/dx to use for each crop
    :param bounds: List of distance from frame to crop [behind, left, front, right]
    :param L: resolution of crop along length of car
    :param W: resolution of crop along width of the car
    '''
    B = frame.size(0)
    C = maps.size(1)

    # sample grid aligned with agent coords in the world space
    xys = gen_car_coords(frame[:, :2], frame[:, 2:], C, L, W, bounds=bounds)
    xys[torch.isnan(xys)] = 0.0 # make any nan frames just default to value at (0,0)

    # convert to pixel coordinates
    xys = xys / dx[mapixes].view(B, 1, 1, 1, 2)
    xys = torch.round(xys).long()

    front_ixes = mapixes.view(B, 1, 1, 1).expand(B, C, L, W)
    chan_ixes = torch.arange(C).view(1, C, 1, 1).expand(B, C, L, W)

    # TODO Warning! forces xys outside the image to be set to the map at pixel (0, 0)
    outside = (xys[:, :, :, :, 1] < 0) | (xys[:, :, :, :, 1] >= maps.shape[2]) | (xys[:, :, :, :, 0] < 0) | (xys[:, :, :, :, 0] >= maps.shape[3])
    xys[outside] = 0
    obs = maps[front_ixes, chan_ixes, xys[:, :, :, :, 1], xys[:, :, :, :, 0]]
    return obs

def check_on_layer(drivables, dx, cars, lw, mapixes):
    '''
    Compute fraction of car that is on area of the map marked 1.

    :param drivables: binarized drivable maps (or other layer type) (M x H x W)
    :param dx: m / pix ratio for each dim of each map M x 2
    :param cars: B x 4 car kinematics to check (x,y,hx,hy)
    :param lw: B x 2 car attributes to check (l,w)
    :param mapixes: (B, ) the map index into maps/dx to use for each check
    '''
    # set L and W using dx
    mdx = torch.mean(dx)
    mlw = torch.mean(lw, dim=0)
    L = torch.round(mlw[0] / mdx).int().item()
    W = torch.round(mlw[1] / mdx).int().item()

    B = cars.size(0)
    # sample grid aligned with agent coords in the world space
    xys = gen_car_coords(cars[:, :2], cars[:, 2:], 1, L, W, ls=lw[:,0], ws=lw[:,1])[:,0] # don't need multiple channels

    # convert to pixel coordinates
    xys = xys / dx[mapixes].view(B, 1, 1, 2)
    xys = torch.round(xys).long()

    front_ixes = mapixes.view(B, 1, 1).expand(B, L, W)

    # TODO Warning! forces xys outside the image to be set to the map at pixel (0, 0)
    outside = (xys[:, :, :, 1] < 0) | (xys[:, :, :, 1] >= drivables.shape[1]) | (xys[:, :, :, 0] < 0) | (xys[:, :, :, 0] >= drivables.shape[2])
    xys[outside] = 0
    # 1's mean drivable
    car_pix = drivables[front_ixes, xys[:, :, :, 1], xys[:, :, :, 0]]
    driv_frac = torch.sum(car_pix.float(), dim=[1,2]) / (L*W)
    return driv_frac

def check_line_layer(drivables, dx, start, end, mapixes):
    '''
    Checks if the line from start to end (in world space) intersects with the
    given map layer.

    :param drivables: binarized drivable maps (or other layer type) (M x H x W)
    :param dx: m / pix ratio for each dim of each map M x 2
    :param start: B x 2 start of lines (x,y)
    :param end: B x 2 end of lines (x,y)
    :param mapixes: (B, ) the map index into maps/dx to use for each check

    :return line_itsct: true if the line intersects with binarized non-drivable region.
    '''    
    B = start.size(0)
    line_len = torch.norm(start - end, dim=-1)
    # compute number of bins for indexing
    mdx = torch.mean(dx)
    L =  torch.max(torch.round(line_len / mdx).int()).item()
    # get rasterization for each line
    line_w = torch.linspace(0.0, 1.0, L).view(1, L, 1).expand(B, L, 2).to(start.device)
    line_interp = start.view(B, 1, 2)*(1.0 - line_w) + end.view(B, 1, 2)*line_w # (B, L, 2)

    # convert to pixel coordinates
    xys = line_interp / dx[mapixes].view(B, 1, 2)
    xys = torch.round(xys).long()

    front_ixes = mapixes.view(B, 1).expand(B, L)
    # 1's mean drivable
    line_pix = drivables[front_ixes, xys[:, :, 1], xys[:, :, 0]]
    line_non_drivable = line_pix == 0
    line_itsct = torch.sum(line_non_drivable, dim=-1) > 0

    return line_itsct

def get_coll_point(drivables, dx, cars, lw, mapixes,
                    return_iou=False):
    '''
    Estimates point of collision with the given map. This is in meters in world space.
    If no collision or vehicle is fully in the non-drivable area, returns nan. 

    :param drivables: binarized drivable maps (or other layer type) (M x H x W)
    :param dx: m / pix ratio for each dim of each map M x 2
    :param cars: B x 4 car kinematics to check (x,y,hx,hy)
    :param lw: B x 2 car attributes to check (l,w)
    :param mapixes: (B, ) the map index into maps/dx to use for each check

    :return: collision_pnt (B,)
    :return: collision_iou (B, ) not actually IoU... it's actually the fraction of 
                                area within the car that is on non-drivable.
    '''
    # set L and W using dx
    mdx = torch.mean(dx)*0.5 # make a bit more fine-grained
    mlw = torch.mean(lw, dim=0)
    L = torch.round(mlw[0] / mdx).int().item()
    W = torch.round(mlw[1] / mdx).int().item()

    B = cars.size(0)
    # sample grid aligned with agent coords in the world space
    xys_world = gen_car_coords(cars[:, :2], cars[:, 2:], 1, L, W, ls=lw[:,0], ws=lw[:,1])[:,0] # B x L x W x 2

    # convert to pixel coordinates
    xys = xys_world / dx[mapixes].view(B, 1, 1, 2)
    xys = torch.round(xys).long()

    front_ixes = mapixes.view(B, 1, 1).expand(B, L, W)

    # TODO Warning! forces xys outside the image to be set to the map at pixel (0, 0)
    outside = (xys[:, :, :, 1] < 0) | (xys[:, :, :, 1] >= drivables.shape[1]) \
                    | (xys[:, :, :, 0] < 0) | (xys[:, :, :, 0] >= drivables.shape[2])
    xys[outside] = 0
    # 1's mean drivable
    car_pix = drivables[front_ixes, xys[:, :, :, 1], xys[:, :, :, 0]].unsqueeze(-1) # B x L x W x 1
    non_drivable = car_pix == 0 # torch.where(car_pix == 0, torch.ones_like(car_pix), torch.ones_like(car_pix)*np.nan)

    # roughly approx collision point with mean of non-drivable pixels in the car
    # NOTE: this will be incorrect when multiple connected components of non-drivable pixels show up.
    num_non_drivable = torch.sum(non_drivable, dim=(1,2))
    # NOTE: only want vehicles that are partially in non-drivable
    coll_pt = (xys_world * non_drivable).sum(dim=(1,2)) # (B, 2)
    coll_pt = coll_pt / num_non_drivable # results in nan for any vehicle fully on drivable
    off_road_mask = num_non_drivable[:,0] == L*W
    coll_pt[off_road_mask] = np.nan # set manually for fully off road vehicles

    if return_iou:
        coll_iou = num_non_drivable[:, 0] / float(L*W)
        # make sure nans are the same as for coll_pt
        coll_iou[coll_iou == 0] = np.nan # totally on road
        coll_iou[off_road_mask] = np.nan # totally off road
        return coll_pt, coll_iou
    else:
        return coll_pt

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def objects2frame(history, center, toworld=False):
    N, A, B = history.shape
    theta = np.arctan2(center[3], center[2])
    if not toworld:
        newloc = history[:, :, :2] - center[:2].reshape((1, 1, 2))
        rot = get_rot(theta).T
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) - theta
        newloc = np.dot(newloc.reshape((N*A, 2)), rot).reshape((N, A, 2))
    else:
        rot = get_rot(theta)
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) + theta
        newloc = np.dot(history[:, :, :2].reshape((N*A, 2)),
                        rot).reshape((N, A, 2))
    newh = np.stack((np.cos(newh), np.sin(newh)), 2)
    if toworld:
        newloc += center[:2]
    return np.append(newloc, newh, axis=2)

def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box

#
# Visualization
# 

def get_adv_coloring(NA, attack_agt, tgt_agt,
                    alpha=False,
                    linewidth=False,
                    markersize=False,
                    cycle_color=False):
    ''' Returns color list for visualizing number of agents where
        the attacking car is shown in green and target in red '''
    if cycle_color:
        base_colors = [plt_color(n % 9) for n in range(NA)]
    else:
        base_colors = ['cornflowerblue']*NA
    if attack_agt is not None:
        base_colors[attack_agt] = 'orangered'
    if tgt_agt is not None:
        base_colors[tgt_agt] = 'lawngreen'
    out_list = [base_colors]
    if alpha:
        base_alpha = [1.0]*NA
        if attack_agt is not None:
            base_alpha[attack_agt] = 1.0
        if tgt_agt is not None:
            base_alpha[tgt_agt] = 1.0
        out_list.append(base_alpha)
    if linewidth:
        base_lw = [2.5]*NA
        if attack_agt is not None:
            base_lw[attack_agt] = 3.5
        if tgt_agt is not None:
            base_lw[tgt_agt] = 3.5
        out_list.append(base_lw)
    if markersize:
        base_msize = [10.0]*NA
        if attack_agt is not None:
            base_msize[attack_agt] = 15.0
        if tgt_agt is not None:
            base_msize[tgt_agt] = 15.0
        out_list.append(base_msize)
    
    if len(out_list) > 1:
        return tuple(out_list)
    else:
        return out_list[0]

def viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path,
                    state_normalizer, att_normalizer,
                    future_pred=None,
                    aidx=None,
                    viz_traj=False,
                    make_video=True,
                    show_gt=False,
                    traj_color_val=None,
                    traj_color_bounds=None,
                    viz_bounds=[-17.0, -38.5, 60.0, 38.5],
                    center_viz=False,
                    car_colors=None,
                    show_gt_idx=None,
                    crop_t=None,
                    ow_gt=None):
    '''
    Given a batch of scene graphs, visualizes the scene at the requested batch index.

    :param scene_graph: the scenes to visualize from, should contain past and future. Assumed NORMALIZED.
    :param map_idx: (B,) the maps corresponding to each scene in the batch
    :param map_env: map environment used to render map crops
    :param bidx: which scene in the batch to visualize
    :param out_path: path previx to write out (extension .png or .mp4 will be added)
    :param state_normalizer:
    :param att_normalizer:
    :param future_pred: (NA, FT, 4) or (NA, NS, FT, 4) if given, overrides the .future attribute in the scene graph 
                        and will be visulized instead. Assumed NORMALIZED.
    :param aidx: if given, only visualizes the agent at node aidx in the requested scene
    :param show_gt: if True, shows the scene graph future along with future pred if both given.
    :param future_mdist: future mahalanobis distance of samples in prior distrib.
    :param car_colors: list of size NA, color to use for visualizing each car.
    :param ow_gt: a NORMALIZED GT trajectory to use rather than scene_graph.future_gt
    '''
    if show_gt_idx is not None:
        show_gt = True
    # first must unnormalize to get true coords
    scene_graph = normalize_scene_graph(scene_graph,
                                        state_normalizer,
                                        att_normalizer,
                                        unnorm=True)
    # also unnormalize future pred if necessary
    NS = 1
    NA, PT, _ = scene_graph.past_gt.size()
    FT = scene_graph.future_gt.size(1)
    if future_pred is not None:
        future_pred = state_normalizer.unnormalize(future_pred)
        if len(future_pred.size()) == 4:
            NS = future_pred.size(1)
            FT = future_pred.size(2)
        else:
            FT = future_pred.size(1)
            future_pred = future_pred.view(NA, 1, FT, 4)
        
    binds = scene_graph.batch == bidx
    if center_viz:
        # viz around centroid of all agents
        fut_traj = future_pred[binds][:,:,:,:2] if future_pred is not None else scene_graph.future_gt[:,:,:2].view(NA, 1, FT, 2)[binds]
        past_traj = scene_graph.past_gt[:,:,:2].view(NA, 1, PT, 2).expand(NA, NS, PT, 2)[binds]
        all_traj = torch.cat([past_traj, fut_traj], dim=2).view(-1, 2)
        val_mask = torch.isnan(all_traj).sum(1) == 0
        viz_cent = torch.mean(all_traj[val_mask], dim=0)
        crop_pos = torch.cat([viz_cent.view(1, 2), torch.Tensor([[1.0, 0.0]]).to(viz_cent)], dim=1)
        crop_pos = crop_pos.expand(NA, 4)
    else:
        # in past frame -1 for now
        # crop it around the cars themselves, will use ego vehicle later
        if crop_t is None:
            crop_pos = scene_graph.past_gt[:, -1, :4] # (NA, 4)
        else:
            # crop based on a specific time step
            crop_in_past = crop_t < PT
            if crop_in_past:
                crop_pos = scene_graph.past_gt[:, crop_t, :4] # (NA, 4)
            else:
                crop_pos = scene_graph.future_gt[:, crop_t-PT, :4] # (NA, 4)

    scene_graph.pos = crop_pos
    rend_res = map_env.get_map_crop(scene_graph, map_idx, bounds=viz_bounds)

    viz_past = scene_graph.past_gt[binds][:, :, :4]
    NA = viz_past.size(0) # new number of agents for only desired sequence
    viz_past = viz_past.view(NA, 1, PT, 4).expand(NA, NS, PT, 4)
    viz_future = scene_graph.future_gt[binds][:, :, :4] if future_pred is None \
                        else future_pred[binds][:, :, :, :4]
    if future_pred is None:
        viz_future = viz_future.view(NA, 1, FT, 4)

    viz_objs = torch.cat([viz_past, viz_future], dim=2)
    viz_lw = scene_graph.lw[binds]
    NT = viz_objs.size(2)
    # convert to pixel crop coordinates
    # by default do it in frame of ego vehicle (idx 0)
    map_crop = rend_res[binds][0]
    crop_objs, crop_lw = map_env.objs2crop(crop_pos[binds][0],
                                            viz_objs.view(NA*NS*NT, 4),
                                            viz_lw,
                                            map_idx[bidx],
                                            bounds=viz_bounds)
    crop_objs = crop_objs.reshape(NA, NS, NT, 4)
    if aidx is not None:
        crop_objs = crop_objs[aidx:aidx+1]

    gt_crop_objs = None
    if show_gt and future_pred is not None:
        gt_NT = scene_graph.future_gt.size(1) + PT
        if ow_gt is not None:
            gt_future = state_normalizer.unnormalize(ow_gt[binds])
        else:
            gt_future = scene_graph.future_gt[binds][:, :, :4]
        gt_future = gt_future.view(NA, 1, gt_future.size(1), 4)
        gt_objs = torch.cat([viz_past[:, 0:1], gt_future], dim=2)
        gt_crop_objs, _ = map_env.objs2crop(crop_pos[binds][0],
                                            gt_objs.view(NA*gt_NT, 4),
                                            viz_lw,
                                            map_idx[bidx],
                                            bounds=viz_bounds)
        gt_crop_objs = gt_crop_objs.reshape(NA, 1, gt_NT, 4)
        if aidx is not None:
            gt_crop_objs = gt_crop_objs[aidx:aidx+1]
        if show_gt_idx is not None:
            gt_crop_objs = gt_crop_objs[show_gt_idx:show_gt_idx+1] 

    # viz image
    viz_map_crop(map_crop.cpu(), out_path + '.png',
                    crop_objs.cpu(),
                    crop_lw.cpu(),
                    gt_kin=gt_crop_objs,
                    viz_traj=viz_traj,
                    indiv=False,
                    color_traj=traj_color_val,
                    color_traj_bounds=traj_color_bounds,
                    car_colors=car_colors)
    if make_video:
        # viz video
        viz_map_crop_video(map_crop.cpu(), out_path,
                                crop_objs.cpu(),
                                crop_lw.cpu(),
                                viz_traj=viz_traj,
                                indiv=False,
                                car_colors=car_colors)

    # re-normalize scene graph so same as passed in.
    scene_graph = normalize_scene_graph(scene_graph,
                                        state_normalizer,
                                        att_normalizer)

def create_video(img_path_form, out_path, fps):
    '''
    Creates a video from a format for frame e.g. 'data_out/frame%04d.png'.
    Saves in out_path.
    '''
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-r', str(fps), '-i', img_path_form,
                    '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_path])

def viz_map_crop_video(x, outdir, car_kin=None, car_lw=None, viz_traj=False, indiv=True,
                        car_colors=None, car_alpha=None, fps=2):
    '''
    :param x: cropped rasterized layers (N, H, W)
    :param car_kin: kinematics (NA x T x 4) or (NA x NS x T x 4)
    :param car_lw: attributes length, width (NA x 2)
    '''
    if len(car_kin.size()) == 3:
        NA, NT, _ = car_kin.size()
        car_kin = car_kin.view(NA, 1, NT, 4)
    NA, NS, NT, _= car_kin.size()
    for s in range(NS):
        cur_outdir = outdir + '_%03d' % (s)
        if not os.path.exists(cur_outdir):
            os.makedirs(cur_outdir)
        for t in range(NT):
            viz_map_crop(x, os.path.join(cur_outdir, 'frame%04d.png' % (t)),
                        car_kin=car_kin[:, s:s+1, t:t+1, :], car_lw=car_lw, 
                        viz_traj=viz_traj, indiv=indiv, car_colors=car_colors,
                        car_alpha=car_alpha)
        create_video(os.path.join(cur_outdir, 'frame%04d.png'), cur_outdir + '.mp4', fps)
        shutil.rmtree(cur_outdir)

def viz_map_crop(x, out_path, car_kin=None, car_lw=None, gt_kin=None, viz_traj=False, indiv=True,
                 color_traj=None, color_traj_bounds=None, car_colors=None, car_alpha=None, traj_linewidth=None,
                 traj_markers=None, traj_markersize=None):
    '''
    :param x: cropped rasterized layers (N, H, W)
    :param car_kin: kinematics (NA x T x 4) or (NA x NS x T x 4)
    :param car_lw: attributes length, width (NA x 2)
    '''
    def style_ax():
        plt.grid(b=None)
        plt.xticks([])
        plt.yticks([])

    if not indiv:
        # fig = plt.figure(figsize=(6,6))
        fig = plt.figure(dpi=200)
    else:
        fig = plt.figure(figsize=(6*(x.shape[0]+1), 6))
        gs = mpl.gridspec.GridSpec(1, x.shape[0]+1)

        for i in range(x.shape[0]):
            plt.subplot(gs[0, i])
            # transpose because Map crops are pulled out from the base map in a transposed manner...
            plt.imshow(x[i].T, origin='lower', vmin=0, vmax=1)

        plt.subplot(gs[0, i+1])

    render_map_observation(x)
    if gt_kin is not None and car_lw is not None:
        gt_car_colors = ['white']*gt_kin.size(0)
        render_obj_observation(gt_kin, car_lw, viz_traj=viz_traj, viz_init_car=False, color=gt_car_colors)
    if car_kin is not None and car_lw is not None:
        render_obj_observation(car_kin, car_lw, viz_traj=viz_traj,
                            color_traj=color_traj, color_traj_bounds=color_traj_bounds,
                            color=car_colors,
                            alpha=car_alpha,
                            traj_linewidth=traj_linewidth,
                            traj_markers=traj_markers,
                            traj_markersize=traj_markersize)


    style_ax()
    plt.xlim(0, x.shape[2])
    plt.ylim(0, x.shape[1])
    plt.tight_layout()
    print('saving', out_path)
    plt.savefig(out_path, bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

def make_rgba(probs, color):
    H, W = probs.shape
    return np.stack((
                     np.full((H, W), color[0]),
                     np.full((H, W), color[1]),
                     np.full((H, W), color[2]),
                     probs,
                     ), 2)

def plt_color(i):
    clist = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return clist[i]

def render_map_observation(x):
    # ['drivable_area', 'carpark_area', 'road_divider', 'lane_divider', 'walkway', 'ped_crossing']
    map_color_list = ['darkgray', 'coral', 'orange', 'gold', 'lightblue', 'lightblue']
    map_alpha_list = [1.0, 0.6, 0.6, 0.6, 1.0, 0.5]

    bg = np.ones_like(x[0].T)
    c = mpl.colors.to_rgba('white')[:3]
    showimg = make_rgba(bg, c)
    plt.imshow(showimg, origin='lower')

    for i in range(x.shape[0]):
        pltcolor = map_color_list[i % 9]
        c = mpl.colors.to_rgba(pltcolor)[:3]
        showimg = make_rgba(x[i].T*map_alpha_list[i], c)
        plt.imshow(showimg, origin='lower')

def render_obj_observation(car_kin, car_lw, viz_traj=False, viz_init_car=True, color=None,
                            color_traj=None, color_traj_bounds=None,
                            alpha=None, traj_linewidth=None, traj_markers=None, traj_markersize=None):
    traj_cmap = 'gist_rainbow'

    use_color_traj = color_traj is not None and color_traj_bounds is not None
    if use_color_traj:
        color_traj_cmap = mpl.cm.get_cmap('plasma')
        color_traj_norm = mpl.colors.Normalize(vmin=color_traj_bounds[0], vmax=color_traj_bounds[1])
    if car_kin is not None and car_lw is not None:
        if viz_traj:
            if traj_markers is None:
                traj_markers = [True]*car_kin.size(0)
            
            # only show position trajectories
            if len(car_kin.size()) == 3:
                NA, T, _ = car_kin.size()
                for n in range(NA):
                    cur_traj = car_kin[n]
                    val_steps = torch.sum(torch.isnan(cur_traj), dim=1) == 0
                    cur_traj = cur_traj[val_steps]
                    carr = np.linspace(0, 1.0, T)[val_steps]
                    NT = cur_traj.size(0)
                    cur_alph = alpha[n] if alpha is not None else 0.7
                    cur_linewidth = traj_linewidth[n] if traj_linewidth is not None else 1.0
                    cur_msize = traj_markersize[n] if traj_markersize is not None else 8.0
                    if cur_traj.size(0) > 0:
                        cur_col = color[n] if color is not None else plt_color(n % 9)
                        plt.plot(cur_traj[:, 0], cur_traj[:, 1], '-', c=cur_col, alpha=cur_alph, linewidth=cur_linewidth)
                    if traj_markers[n]:
                        plt.scatter(cur_traj[:, 0], cur_traj[:, 1], c=carr, s=cur_msize, norm=mpl.colors.Normalize(0.0, 1.0),
                                    cmap=traj_cmap, alpha=cur_alph)
                    else:
                        cur_col = color[n] if color is not None else plt_color(n % 9)
                        plt.scatter(cur_traj[:, 0], cur_traj[:, 1], c=cur_col, s=cur_msize, alpha=cur_alph)
            elif len(car_kin.size()) == 4:
                NA, NS, T, _ = car_kin.size()
                for n in range(NA):
                    for s in range(NS):
                        cur_traj = car_kin[n, s]
                        val_steps = torch.sum(torch.isnan(cur_traj), dim=1) == 0
                        cur_traj = cur_traj[val_steps]
                        carr = np.linspace(0, 1.0, T)[val_steps]
                        NT = cur_traj.size(0)
                        cur_alph = alpha[n] if alpha is not None else 0.7
                        cur_linewidth = traj_linewidth[n] if traj_linewidth is not None else 1.0
                        cur_msize = traj_markersize[n] if traj_markersize is not None else 8.0
                        if cur_traj.size(0) > 0:
                            if use_color_traj:
                                cur_col = color_traj_cmap(color_traj_norm(color_traj[n, s].cpu().item()))
                            else:
                                cur_col = color[n] if color is not None else plt_color(n % 9)
                            plt.plot(cur_traj[:, 0], cur_traj[:, 1], '-', c=cur_col, alpha=cur_alph, linewidth=cur_linewidth)
                        if not use_color_traj and traj_markers[n]:
                            plt.scatter(cur_traj[:, 0], cur_traj[:, 1], c=carr, s=cur_msize, norm=mpl.colors.Normalize(0.0, 1.0), 
                                        cmap=traj_cmap, alpha=cur_alph)
                        elif not use_color_traj:
                            cur_col = color[n] if color is not None else plt_color(n % 9)
                            plt.scatter(cur_traj[:, 0], cur_traj[:, 1], c=cur_col, s=cur_msize, alpha=cur_alph)

            if viz_init_car:
                # always plot the car at first frame
                if len(car_kin.size()) == 3:
                    NA, _, _ = car_kin.size()
                    for n in range(NA):
                        # if torch.sum(torch.isnan(car_kin[n, 0])) == 0:
                        cur_col = color[n] if color is not None else plt_color(n % 9)
                        cur_alph = alpha[n] if alpha is not None else 0.7
                        nonnan_idx = torch.nonzero(~torch.isnan(car_kin[n, :, 0]), as_tuple=True)[0][0]
                        plot_car(car_kin[n, nonnan_idx, 0], car_kin[n, nonnan_idx, 1], car_kin[n, nonnan_idx, 2], car_kin[n, nonnan_idx, 3],
                                    car_lw[n, 0], car_lw[n, 1], cur_col, alpha=cur_alph)
                elif len(car_kin.size()) == 4:
                    NA, NS, _, _ = car_kin.size()
                    for n in range(NA):
                        for s in range(1):
                            cur_col = color[n] if color is not None else plt_color(n % 9)
                            cur_alph = alpha[n] if alpha is not None else 0.7
                            nonnan_idx = torch.nonzero(~torch.isnan(car_kin[n, s, :, 0]), as_tuple=True)[0][0]
                            plot_car(car_kin[n, s, nonnan_idx, 0], car_kin[n, s, nonnan_idx, 1], car_kin[n, s, nonnan_idx, 2], car_kin[n, s, nonnan_idx, 3],
                                        car_lw[n, 0], car_lw[n, 1], cur_col, alpha=cur_alph)
        else:
            # otherwise plot the car
            if len(car_kin.size()) == 3:
                NA, T, _ = car_kin.size()
                for n in range(NA):
                    for t in range(T):
                        if torch.sum(torch.isnan(car_kin[n, t])) == 0:
                            cur_alpha = alpha[n] if alpha is not None else 0.7
                            cur_alpha = cur_alpha if t == 0 else 0.1 + (1.0 - (float(t) / T)) * 0.2 # decrease visiblity over time
                            cur_col = color[n] if color is not None else plt_color(n % 9)
                            plot_car(car_kin[n, t, 0], car_kin[n, t, 1], car_kin[n, t, 2], car_kin[n, t, 3],
                                        car_lw[n, 0], car_lw[n, 1], cur_col, alpha=cur_alpha)
            elif len(car_kin.size()) == 4:
                NA, NS, T, _ = car_kin.size()
                for n in range(NA):
                    for s in range(NS):
                        for t in range(T):
                            if torch.sum(torch.isnan(car_kin[n, s, t])) == 0:
                                cur_alpha = alpha[n] if alpha is not None else 0.7
                                cur_alpha = cur_alpha if t == 0 else 0.1 + (1.0 - (float(t) / T)) * 0.2 # decrease visiblity over time
                                cur_col = color[n] if color is not None else plt_color(n % 9)
                                plot_car(car_kin[n, s, t, 0], car_kin[n, s, t, 1], car_kin[n, s, t, 2], car_kin[n, s, t, 3],
                                            car_lw[n, 0], car_lw[n, 1], cur_col, alpha=cur_alpha)

def plot_box(box, lw, color='g', alpha=0.7, no_heading=False):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array([
        box[:2],
        box[:2] + l/2.*np.array([np.cos(h), np.sin(h)]),
    ])

    plt.fill(simple_box[:, 0], simple_box[:, 1], color=color, edgecolor='k', alpha=alpha, linewidth=1.0, zorder=3)
    if not no_heading:
        plt.plot(arrow[:, 0], arrow[:, 1], 'k', alpha=alpha, zorder=3)

def plot_car(x, y, hx, hy, l, w, color='b', alpha=0.7, no_heading=False):
    plot_box(np.array([x.item(), y.item(), hx.item(), hy.item()]), [l.item(), w.item()],
             color=color, alpha=alpha, no_heading=no_heading)

