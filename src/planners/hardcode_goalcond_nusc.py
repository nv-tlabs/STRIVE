# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, shutil
from collections import deque
from copy import deepcopy

import torch
import numpy as np
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from planners.planner import PlannerNusc, PlannerConfig
from datasets.nuscenes_utils import create_video

from utils.torch import get_device, count_params, save_state, load_state, compute_kl_weight, c2c

# un-tuned defaults
DEF_CONFIG = {
    'dt' : 0.2, 
    'preddt' : 0.2,
    'nsteps' : 25,
    'cdistang' : 20.0,
    'xydistmax' : 2.0,
    'smax' : 15.0,
    'accmax' : 3.0,
    'predsfacs' : [0.5, 1.0],
    'predafacs' : [0.5],
    'interacdist' : 70.0,
    'planaccfacs' : [1.0],
    'plannspeeds' : 5,
    'col_plim' : 0.1,
    'score_wmin' : 0.7,
    'score_wfac' : 0.05
}

# large-scale tuned on generated scenarios from validation set
TUNED_VAL_FINAL_1 = {
    'dt' : 0.2, 
    'preddt' : 0.2,
    'nsteps' : 25,
    'cdistang' : 20.0,
    'xydistmax' : 2.0,
    'smax' : 20.0,
    'accmax' : 4.0,
    'predsfacs' : [0.5, 1.0],
    'predafacs' : [0.5],
    'interacdist' : 70.0,
    'planaccfacs' : [1.0],
    'plannspeeds' : 5,
    'col_plim' : 0.1,
    'score_wmin' : 0.3,
    'score_wfac' : 0.02
}

CONFIG_DICT = {
    'default' : DEF_CONFIG,
    'final_tuned_val_1' : TUNED_VAL_FINAL_1
}

class HardcodeNuscPlanner(PlannerNusc):
    def __init__(self, map_env, cfg):
        super(HardcodeNuscPlanner, self).__init__(map_env, cfg)
        self.lane_graphs = self.map_env.lane_graphs
        self.init_wstate = None
        self.batch_mask = None
        self.B = None
        self.batch_maps = None
        self.device = None
        self.ego_idx = 0 # index of the vehicle to replace when being rolled out
        assert isinstance(self.cfg, PlannerConfig)

    def aidx_str(self, aidx):
        return '%04d' % (aidx)

    def state_conv(self, graph_state, veh_att):
        '''
        Converts from the graph network state rep to the expected input of this planner.
        Assumes a SINGLE scene graph (not a batch) and that state is UNNORMALIZED.

        :param graph_state: tensor (NA x 6) (x,y,hx,hy,s,hdot) UNNORMALIZED
        :param veh_att: tensor (NA x 2) (l,w) UNNORMALIZED
        '''
        NA = graph_state.size(0)
        graph_state = graph_state.cpu().numpy()
        lw = veh_att.cpu().numpy()
        wstate = {'t': 0.0, 'objs': {}}
        for aidx in range(NA):
            objid = 'ego' if aidx == self.ego_idx else self.aidx_str(aidx)
            x, y, hcos, hsin, s, hdot = graph_state[aidx]
            h = np.arctan2(hsin, hcos)
            l, w = lw[aidx]
            wstate['objs'][objid] = {'x': x, 'y': y, 'h': h, 's': s, 'l': l, 'w': w}
        return wstate

    def create_init_state(self, init_state, vehicle_atts, B, batch_mask):
        ''' create initial state list accounting for batched scene graph '''
        init_state_list = []
        for b in range(B):
            cur_mask = (batch_mask == b)
            cur_init_state = self.state_conv(init_state[cur_mask], vehicle_atts[cur_mask])
            init_state_list.append(cur_init_state)
        return init_state_list

    def reset(self, init_state, vehicle_atts, batch_mask, batch_size, map_idx, ego_idx=0):
        '''
        Prepare for a new rollout starting from the given initial state.

        Assumes planner-controlled vehicle at index 0.

        :param init_state: tensor (NA x 6) (x,y,hx,hy,s,hdot) UNNORMALIZED
        :param vehicle_atts: tensor (NA x 2) (l,w) UNNORMALIZED
        :param batch_mask: tensor (NA,) which batch each agent belongs to.
        :param batch_size: int, how many scene graphs are in this batch
        :param map_idx: (B, ) which maps each scene graph in the batch will be rolled out on
        :param ego_idx: which index within each scene to replace with the planner
        '''
        self.ego_idx = ego_idx
        self.B = batch_size
        self.batch_mask = batch_mask
        self.init_wstate = self.create_init_state(init_state, vehicle_atts, self.B, self.batch_mask)
        self.batch_maps = [self.map_env.map_list[map_idx[b]] for b in range(self.B)]
        self.batch_mapidx = map_idx.cpu().numpy()

    def get_output_state(self, wstate):
        '''
        pull out the ego state that we'll want to output from the planner world state.
        '''
        x = wstate['objs']['ego']['control']['x']
        y = wstate['objs']['ego']['control']['y']
        h = wstate['objs']['ego']['control']['h']
        hcos = np.cos(h)
        hsin = np.sin(h)
        return np.array([x, y, hcos, hsin])

    def create_other_agents(self, init_agts, agt_obs, agt_t):
        '''
        :param init_agts: dict of init state for all agts
        :param agent_obs: (NA-1, T, 4) UNNORMALIZED future trajectories of all other agents in a SINGLE scene besides ego. 
        :param agent_t: np array shape (T) the timesteps corresponding to agent_obs (starting from 0.0)

        :return: data dict that includes interpolators for each agent to update wstate during rollout
        '''
        agt_dt = agt_t[1] - agt_t[0]
        v = {'objs' : {}}
        for aidx in range(agt_obs.shape[0]):
            strin = aidx+1 if aidx >= self.ego_idx else aidx
            astr = self.aidx_str(strin) # skipping ego
            v['objs'][astr] = {}
            state_t0 = np.array([[init_agts['objs'][astr]['x'],
                                 init_agts['objs'][astr]['y'],
                                 np.cos(init_agts['objs'][astr]['h']),
                                 np.sin(init_agts['objs'][astr]['h'])]])
            all_states = np.concatenate([state_t0, agt_obs[aidx]], axis=0)
            # only interp up to first observed nan value
            nan_states = np.isnan(np.sum(all_states, axis=1))
            first_nan_idx = np.nonzero(nan_states)[0]
            first_nan_idx = all_states.shape[0] if first_nan_idx.shape[0] == 0 else first_nan_idx[0]
            if first_nan_idx == 1:
                # only a single non-nan value, skip it
                cur_traj = [{'t' : 0.0}] # must include a traj though to be valid in rollout
                v['objs'][astr]['traj'] = cur_traj
                continue
            cur_traj = [{'t' : 0.0}] + [{'t' : t} for t in agt_t[:first_nan_idx-1]]
            v['objs'][astr]['traj'] = cur_traj
            all_t = np.append(np.array([0.0]), agt_t[:first_nan_idx-1])

            cur_interp = interp1d(all_t, all_states[:first_nan_idx], axis=0,
                                  copy=False, bounds_error=True, assume_sorted=True)
            v['objs'][astr]['interp'] = cur_interp

        return v

    def rollout(self, agent_obs, agent_t, agent_ptr, planner_t,
                init_state=None,
                control_all=False,
                viz=None,
                coll_t=None):
        '''
        Rollout agent given full sequence of past (and possibly future) observations
        of all other agents for num_steps.
        If init_state is given starts from this state rather than the current
        init_state set in self.reset().

        Returns the resulting kinematic trajectory.

        :param agent_obs: (NA-B, T, 4) UNNORMALIZED future trajectories of all other agents in scene besides ego. 
                            if controll_all=True, then this can be None
        :param agent_t: (T) time steps at which observed (starting at 0.0)
        :param agent_ptr: (B) points to start and end of each batch within the agent_obs array
        :param planner_t: np array shape (T) the timesteps that planner states need to be returned
        :param init_state: tensor (NA x 6) (x,y,hx,hy,s,hdot) UNNORMALIZED the initial state to start rollout from
                            instead of that passed to .reset()
        :param control_all: bool if True, the planner controls all agents in the scene, so agent_obs/agent_t is ignored.

        :return planner_fut: (B, T, 4)
        '''
        if (init_state is None and self.init_wstate is None) or self.B is None or self.batch_mask is None \
                or self.batch_maps is None:
            print('Must call reset before rolling out!')
            exit()
        batch_wstate = deepcopy(self.init_wstate) # updates in place
        if init_state is not None:
            batch_wstate = self.create_init_state(init_state, vehicle_atts, self.B, self.batch_mask)

        # print(batch_wstate[0])   

        # set params that we don't change from default
        lane_ds = 0.4
        lane_sig = 3.5
        sbuffer = 4.0

        # prepare obther agt observations in expected format
        batch_agt_obs = None
        if agent_obs is not None:
            batch_agt_obs = []
            assert(agent_obs.shape[1] == agent_t.shape[0])
            for b in range(self.B):
                cur_agts = agent_obs[agent_ptr[b]:agent_ptr[b+1]]
                assert(cur_agts.shape[0] == (len(batch_wstate[b]['objs'])-1))
                cur_agt_data = self.create_other_agents(batch_wstate[b], cur_agts, agent_t)
                batch_agt_obs.append(cur_agt_data)

        batch_wstate_out = []
        Tsteps = int(planner_t[-1] / self.cfg.dt)
        output_planner_t = np.linspace(self.cfg.dt, self.cfg.dt*Tsteps, Tsteps+1)
        for b in range(self.B):
            cur_wstate_out = []
            v = None
            if agent_obs is not None:
                v = batch_agt_obs[b]
            wstate = batch_wstate[b]

            cfg = self.cfg
            lane_graph = self.lane_graphs[self.batch_maps[b]]
            # predict future route for every object - associate with a lane graph
            compute_splines(wstate, lane_graph, cfg.cdistang, cfg.xydistmax,
                            lane_ds, lane_sig, sbuffer, cfg.smax, cfg.nsteps*cfg.preddt)
            compute_action(wstate, 'ego', cfg.dt, cfg.nsteps, cfg.preddt, cfg.predsfacs, cfg.predafacs,
                            cfg.interacdist, cfg.accmax, f'plan{b}_{0:04}', lane_graph, cfg.planaccfacs, cfg.smax,
                            cfg.plannspeeds, cfg.col_plim, debug=False, score_wmin=cfg.score_wmin, score_wfac=cfg.score_wfac)
            cur_wstate_out.append(self.get_output_state(wstate))
            if viz is not None:
                viz_wstate(wstate, lane_graph, 70.0, os.path.join(viz, '%04d.jpg' % (b*(Tsteps+1))))

            for istep in range(Tsteps):
                wstate = update_wstate(wstate, v, cfg.dt)

                cfg = self.cfg
                compute_splines(wstate, lane_graph, cfg.cdistang, cfg.xydistmax,
                            lane_ds, lane_sig, sbuffer, cfg.smax, cfg.nsteps*cfg.preddt)
                compute_action(wstate, 'ego', cfg.dt, cfg.nsteps, cfg.preddt, cfg.predsfacs, cfg.predafacs,
                            cfg.interacdist, cfg.accmax, f'plan{b}_{0:04}', lane_graph, cfg.planaccfacs, cfg.smax,
                            cfg.plannspeeds, cfg.col_plim, debug=False, score_wmin=cfg.score_wmin, score_wfac=cfg.score_wfac)
                cur_wstate_out.append(self.get_output_state(wstate))
                if viz is not None:
                    viz_wstate(wstate, lane_graph, 70.0, os.path.join(viz, '%04d.jpg' % (b*(Tsteps+1) + 1 + istep)))
            
            # interpolate planner trajectory to the expected output timestamps
            cur_wstate_out = np.stack(cur_wstate_out, axis=0)
            batch_wstate_out.append(cur_wstate_out)

        batch_wstate_out = np.stack(batch_wstate_out, axis=0)
        plan_interp = interp1d(output_planner_t, batch_wstate_out, axis=1, copy=False, bounds_error=True, assume_sorted=True)
        plan_out = plan_interp(planner_t)
        plan_out = torch.from_numpy(plan_out)

        if viz is not None:
            create_video(os.path.join(viz, '%04d.jpg'), viz + '.mp4', 2*int(1.0 / cfg.dt))
            shutil.rmtree(viz)

        return plan_out

#
# Utilities
# 
def compute_splines(wstate, lane_graph, cdisttheta, xydistmax, lane_ds, lane_sig, sbuffer,
                    smax, tmax):
    assert(smax > 0), f'{smax}'
    for objid,obj in wstate['objs'].items():
        matches = get_lane_matches(obj['x'], obj['y'], obj['h'], lane_graph,
                                   cdistmax=1.0 - np.cos(np.radians(cdisttheta)),
                                   xydistmax=xydistmax)
        # update wstate in place
        obj['final_matches'] = cluster_matches_combine(obj['x'], obj['y'], matches, lane_graph)
        backdist = 1.0 if obj['s'] > 0 else 1.0+abs(obj['s'])*tmax
        fordist = 1.0+smax*tmax if obj['s'] < 0 else max(1.0+smax*tmax, 1.0+obj['s']*tmax)
        obj['splines'] = get_prediction_splines(obj['final_matches'], lane_graph, backdist=backdist, fordist=fordist,
                                                xydistmax=xydistmax, egoxy=np.array([obj['x'],obj['y']]),
                                                lane_ds=lane_ds, lane_sig=lane_sig, sbuffer=sbuffer,
                                                egoh=obj['h'])
    return wstate

def get_lane_matches(x, y, h, lane_graph, cdistmax, xydistmax):
    # check heading
    cdist = 1.0 - lane_graph['edges'][:,2]*np.cos(h) - lane_graph['edges'][:,3]*np.sin(h)
    kept = cdist < cdistmax

    if kept.sum() > 0:
        la_xy = lane_graph['edges'][kept,0:2]
        la_h = lane_graph['edges'][kept,2:4]
        la_l = lane_graph['edges'][kept,4]

        closest, dist = edge_closest_point(la_xy, la_h, la_l, np.array([x, y]))

        options = dist < xydistmax

        all_matches = {'closest': closest[options],
                       'ixes': lane_graph['edgeixes'][kept][options],
                      }
    else:
        all_matches = {
            'closest': np.empty((0, 2)),
            'ixes': np.empty((0, 2), dtype=np.int64),
        }

    return all_matches


def cluster_matches_combine(x, y, matches, lane_graph):
    """Clusters the matches and chooses the closest match for each cluster.
    """
    if len(matches['closest']) == 0:
        return matches

    ixes = []
    closest = []
    seen = {(v0,v1): False for v0,v1 in matches['ixes']}
    ordering = np.argsort(np.linalg.norm(np.array([[x, y]]) - matches['closest'], axis=1))
    for (v0,v1),close in zip(matches['ixes'][ordering], matches['closest'][ordering]):
        if seen[v0,v1]:
            continue
        ixes.append([v0,v1])
        closest.append(close)

        # make everything that is connected to this edge seen
        seen = cluster_bfs(v0, v1, seen, lane_graph, go_forward=True)
        seen = cluster_bfs(v0, v1, seen, lane_graph, go_forward=False)

    return {
        'ixes': np.array(ixes),
        'closest': np.array(closest),
    }


def edge_closest_point(la_xy, la_h, la_l, query):
    diff = query.reshape((1, 2)) - la_xy
    lmag = diff[:,0]*la_h[:,0] + diff[:,1]*la_h[:,1]
    lmag[lmag < 0] = 0.0
    lmagkept = lmag > la_l
    lmag[lmagkept] = la_l[lmagkept]
    closest = la_xy + lmag[:, np.newaxis] * la_h
    dist = query.reshape((1, 2)) - closest
    dist = np.linalg.norm(dist, axis=1)
    return closest, dist


def cluster_bfs(v0, v1, seen, lane_graph, go_forward):
    qu = deque()
    qu.append((v0,v1))
    while len(qu) > 0:
        c0,c1 = qu.popleft()
        seen[c0,c1] = True
        if go_forward:
            for newix in lane_graph['out_edges'][c1]:
                if (c1,newix) in seen and not seen[c1,newix]:
                    qu.append((c1,newix))
        else:
            for newix in lane_graph['in_edges'][c0]:
                if (newix,c0) in seen and not seen[newix,c0]:
                    qu.append((newix,c0))
    return seen


def expand_verts(v0, xys, conns, mindist):
    """Does BFS from e0.
    lanes are represented by {'v': [v0, v1,....],
                              'l': 0.0,
                              }

    Note that this function can return lanes of length less than mindist meters
    if a lane reaches a terminal node in the graph.
    """
    assert(mindist >= 0), f'only non-negative distances allowed {mindist}'

    qu = deque()
    qu.append({'v': [v0],
               'l': 0.0})
    all_lanes = []
    while len(qu) > 0:
        lane = qu.popleft()
        while lane['l'] <= mindist:
            v = lane['v'][-1]
            if len(conns[v]) == 0:
                break

            # put branches in the queue if there are any
            for outv in conns[v][1:]:
                newlane = deepcopy(lane)
                newlane['l'] += np.linalg.norm(xys[outv] - xys[v])
                newlane['v'].append(outv)
                qu.append(newlane)
            outv = conns[v][0]

            lane['l'] += np.linalg.norm(xys[outv] - xys[v])
            lane['v'].append(outv)

        all_lanes.append(lane)

    return all_lanes


def extend_forward(xys, le):
    diff = xys[-1] - xys[-2]
    diff /= np.linalg.norm(diff)
    newxy = xys[-1] + diff * le
    xys = np.concatenate((xys, newxy[np.newaxis]), axis=0)
    return xys


def extend_backward(xys, le):
    diff = xys[0] - xys[1]
    diff /= np.linalg.norm(diff)
    newxy = xys[0] + diff * le
    xys = np.concatenate((newxy[np.newaxis], xys), axis=0)
    return xys


def local_lane_closest(xys, ix0, egoxy):
    assert(xys.shape[0] >= 2), xys.shape

    diff = xys[1:] - xys[:-1]
    dist = np.linalg.norm(diff, axis=1)
    ec,ed = edge_closest_point(xys[:-1], diff / dist[:, np.newaxis], dist, egoxy)

    cix = ix0
    # find the closest point local to egoxy (this is a little complicated but justified)
    while 0<=cix-1 and ed[cix-1]<ed[cix]:
        cix -= 1
    while cix+1<len(ed) and ed[cix+1]<ed[cix]:
        cix += 1

    if cix+1<len(ed):
        assert(ed[cix+1]>=ed[cix]), f'{ed[cix]} {ed[cix+1]}'
    if 0<=cix-1:
        assert(ed[cix-1]>=ed[cix]), f'{ed[cix-1]} {ed[cix]}'

    return cix, ec[cix]


def xy2spline(xy, ix0, blim, flim, egoh):
    diff = xy[1:] - xy[:-1]
    dist = np.linalg.norm(diff, axis=1)
    head = diff / dist[:, np.newaxis]
    head = np.concatenate((
        head, head[[-1]],
    ), 0)
    xyhh = np.concatenate((xy, head), 1)

    # force spline to pass through current heading
    xyhh[ix0,2] = np.cos(egoh)
    xyhh[ix0,3] = np.sin(egoh)

    t = np.zeros(len(xy))
    t[1:] = np.cumsum(dist)
    t -= t[ix0]
    assert(t[0] < blim), f'{t[0]} {blim}'
    assert(t[-1] > flim), f'{t[-1]} {flim}'
    return interp1d(t, xyhh, kind='linear', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)


def constant_heading_spline(egoxy, egoh, backdist, fordist):
    t = np.array([-backdist, fordist])
    x = np.array([
        [egoxy[0]-backdist*np.cos(egoh), egoxy[1]-backdist*np.sin(egoh), np.cos(egoh), np.sin(egoh)],
        [egoxy[0]+fordist*np.cos(egoh), egoxy[1]+fordist*np.sin(egoh), np.cos(egoh), np.sin(egoh)],
    ])
    return interp1d(t, x, kind='linear', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)


def get_prediction_splines(final_matches, lane_graph, backdist, fordist, xydistmax,
                           egoxy, lane_ds, lane_sig, sbuffer, egoh):
    """backdist: return splines that extend backwards at least this many meters
       fordist: return splines that extend forwards at least this many meters
       xydistmax: bound on how far the egoxy is from a lane
       egoxy: current (x,y) location of the car
       lane_ds: (meters) resolution when we warp the lane to pass through the ego
       lane_sig: (meters) how smoothly do splines return back to the lane graph
                 larger -> smoother
       sbuffer: (meters) needs to be large enough that when the splines are warped,
                 the spline is still "long enough".
       egoh: (radians) heading of the ego agent. Splines are guaranteed to pass exactly
               through the ego (x,y,h).

        Note that due to the buffers on the minimum distance for the splines, there is a 
        possibility that there are duplicates in the splines that are ultimately returned.

        Returns constant-heading spline if there are no lane matches.
    """

    if final_matches['ixes'].shape[0] == 0:
        return [constant_heading_spline(egoxy, egoh, backdist, fordist)]

    all_interps = []
    for (v0,v1),close in zip(final_matches['ixes'], final_matches['closest']):
        forward_lanes = expand_verts(v1, lane_graph['xy'], lane_graph['out_edges'],
                                     mindist=fordist+sbuffer+xydistmax)
        backward_lanes = expand_verts(v0, lane_graph['xy'], lane_graph['in_edges'],
                                      mindist=backdist+sbuffer+xydistmax)
        for flane in forward_lanes:
            for blane in backward_lanes:
                xys = np.concatenate((
                    lane_graph['xy'][blane['v'][::-1]],
                    lane_graph['xy'][flane['v']],
                ), axis=0)
                assert(xys.shape[0] >= 2), xys
                ix0 = len(blane['v']) - 1

                if flane['l'] <= fordist+sbuffer+xydistmax:
                    xys = extend_forward(xys, 1.0 + fordist + sbuffer + xydistmax - flane['l'])
                if blane['l'] <= backdist+sbuffer+xydistmax:
                    xys = extend_backward(xys, 1.0 + backdist + sbuffer + xydistmax - blane['l'])
                    ix0 += 1

                cix, cclose = local_lane_closest(xys, ix0, egoxy)

                tdist = np.zeros(len(xys))
                tdist[1:] = np.cumsum(np.linalg.norm(xys[1:] - xys[:-1], axis=1))
                tdist = tdist - tdist[cix] - np.linalg.norm(cclose - xys[cix])
                # assert(tdist[0] < -backdist-sbuffer), f'{tdist[0]} {-backdist-sbuffer}'
                # assert(tdist[-1] > fordist+sbuffer), f'{tdist[-1]} {fordist+sbuffer}'
                if tdist[0] >= -backdist-sbuffer:
                    print(f'WARNING: spline not far enough back {tdist[0]} {-backdist-sbuffer}')
                if tdist[-1] <= fordist+sbuffer:
                    print(f'WARNING: spline not far enough forward {tdist[-1]} {fordist+sbuffer}')
                interp = interp1d(tdist, xys, kind='linear', axis=0, copy=False,
                                  bounds_error=True, assume_sorted=True)
                numback = int((backdist+sbuffer)/lane_ds)+1
                numfor = int((fordist+sbuffer)/lane_ds)+1
                teval = np.concatenate((
                    np.linspace(-backdist-sbuffer, 0.0, numback+1)[:-1],
                    np.linspace(0.0, fordist+sbuffer, numfor),
                ), 0)
                xys = interp(teval)
                xys = xys + (egoxy - cclose)[np.newaxis, :] * np.exp(-np.square(teval) / lane_sig**2)[:, np.newaxis]

                spline = xy2spline(xys, numback, blim=-backdist, flim=fordist, egoh=egoh)

                all_interps.append(spline)
    return all_interps


def xyh2speed(x0, y0, x1, y1, h1, dt):
    sabs = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / dt
    ssign = 1 if (x1-x0)*np.cos(h1) + (y1-y0)*np.sin(h1) >= 0 else -1
    return ssign * sabs


def viz_wstate(wstate, lane_graph, window, imname):
    fig = plt.figure(figsize=(12, 12))
    gs = mpl.gridspec.GridSpec(1, 1, left=0.04, right=0.96, top=0.96, bottom=0.04)
    ax = plt.subplot(gs[0, 0])

    # plot lane graph
    plt.plot(lane_graph['edges'][:,0],lane_graph['edges'][:,1], '.', markersize=2)
    mag = 0.3
    plt.plot(lane_graph['edges'][:,0]+mag*lane_graph['edges'][:,2],
            lane_graph['edges'][:,1]+mag*lane_graph['edges'][:,3], '.', markersize=1)

    # plot objects
    for objid,obj in wstate['objs'].items():
        carcolor = 'g' if 'control' in obj else 'b'
        plot_car(obj['x'], obj['y'], obj['h'], obj['l'], obj['w'], color=carcolor)

    # # plot lane assignments
    for objid,obj in wstate['objs'].items():
        plt.plot(obj['final_matches']['closest'][:,0], obj['final_matches']['closest'][:,1],
                'b.', markersize=8, alpha=0.3)

    if 'ego' in wstate['objs']:
        centerx,centery = wstate['objs']['ego']['x'], wstate['objs']['ego']['y']
    else:
        centerx,centery = 200.0,200.0
    plt.xlim((centerx - window, centerx + window))
    plt.ylim((centery - window, centery + window))
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)


def update_wstate(wstate, v, dt):
    t0 = wstate['t']
    t1 = t0 + dt
    newwstate = {'t': t1, 'objs': {}}
    for objid,obj in wstate['objs'].items():
        if 'control' in obj:
            speed = xyh2speed(obj['x'], obj['y'], obj['control']['x'],
                              obj['control']['y'], obj['control']['h'], dt)
            newwstate['objs'][objid] = {
                'x': obj['control']['x'],
                'y': obj['control']['y'],
                'h': obj['control']['h'],
                's': speed, 'l': obj['l'], 'w': obj['w'],
            }
        elif v['objs'][objid]['traj'][0]['t'] <= t1 <= v['objs'][objid]['traj'][-1]['t']:
            x,y,hcos,hsin = v['objs'][objid]['interp'](t1)
            h = np.arctan2(hsin, hcos)
            speed = xyh2speed(obj['x'], obj['y'], x, y, h, dt)
            newwstate['objs'][objid] = {'x': x, 'y': y, 'h': h,
                                        's': speed, 'l': obj['l'], 'w': obj['w']}
    return newwstate


def compute_splines(wstate, lane_graph, cdisttheta, xydistmax, lane_ds, lane_sig, sbuffer,
                    smax, tmax):
    assert(smax > 0), f'{smax}'
    for objid,obj in wstate['objs'].items():
        matches = get_lane_matches(obj['x'], obj['y'], obj['h'], lane_graph,
                                   cdistmax=1.0 - np.cos(np.radians(cdisttheta)),
                                   xydistmax=xydistmax)
        # update wstate in place
        obj['final_matches'] = cluster_matches_combine(obj['x'], obj['y'], matches, lane_graph)
        backdist = 1.0 if obj['s'] > 0 else 1.0+abs(obj['s'])*tmax
        fordist = 1.0+smax*tmax if obj['s'] < 0 else max(1.0+smax*tmax, 1.0+obj['s']*tmax)
        obj['splines'] = get_prediction_splines(obj['final_matches'], lane_graph, backdist=backdist, fordist=fordist,
                                                xydistmax=xydistmax, egoxy=np.array([obj['x'],obj['y']]),
                                                lane_ds=lane_ds, lane_sig=lane_sig, sbuffer=sbuffer,
                                                egoh=obj['h'])
    return wstate


def postprocess_act_for_speed(x0, y0, h0, x1, y1, h1, s1, dt):
    """This is a kind of overly complicated function that just 
    makes sure that x1,y1,h1 are chosen such that the speed
    is exactly s1.
    """
    def ret_constant_heading():
        return x0+np.cos(h0)*s1*dt, y0+np.sin(h0)*s1*dt, h0
    sp = xyh2speed(x0, y0, x1, y1, h1, dt)
    # make sure the speed sign is correct
    if np.sign(sp) != np.sign(s1):
        # print('Warning: found a bad speed sign so falling back to constant velocity', sp, s1)
        newx,newy,newh = ret_constant_heading()
    else:
        diff = np.array([x1 - x0, y1 - y0])
        dist = np.linalg.norm(diff)
        if dist == 0.0:
            assert(s1 == 0.0), f"{diff} {s1}"
            newx,newy,newh = ret_constant_heading()
        else:
            diff /= dist
            # diff already has the sign so take abs(s) here (checked)
            newx,newy,newh = x0+diff[0]*abs(s1)*dt, y0+diff[1]*abs(s1)*dt, h1
    spp = xyh2speed(x0, y0, newx, newy, newh, dt)
    assert(np.abs(spp - s1)<1e-6), f'{spp} {s1} {sp}'
    return newx,newy,newh



def compute_speed_profile(s, stgt, acc, nsteps, preddt):
    """accelerate with magnitude no greater than acc from speed
    s to speed stgt. Returns profile of length nsteps+1 (eg the
    first index always has speed s).
    """
    if stgt > s:
        sprof = s + np.arange(nsteps+1) * acc * preddt
        sprof[sprof > stgt] = stgt
    elif stgt < s:
        sprof = s - np.arange(nsteps+1) * acc * preddt
        sprof[sprof < stgt] = stgt
    else:
        sprof = s + np.zeros(nsteps+1)
    return sprof


def sprof2dists(sprof, preddt):
    """Converts n+1 length sprof to distances
    """
    teval = np.zeros(len(sprof))
    teval[1:] = np.cumsum(sprof[1:]*preddt)
    return teval


def collect_other_trajs(wstate, egoid, nsteps, preddt, predsfacs,
                        predafacs, interacdist, maxacc):
    egoobj = wstate['objs'][egoid]
    egox,egoy,egoh = egoobj['x'],egoobj['y'],egoobj['h']
    other_trajs = []
    for otherid,other in wstate['objs'].items():
        if otherid == egoid or np.sqrt((egox-other['x'])**2+(egoy-other['y'])**2) > interacdist:
            continue
        sprofs = [compute_speed_profile(s=other['s'], stgt=other['s']*sfac,
                                        acc=maxacc*afac, nsteps=nsteps, preddt=preddt)
                                        for sfac in predsfacs for afac in predafacs]
        tevals = [sprof2dists(sprof, preddt) for sprof in sprofs]
        for interp in other['splines']:
            for teval in tevals:
                xyhh = interp(teval)
                traj = np.empty((nsteps+1, 5))
                traj[:, :2] = xyhh[:, :2]
                traj[:, 2] = np.arctan2(xyhh[:, 3], xyhh[:, 2])
                traj[:, 3] = other['l']
                traj[:, 4] = other['w']
                other_trajs.append(traj)

    if len(other_trajs) > 0:
        other_trajs = np.transpose(np.array(other_trajs), (1, 0, 2))
    else:
        other_trajs = np.empty((nsteps+1, 0, 5))

    return other_trajs


def score_dists(dists, score_wmin, score_wfac):
    w = score_wmin + np.arange(len(dists)) * score_wfac
    probs = 1.0 + np.tanh(-dists * w)
    probs[dists < 0] = 1.0
    return probs


def debug_planner(egotraj, otherobjs, lane_graph, debugname, prob, egoobj, sprofi):
    egocircles = boxes2circles(egotraj)
    othercircles = boxes2circles(otherobjs)

    for ti in range(otherobjs.shape[0]):
        fig = plt.figure(figsize=(10, 10))
        gs = mpl.gridspec.GridSpec(1, 1, left=0.02, right=0.98, bottom=0.02, top=0.98)
        ax = plt.subplot(gs[0, 0])

        # plot lane graph
        plt.plot(lane_graph['edges'][:,0],lane_graph['edges'][:,1], '.', markersize=2)
        mag = 0.3
        plt.plot(lane_graph['edges'][:,0]+mag*lane_graph['edges'][:,2],
                 lane_graph['edges'][:,1]+mag*lane_graph['edges'][:,3], '.', markersize=1)

        for x,y,h,l,w in otherobjs[ti]:
            plot_car(x, y, h, l, w, color='b', alpha=0.2)
        plot_car(egotraj[ti,0,0],egotraj[ti,0,1], egotraj[ti,0,2],
                 egotraj[ti,0,3], egotraj[ti,0,4], color='g', alpha=0.2)

        # plot circles
        for circs in othercircles[ti]:
            for x,y,r in circs:
                ax.add_patch(plt.Circle((x, y), r, color='k'))
        for circs in egocircles[ti]:
            for x,y,r in circs:
                ax.add_patch(plt.Circle((x,y), r, color='k'))
        plt.title(prob)

        ax.set_aspect('equal')
        plt.xlim((egoobj['x']-60.0, egoobj['x']+60.0))
        plt.ylim((egoobj['y']-60.0, egoobj['y']+60.0))
        imname = f'{debugname}_{sprofi:04}_{ti:03}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close(fig)

def plot_plan_info(otherobjs, sprofs, debugname, egoobj, egospline, lane_graph, nsteps, col_plim,
                   prefer_stop, debug, score_wmin, score_wfac):
    egotraj = np.empty((nsteps+1, 1, 5))
    egotraj[:, :, 3] = egoobj['l']
    egotraj[:, :, 4] = egoobj['w']

    if otherobjs.shape[1] == 0:
        return sprofs[np.argmax([sprof['teval'][-1] for sprof in sprofs])]

    all_probs = []
    for sprofi,sprof in enumerate(sprofs):
        egolocs = egospline(sprof['teval'])
        egotraj[:, 0, :2] = egolocs[:, :2]
        egotraj[:, 0, 2] = np.arctan2(egolocs[:, 3], egolocs[:, 2])
        # need to compute some metric for this sprofile based on otherobjs...
        otherdists = approx_bbox_distance(egotraj, otherobjs)[:,0]
        probs = score_dists(otherdists, score_wmin, score_wfac)
        prob = 1.0 - np.product(1.0 - probs)
        all_probs.append(prob)
        if debug:
            debug_planner(egotraj, otherobjs, lane_graph, debugname, prob, egoobj, sprofi)

    pos_ixes = [i for i in range(len(sprofs)) if all_probs[i] < col_plim]
    if len(pos_ixes) == 0:
        chosen_ix = np.argmin(all_probs)
    else:
        dists = [sprofs[i]['teval'][-1] for i in pos_ixes]
        if prefer_stop:
            distix = np.argmin(dists)
        else:
            distix = np.argmax(dists)
        chosen_ix = pos_ixes[distix]

    return sprofs[chosen_ix]


def gen_sprofiles(s0, preddt, nsteps, planaccfacs, maxacc, smax, NS, debugname):
    n1 = nsteps // 2
    n2 = nsteps - n1
    sprofs = []
    for fac in planaccfacs:
        acc = fac*maxacc
        stop = min(smax, s0 + n1*preddt*acc)
        sbot = max(0.0, s0 - n1*preddt*acc)
        for s1 in np.linspace(sbot, stop, NS):
            sprof1 = compute_speed_profile(s0, s1, acc, n1, preddt)
            stop = min(smax, sprof1[-1] + n2*preddt*acc)
            sbot = max(0.0, sprof1[-1] - n2*preddt*acc)
            for s2 in np.linspace(sbot, stop, NS):
                sprof2 = compute_speed_profile(sprof1[-1], s2, acc, n2, preddt)
                sprof = np.concatenate((sprof1, sprof2[1:]))
                teval = sprof2dists(sprof, preddt)
                sprofs.append({'sprof': sprof,
                               'teval': teval,
                               'acc': acc,
                               's1': s1,
                               's2': s2})

    return sprofs


def compute_action(wstate, objid, dt, nsteps, preddt, predsfacs, predafacs,
                   interacdist, maxacc, debugname, lane_graph, planaccfacs, smax,
                   plannspeeds, col_plim, debug, score_wmin, score_wfac):
    obj = wstate['objs'][objid]

    spline = obj['splines'][0]

    sprofs = gen_sprofiles(obj['s'], preddt, nsteps, planaccfacs, maxacc, smax,
                           plannspeeds, debugname)
    otherobjs = collect_other_trajs(wstate, objid, nsteps, preddt, predsfacs, predafacs,
                                    interacdist, maxacc)
    sprof = plot_plan_info(otherobjs, sprofs, debugname, obj, spline, lane_graph, nsteps, col_plim,
                           prefer_stop=True if len(obj['final_matches']['closest'])==0 else False,
                           debug=debug, score_wmin=score_wmin, score_wfac=score_wfac)
    stgt = compute_speed_profile(obj['s'], sprof['s1'], sprof['acc'], 1, dt)[1]

    # stgt = min(smax, obj['s'] + maxacc*preddt)
    newx,newy,newhcos,newhsin = spline(dt*stgt)
    newh = np.arctan2(newhsin, newhcos)
    # newx,newy,newh = postprocess_act_for_speed(obj['x'], obj['y'],
                                            #    newx, newy, newh, stgt, dt)
    newx,newy,newh = postprocess_act_for_speed(obj['x'], obj['y'], obj['h'],
                                               newx, newy, newh, stgt, dt)

    obj['control'] = {
        'x': newx,
        'y': newy,
        'h': newh,
    }


def boxes2circles(b):
    B,NA,_ = b.shape

    XY,Hi,Li,Wi = b[:,:,[0,1]],b[:,:,2],b[:,:,3],b[:,:,4]
    L = np.maximum(Li, Wi)
    W = np.minimum(Li, Wi)
    kept = Li < Wi
    H = np.copy(Hi)
    H[kept] = H[kept] + np.pi/2.0

    v0 = ((L-W)/2 + W/4)[:,:,np.newaxis] * np.stack((np.cos(H), np.sin(H)), 2)
    v1 = (W/4)[:,:,np.newaxis] * np.stack((-np.sin(H), np.cos(H)), 2)

    circles = np.empty((B, NA, 5, 3))
    circles[:, :, 0, [0,1]] = XY + v0 + v1
    circles[:, :, 1, [0,1]] = XY - v0 + v1
    circles[:, :, 2, [0,1]] = XY - v0 - v1
    circles[:, :, 3, [0,1]] = XY + v0 - v1
    circles[:, :, 4, [0,1]] = XY
    circles[:, :, 4, 2] = W/2
    circles[:, :, :4, 2] = W[:,:,np.newaxis] / 4

    return circles


def approx_bbox_distance(b0, b1):
    B,NA0,_ = b0.shape
    _,NA1,_ = b1.shape

    bc0 = boxes2circles(b0).reshape((B, NA0, 5, 1, 1, 3))
    bc1 = boxes2circles(b1).reshape((B, 1, 1, NA1, 5, 3))

    dist = np.linalg.norm(bc1[:,:,:,:,:,[0,1]] - bc0[:,:,:,:,:,[0,1]], axis=5)\
           - bc0[:,:,:,:,:,2] - bc1[:,:,:,:,:,2]
    dist = np.amin(dist, axis=(2, 3, 4))
    return dist

######################################


def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


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


def plot_box(box, lw, color='g', alpha=0.7, no_heading=False):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array([
        box[:2],
        box[:2] + l/2.*np.array([np.cos(h), np.sin(h)]),
    ])

    plt.fill(simple_box[:, 0], simple_box[:, 1], color=color, edgecolor='k',
             alpha=alpha, zorder=3, linewidth=1.0)
    if not no_heading:
        plt.plot(arrow[:, 0], arrow[:, 1], 'b', alpha=0.5)


def plot_car(x, y, h, l, w, color='b', alpha=0.5, no_heading=False):
    plot_box(np.array([x, y, np.cos(h), np.sin(h)]), [l, w],
             color=color, alpha=alpha, no_heading=no_heading)