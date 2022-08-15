# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import numpy as np

import torch
from torch import nn

from torch_geometric.nn import MessagePassing

from models.common import MLP

from utils.transforms import transform2frame

class SceneInteractionNet(nn.Module):
    def __init__(self, in_node_channels,
                       in_sem_channels,
                       in_edge_channels,
                       msg_node_channels, # number of channels to use for node embedding during message passing.
                       out_channels,
                       gru_update=False, # uses GRU cell as update function rather than MLP
                       gru_single_step=False, # if true, the GRU update uses the current feature as hidden state rather than a given persitent state 
                       k=1, # number of message passing rounds
                       nonlinearity=nn.ReLU
                ):
        super(SceneInteractionNet, self).__init__()

        # get initial node embeddings
        self.mlp_in = MLP([in_node_channels, 128, 128, msg_node_channels],
                           nonlinearity=nonlinearity)
        # do message passing
        self.gru_update = gru_update
        self.gru_single_step = gru_single_step
        interaction_layers = []
        for ki in range(k):
            graph_conv = AgentInteractionConv(msg_node_channels,
                                              in_sem_channels,
                                              in_edge_channels,
                                              msg_node_channels,
                                              hidden_size=128,
                                              gru_update=self.gru_update,
                                              gru_single_step=self.gru_single_step,
                                              nonlinearity=nonlinearity,
                                              aggr='max')
            interaction_layers.append(graph_conv)
        self.msg = nn.ModuleList(interaction_layers)
        # get output embeddings
        self.mlp_out = MLP([msg_node_channels, 128, 128, out_channels],
                            nonlinearity=nonlinearity)

    def forward(self, scene_graph, h=None, return_out=True):
        '''
        :param h: (NA x k x msg_node_channels) The current hidden state if using gru_update for each message passing layer.
                    or (NA x k x NS x msg_node_channels) if using multiple samples

        :return x: output of the interaction (NA x out_channels) or (NA x NS x out_channels) if input includes multiple samples
        :return h: [ONLY if using gru_update and h is passed in] (NA x k x msg_node_channels) output hidden state at each layer
                                                                    or (NA x k x NS x msg_node_channels)
        '''
        x = self.mlp_in(scene_graph.x)
        h_out = []
        for k, layer in enumerate(self.msg):
            cur_h = None if h is None else h[:, k]
            x = layer(x, scene_graph.edge_index, scene_graph.pos, scene_graph.sem, h=cur_h)
            if self.gru_update and h is not None:
                h_out.append(x)
        if return_out:
            x = self.mlp_out(x)
        if self.gru_update and h is not None:
            hout = torch.stack(h_out, dim=1)
            if return_out:
                return x, hout
            else:
                return hout
        else:
            return x

class AgentInteractionConv(MessagePassing):
    '''
    Graph convolution.
    '''
    def __init__(self, in_node_channels,
                       in_sem_channels,
                       in_edge_channels,
                       out_channels,
                       hidden_size=128, # size of hidden layers of message passing MLP
                       gru_update=False, # GRU is over multiple timesteps (required hidden state h input at each step)
                       gru_single_step=False, # if true, the GRU update uses the current feature as hidden state rather than a given persitent state 
                       nonlinearity=nn.ReLU,
                       aggr='max'): #  "Max" pooling by default.
        super(AgentInteractionConv, self).__init__(aggr=aggr,
                                                   flow='source_to_target')
        self.gru_update = gru_update
        self.gru_single_step = gru_single_step
        # source to target constructs messages to node i for each edge in (j,i)
        edge_mlp_input_size = 2*(in_node_channels + in_sem_channels) + in_edge_channels
        if self.gru_update and not self.gru_single_step:
            edge_mlp_input_size += 2*in_node_channels # also takes in current (external) hidden state
        self.edge_mlp = MLP([edge_mlp_input_size,
                            hidden_size,
                            hidden_size,
                            out_channels],
                            nonlinearity=nonlinearity)
        # node update function
        if self.gru_update:
            self.update_mlp = MLP([in_node_channels + out_channels + in_sem_channels,
                                    hidden_size,
                                    hidden_size,
                                    out_channels],
                                    nonlinearity=nonlinearity)
            self.update_func = nn.GRUCell(out_channels,
                                          in_node_channels)
        else:
            self.update_mlp = MLP([in_node_channels + out_channels + in_sem_channels,
                                    hidden_size,
                                    out_channels],
                                    nonlinearity=nonlinearity)
        self.out_channels = out_channels
        
    def forward(self, x, edge_index, pos, sem, h=None):
        '''
        :param x: (N x in_node_channels) or (N x NS x in_node_channels) INPUTS to each node
        :param edge_index: (2 x num_edges)
        :param pos: (N x in_edge_channels) or (N x NS x in_edge_channels) (x,y,hx,hy) used to compute relative transforms
        :param sem: (N x in_sem_channels) one-hot vector representing semantic class
        :param h: [OPTIONAL for GRU update] (N x in_node_channels) the current hidden state of each node if using and update GRU
        '''
        if len(x.size()) == 3:
            # have multiple samples
            NA, self.NS, D = x.size()
            x = x.reshape(NA, self.NS*D)
            pos = pos.reshape(NA, -1)
        else:
            self.NS = None
        return self.propagate(edge_index, x=x, pos=pos, sem=sem, h=h)

    def message(self, x_i, x_j, pos_i, pos_j, sem_i, sem_j, h_i, h_j):
        '''
        :param x_i: (num_edges, in_node_channels) or (num_edges, num_samples, in_node_channels)
        :param x_j: (num_edges, in_node_channels) or (num_edges, num_samples, in_node_channels) for neighbors
        :param pos_i: (num_edges, in_edge_channels) or (num_edges, num_samples, in_edge_channels)
        :param pos_j: (num_edges, in_edge_channels) or (num_edges, num_samples, in_edge_channels) for neighbors
        :param sem_i: (num_edges, in_sem_channels)
        :param sem_j: (num_edges, in_sem_channels) for neighbors
        :param h_i: [OPTIONAL for GRU update] (num_edges, in_node_channels)
        :param h_j: [OPTIONAL for GRU update] (num_edges, in_node_channels) for neighbors
        '''
        if x_i.size(0) == 0:
            if self.NS is not None:
                return torch.zeros((0, self.out_channels*self.NS)).to(x_i.device)
            else:
                return torch.zeros((0, self.out_channels)).to(x_i.device)
        if self.NS is not None:
            NE, NS = x_i.size(0), self.NS
            pos_i = pos_i.reshape(NE*NS, -1)
            pos_j = pos_j.reshape(NE*NS, -1)
        # need agent j in the frame of agent i
        # NOTE: if doing multiple message passing rounds with same pos, this is inefficient and instead should be precomputed for all edges
        rel_trans = transform2frame(pos_i, pos_j.unsqueeze(1))[:,0,:]
        # .pos may be nan if a node is unobserved, in this case set trans to 0
        rel_trans = torch.where(torch.isnan(rel_trans), torch.zeros_like(rel_trans), rel_trans)

        if self.NS is not None:
            # have multiple samples
            x_i = x_i.reshape(NE, self.NS, -1)
            x_j = x_j.reshape(NE, self.NS, -1)
            sem_i = sem_i.unsqueeze(1).expand(NE, NS, sem_i.size(1))
            sem_j = sem_j.unsqueeze(1).expand(NE, NS, sem_j.size(1))
            rel_trans = rel_trans.reshape(NE, NS, -1)
            if self.gru_update and h_i is not None and h_j is not None:
                h_i = h_i.unsqueeze(1).expand(NE, NS, h_i.size(1))
                h_j = h_j.unsqueeze(1).expand(NE, NS, h_j.size(1))

        msg_in = torch.cat([x_i, x_j, sem_i, sem_j, rel_trans], dim=-1)
        if self.gru_update and h_i is not None and h_j is not None:
            msg_in = torch.cat([msg_in, h_i, h_j], dim=-1)

        if self.NS is not None:
            edge_out = self.edge_mlp(msg_in)
            return edge_out.reshape(NE, NS*edge_out.size(-1))
        else:
            return self.edge_mlp(msg_in)

    def update(self, aggr_out, x, sem, h):
        '''
        :param aggr_out: (N x out_channels) output of the aggregation step. 
                NOTE: if there are no edges connected to a node, this is just all zeros.
        :param x: (N x in_node_channels) or (N x num_samples x in_node_channels)
        :param sem: (N x in_sem_channels)
        :param h: [OPTIONAL for GRU update] (N x in_node_channels) or (N x num_samples x in_node_channels) current hidden state 
        '''
        if self.NS is not None:
            NA, NS = x.size(0), self.NS
            x = x.reshape(NA, NS, -1)
            aggr_out = aggr_out.reshape(NA, NS, -1)
            sem = sem.unsqueeze(1).expand(-1, NS, -1)

        update_in = torch.cat([x, aggr_out, sem], dim=-1)
        if self.gru_update and h is not None:
            if self.NS is not None:
                update_in = update_in.reshape(NA*NS, -1)
                h = h.reshape(NA*NS, -1)
                update_res = self.update_func(update_in, h)
                return update_res.reshape(NA, NS, -1)
            else:
                return self.update_func(update_in, h)
        elif self.gru_update and self.gru_single_step:
            if self.NS is not None:
                update_in = update_in.reshape(NA*NS, -1)
                update_prepr = self.update_mlp(update_in)
                update_res = self.update_func(update_prepr, x.reshape(NA*NS, -1))
                return update_res.reshape(NA, NS, -1)
            else:
                prepr_out = self.update_mlp(update_in)
                return self.update_func(prepr_out, x)
        else:
            return self.update_mlp(update_in)
