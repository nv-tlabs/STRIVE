# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import itertools

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal

from models.interaction_net import SceneInteractionNet
from models.common import MLP, car_dynamics

from datasets.utils import normalize_scene_graph
from utils.transforms import transform2frame, kinematics2angle, kinematics2vec
from utils.torch import calc_conv_out
from utils.logger import throw_err, Logger

TRAJ_ENCODER_CHOICES = ['mlp', 'gru']

class TrafficModel(nn.Module):
    def __init__(self, npast, nfuture, map_obs_size_pix, nclasses,
                 map_feat_size=64,
                 past_feat_size=64,
                 future_feat_size=64,
                 latent_size=32,
                 output_bicycle=True,
                 traj_encoder='mlp',
                 conv_channel_in=4,
                 conv_kernel_list=[7, 5, 5, 3, 3, 3],
                 conv_stride_list=[2, 2, 2, 2, 2, 2],
                 conv_filter_list=[16, 32, 64, 64, 128, 128]
                 ):
        '''
        :param npast: number of past steps to take as input
        :param nfuture: number of future steps to predict as output
        :param map_obs_size_pix: width (in pixels) of map crop
        :param nclasses: number of different semantic classes
        '''
        super(TrafficModel, self).__init__()
        self.normalizer = self.att_normalizer = None # normalizer for state and vehicle attributes
        self.PT = npast
        self.FT = nfuture
        self.dt = 0.5 # for nusc dataset
        self.NC = nclasses
        self.output_bicycle = output_bicycle
        if self.output_bicycle:
            self.bicycle_params = None
            Logger.log('Using bicycle model as output parameterization of model...')

        self.state_size = 6 #(x,y,hx,hy,s,hdot)
        self.att_feat_size = 2 #(l,w)

        self.traj_encoder_type = traj_encoder
        if self.traj_encoder_type not in TRAJ_ENCODER_CHOICES:
            throw_err('Trajectory encoder type %s not recognized!' % (self.traj_encoder_type))
        else:
            Logger.log('Using %s past/future encoder...' % (self.traj_encoder_type))

        #
        # Map encoding
        #
        self.mapH = map_obs_size_pix
        self.mapW = map_obs_size_pix
        self.map_obs_size_pix = map_obs_size_pix

        conv_layer_list = []
        final_conv_out = map_obs_size_pix
        assert len(conv_kernel_list) == len(conv_stride_list)
        assert len(conv_kernel_list) == len(conv_filter_list)
        conv_filter_list = [conv_channel_in] + conv_filter_list
        for lidx in range(len(conv_kernel_list)):
            cur_conv = nn.Conv2d(conv_filter_list[lidx],
                                 conv_filter_list[lidx+1],
                                 kernel_size=conv_kernel_list[lidx],
                                 stride=conv_stride_list[lidx],
                                 padding=0)
            cur_gn = nn.GroupNorm(1, conv_filter_list[lidx+1])
            conv_layer_list.extend([cur_conv, cur_gn, nn.ReLU()])
            final_conv_out = calc_conv_out(final_conv_out, conv_kernel_list[lidx], conv_stride_list[lidx])

        self.map_conv = nn.Sequential(*conv_layer_list)
        self.map_feat_in_size = conv_filter_list[-1] * final_conv_out * final_conv_out
        self.map_feat_out_size = map_feat_size
        self.map_feature = nn.Linear(self.map_feat_in_size, self.map_feat_out_size)

        #
        # Motion encoding
        #

        # past encoder
        self.past_feat_size = past_feat_size
        if self.traj_encoder_type == 'mlp':
            self.past_in_size = self.NC + self.PT*(self.state_size + self.att_feat_size + 1) # +1 from visibility flag
            self.past_encoder = MLP([self.past_in_size, 128, 128, 128, self.past_feat_size])
        elif self.traj_encoder_type == 'gru':
            self.past_in_size = self.NC + self.state_size + self.att_feat_size + 1 # +1 from visibility flag
            self.past_encoder = nn.GRU(self.past_in_size,
                                        128, # hidden size
                                        4, # num stacked GRU layers
                                        batch_first=True
                                        )
            self.past_out_layer = nn.Linear(128, self.past_feat_size)

        # future encoder
        self.future_feat_size = future_feat_size
        if self.traj_encoder_type == 'mlp':
            self.future_in_size = self.NC + self.FT*(self.state_size + self.att_feat_size + 1) # +1 from visibility flag
            self.future_encoder = MLP([self.future_in_size, 128, 128, 128, self.future_feat_size])
        elif self.traj_encoder_type == 'gru':
            self.future_in_size = self.NC + self.state_size + self.att_feat_size + 1 # +1 from visibility flag
            self.future_encoder = nn.GRU(self.future_in_size,
                                            128, # hidden size
                                            4, # num stacked GRU layers
                                            batch_first=True
                                            )
            self.future_out_layer = nn.Linear(128, self.future_feat_size)

        #
        # Interaction networks
        #
        self.z_size = latent_size
        self.prior_net = SceneInteractionNet(self.past_feat_size + self.map_feat_out_size + self.NC, # agent input feat size
                                         self.NC, # size of semantic feature (one-hot class)
                                         4, # edge feat size (x,y,hx,hy)
                                         2*self.past_feat_size, # interaction node size.
                                         self.z_size*2, # out feat size (need mean and variance)
                                         ) 
        self.posterior_net = SceneInteractionNet(self.future_feat_size + self.past_feat_size + self.map_feat_out_size + self.NC,
                                                 self.NC, # size of semantic feature (one-hot class)
                                                 4,
                                                 2*self.past_feat_size,
                                                 self.z_size*2,
                                                 )

        if self.output_bicycle:
            self.traj_out_size = 2 # (a,hdot) for a single step
        else:
            self.traj_out_size = 4 # (x,y,hx,hy) for a single step
        decode_in_size = self.z_size + self.past_feat_size + self.map_feat_out_size + self.NC
        decode_in_size = decode_in_size + self.att_feat_size
        self.decoder_net = SceneInteractionNet(decode_in_size,
                                                self.NC, # size of semantic feature (one-hot class)
                                                4, # edge feat size (x,y,hx,hy)
                                                64, # node embedding size during message passing
                                                self.traj_out_size
                                                )
        # GRU for state memory
        self.num_memory_layers = 3
        self.decoder_memory = nn.GRU(4, # input size (x,y,hx,hy)
                                    self.past_feat_size, # hidden size
                                    self.num_memory_layers, # num stacked GRU layers
                                    batch_first=True, # batch_first (B, T, D) inputs outputs
                                    )



    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def get_normalizer(self):
        return self.normalizer

    def set_att_normalizer(self, normalizer):
        self.att_normalizer = normalizer

    def get_att_normalizer(self):
        return self.att_normalizer

    def set_bicycle_params(self, bicycle_params):
        '''
        Set bicycle model params including: max speed, max hdot, dt, etc.. used during rollout
        '''
        self.bicycle_params = bicycle_params

    def forward(self, scene_graph, map_idx, map_env,
                use_post_mean=False,
                future_sample=False):
        '''
        Forward pass to be used during training (samples from posterior).
        :param scene_graph: must contain past (NA x PT x 6), past_vis (NA x PT), future (NA x FT x 6), future_vis (NA x FT),
                                 edge_index (2 x num_edges), lw (NA x 2), sem (NA x num_classes), batch (NA)
        :param map_idx: size (B,) and indexes into the maps contained in map_env
        :param map_env: map environment to query for map crop
        :param use_post_mean: if True, uses the mean of the posterior for z, rather than sampling
        :param future_sample: if True, samples the prior and decodes one possible future and returns
        '''
        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat

        # extract past motion feature for each agent
        past_feat = self.encode_past(scene_graph)

        # extract future motion for each agent
        future_feat = self.encode_future(scene_graph)

        # PRIOR
        prior_mu, prior_var = self.prior(scene_graph, map_feat, past_feat)

        # POSTERIOR
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_feat, future_feat)

        # DECODER
        if use_post_mean:
            z_samp = post_mu
        else:
            # sample from posterior in training
            z_samp = self.rsample(post_mu, post_var)
        future_pred = self.decoder(scene_graph, map_feat, past_feat, z_samp, map_idx, map_env) # (NA, FT, 4)

        net_out = {
            'prior_out' : (prior_mu, prior_var),
            'posterior_out' : (post_mu, post_var),
            'future_pred' : future_pred
        }

        if future_sample:
            prior_samp = self.rsample(prior_mu, prior_var)
            future_samp = self.decoder(scene_graph, map_feat, past_feat, prior_samp, map_idx, map_env) # (NA, FT, 4)
            net_out['future_samp'] = future_samp
        
        return net_out

    def reconstruct(self, scene_graph, map_idx, map_env):
        '''
        Given both past and future, reconstruct using the encoder (posterior) by
         taking the mean of the predicted distribution.

        :param scene_graph: must contain past (NA x PT x 6), past_vis (NA x PT), future (NA x FT x 6), future_vis (NA x FT),
                                 edge_index (2 x num_edges), lw (NA x 2), sem (NA x num_classes), batch (NA)
        :param map_idx: is size (B,) and indexes into the maps contained in map_env
        :param map_env: map environment to query for map crop

        :returns: dict with future_pred and posterior_out
        '''
        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat
        # extract past motion feature for each agent
        past_feat = self.encode_past(scene_graph)
        # extract future motion for each agent
        future_feat = self.encode_future(scene_graph)
        # POSTERIOR
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_feat, future_feat)
        # DECODER
        # use posterior mean
        future_pred = self.decoder(scene_graph, map_feat, past_feat, post_mu, map_idx, map_env) # (NA, FT, 4)

        net_out = {
            'posterior_out' : (post_mu, post_var),
            'future_pred' : future_pred
        }
        
        return net_out

    def sample(self, scene_graph, map_idx, map_env, num_samples,
                include_mean=False,
                nfuture=None):
        '''
        Given past trajectories, sample possible futures using the prior.
        This is the serial version that loops through and samples decodes iteratively.
        This is slow, but works for larger batch sizes b/c doesn't use any extra memory.

        :param scene_graph: must contain past (NA x PT x 6), past_vis (NA x PT),
                                 edge_index (2 x num_edges), lw (NA x 2), sem (NA x num_classes), batch (NA)
        :param map_idx: is size (B,) and indexes into the maps contained in map_env
        :param map_env: map environment to query for map crop
        :param num_samples: the number of futures to sample 

        :param include_mean: if true, the Nth sample uses the mean z of the prior distribution
        :param nfuture: number of steps into the future to roll out (if different than model self.nfuture)
        '''
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)
        NS = num_samples
        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat

        # extract past motion feature for each agent
        past_feat = self.encode_past(scene_graph)

        # PRIOR
        prior_mu, prior_var = self.prior(scene_graph, map_feat, past_feat)
        prior_distrib = Normal(prior_mu, torch.sqrt(prior_var))

        net_out = {
            'prior_out' : (prior_mu, prior_var),
            'z_samp' : [],
            'z_logprob' : [],
            'z_mdist' : [], # mahalanobis distance
            'future_pred' : []
        }
        for sidx in range(num_samples):
            # print(sidx)
            if include_mean and sidx == (num_samples-1):
                z_samp = prior_mu
            else:
                z_samp = self.rsample(prior_mu, prior_var)
            z_logprob = prior_distrib.log_prob(z_samp).sum(dim=-1)
            z_mdist = torch.norm((z_samp - prior_mu) / torch.sqrt(prior_var), dim=-1)
            future_pred = self.decoder(scene_graph, map_feat, past_feat, z_samp, map_idx, map_env,
                                        nfuture=nfuture) # (NA, FT, 4)
            net_out['z_samp'].append(z_samp)
            net_out['z_logprob'].append(z_logprob)
            net_out['z_mdist'].append(z_mdist)
            net_out['future_pred'].append(future_pred)
        
        net_out['z_samp'] = torch.stack(net_out['z_samp'], dim=1)
        net_out['z_logprob'] = torch.stack(net_out['z_logprob'], dim=1)
        net_out['z_mdist'] = torch.stack(net_out['z_mdist'], dim=1)
        net_out['future_pred'] = torch.stack(net_out['future_pred'], dim=1)
                
        return net_out

    def sample_batched(self, scene_graph, map_idx, map_env, num_samples,
                        include_mean=False,
                        nfuture=None):
        '''
        Given past trajectories, sample possible futures using the prior. 
        This is a batched version of sampling that uses a lot of memory, but is faster than serial.

        :param scene_graph: must contain past (NA x PT x 6), past_vis (NA x PT),
                                 edge_index (2 x num_edges), lw (NA x 2), sem (NA x num_classes), batch (NA)
        :param map_idx: is size (B,) and indexes into the maps contained in map_env
        :param map_env: map environment to query for map crop
        :param num_samples: the number of futures to sample 

        :param include_mean: if true, the Nth sample uses the mean z of the prior distribution
        :param nfuture: number of steps into the future to roll out (if different than model self.nfuture)
        '''
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)
        NS = num_samples
        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat

        # extract past motion feature for each agent
        past_feat = self.encode_past(scene_graph)

        # PRIOR
        prior_mu, prior_var = self.prior(scene_graph, map_feat, past_feat)

        # sample from prior
        samp_mu = prior_mu.view(1, NA, self.z_size).expand(NS, NA, self.z_size)
        samp_var = prior_var.view(1, NA, self.z_size).expand(NS, NA, self.z_size)
        prior_distrib = Normal(samp_mu, torch.sqrt(samp_var))

        z_samp = self.rsample(samp_mu, samp_var)
        if include_mean:
            z_samp[-1, :, :] = prior_mu
        future_pred = self.decoder(scene_graph, map_feat, past_feat, z_samp.transpose(0, 1), map_idx, map_env,
                                    nfuture=nfuture) # (NA, NS, FT, 4)
        net_out = {
            'prior_out' : (prior_mu, prior_var),
            'z_samp' : z_samp.view(NS, NA, self.z_size).transpose(0, 1),
            'future_pred' : future_pred
        }

        z_logprob = prior_distrib.log_prob(z_samp.view(NS, NA, self.z_size)).sum(dim=-1).transpose(0, 1)
        z_mdist = torch.norm((z_samp.view(NS, NA, self.z_size) - samp_mu) / torch.sqrt(samp_var), dim=-1).transpose(0, 1)

        net_out['z_logprob'] = z_logprob
        net_out['z_mdist'] = z_mdist
                
        return net_out

    def embed(self, scene_graph, map_idx, map_env):
        '''
        Given past and optionally future, embed the given trajectories
        using the prior and (optionally) posterior.

        returns dict with prior_out and (optionally) posterior_out as well as
        other required values to decode (map feat etc..)
        '''
        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat

        # extract past motion feature for each agent
        past_feat = self.encode_past(scene_graph)

        # PRIOR
        prior_mu, prior_var = self.prior(scene_graph, map_feat, past_feat)

        embed_out = {
            'prior_out' : (prior_mu, prior_var),
            'map_feat' : map_feat,
            'past_feat' : past_feat
        }

        if 'future' in scene_graph:
            # extract future motion for each agent
            future_feat = self.encode_future(scene_graph)
             # POSTERIOR
            post_mu, post_var = self.encoder(scene_graph, map_feat, past_feat, future_feat)
            embed_out['posterior_out'] = (post_mu, post_var)

        return embed_out

    def decode_embedding(self, z, embed_out, scene_graph, map_idx, map_env,
                            ext_future=None,
                            nfuture=None):
        '''
        Given inputs/outputs of embed function, decodes to predicted trajectory in world space.
        '''
        future_pred = self.decoder(scene_graph, embed_out['map_feat'], embed_out['past_feat'],
                                        z, map_idx, map_env, ext_future=ext_future,
                                        nfuture=nfuture) # (NA, FT, 4)
        return {'future_pred' : future_pred}

    def encode_map(self, scene_graph, map_idx, map_env):
        '''
        Encodes local map patch around each agent based on the .pos attribute.
        NOTE: assumes the scene graph is NORMALIZED, so will unnormalize before doing the crop.

        :param scene_graph: makes use of .pos attribute (NA x 4) or (NA x NS x 4) with (x,y,hx,hy)
        :param map_idx: index of the map to use (B, )
        :param map_env: map environment to get crop with

        :return: NA x feat_dim
        '''
        NA = scene_graph.pos.size(0)
        NS = None if len(scene_graph.pos.size()) != 3 else scene_graph.pos.size(1)
        # first must unnormalize to get true world space state
        normalize_scene_graph(scene_graph,
                                self.normalizer,
                                self.att_normalizer,
                                unnorm=True)
        # get local crops based on .pos
        map_obs = map_env.get_map_crop(scene_graph, map_idx).to(torch.float) # NA x C x mapH x mapW
        # encode
        map_feat = self.map_conv(map_obs)
        # print(map_feat.size())
        bsize = NA if NS is None else NA*NS
        map_feat = self.map_feature(map_feat.view(bsize, self.map_feat_in_size))
        # print(map_feat.size())

        if NS is not None:
            map_feat = map_feat.reshape(NA, NS, -1)

        # re-normalize scene graph
        normalize_scene_graph(scene_graph,
                                self.normalizer,
                                self.att_normalizer,
                                unnorm=False)
        return map_feat

    def encode_past(self, scene_graph):
        '''
        Extract per-agent feature based on past trajectories (in local reference frame of last past step).
        Scene graph is assumed to be NORMALIZED.

        :param scene_graph: must have .past (NA x PT x 6), .past_vis (NA x PT),
                            .sem (NA x NC) and .lw (NA x 2) to encode

        :return: NA x feat_dim
        '''
        NA, PT, _ = scene_graph.past.size()
        # transform to local frame of last step of past
        local_past_kin = transform2frame(scene_graph.past[:, -1, :4], scene_graph.past[:, :, :4])
        local_past_traj = torch.cat([local_past_kin, scene_graph.past[:, :, 4:]], dim=2)
        # zero out any frames that were not observed
        local_past_traj[scene_graph.past_vis == 0.0] = 0.0
        # and append the visibility to the state as input
        local_past_traj = torch.cat([local_past_traj, scene_graph.past_vis.unsqueeze(-1)], dim=-1)
        # also add vehicle attributes
        veh_in_att = scene_graph.lw.unsqueeze(1).expand(NA, PT, self.att_feat_size) 
        encoder_in = torch.cat([local_past_traj, veh_in_att], dim=-1)
        if self.traj_encoder_type == 'mlp':
            # append semantic class too
            encoder_in = torch.cat([encoder_in.view(NA, -1), scene_graph.sem], dim=1)
        elif self.traj_encoder_type == 'gru':
            encoder_in = torch.cat([encoder_in, scene_graph.sem.view(NA, 1, self.NC).expand(NA, PT, self.NC)], dim=2)
        # then encode
        past_feat = self.past_encoder(encoder_in)

        if self.traj_encoder_type == 'gru':
            past_feat = past_feat[0][:, -1, :] # only want output of last step
            past_feat = self.past_out_layer(past_feat)

        return past_feat

    def encode_future(self, scene_graph):
        '''
        Extract per-agent feature based on future trajectories (in local reference frame of last past setp).
        Scene graph is assumed to be NORMALIZED.

        :param scene_graph: must have .past (NA x PT x 6), .future (NA x FT x 6), .sem (NA x NC)
                            .future_vis (NA x FT), and .lw (NA x 2) to encode.

        :return: NA x feat_dim
        '''
        NA, FT, _ = scene_graph.future.size()
        PT = scene_graph.past.size(1)
        # transform to local frame of last step of past
        local_future_kin = transform2frame(scene_graph.past[:, -1, :4], scene_graph.future[:, :, :4])
        local_future_traj = torch.cat([local_future_kin, scene_graph.future[:, :, 4:]], dim=2)
        # zero out any frames that were not observed
        local_future_traj[scene_graph.future_vis == 0.0] = 0.0
        # and append the visibility to the state as input
        local_future_traj = torch.cat([local_future_traj, scene_graph.future_vis.unsqueeze(-1)], dim=-1)
        # also add vehicle attributes
        veh_in_att = scene_graph.lw.unsqueeze(1).expand(NA, FT, self.att_feat_size)
        encoder_in = torch.cat([local_future_traj, veh_in_att], dim=-1)
        if self.traj_encoder_type == 'mlp':
            # append semantic class too
            encoder_in = torch.cat([encoder_in.view(NA, -1), scene_graph.sem], dim=1)
        elif self.traj_encoder_type == 'gru':
            encoder_in = torch.cat([encoder_in, scene_graph.sem.view(NA, 1, self.NC).expand(NA, FT, self.NC)], dim=2)
        
        # then encode
        future_feat = self.future_encoder(encoder_in)

        if self.traj_encoder_type == 'gru':
            future_feat = future_feat[0][:, -1, :] # only want output of last step
            future_feat = self.future_out_layer(future_feat)

        return future_feat

    def encoder(self, scene_graph, map_feat, past_feat, future_feat):
        '''
        :param scene_graph: pytorch geom Batch of graphs (NOTE will be used in place...)
        :param map_feat: (NA x map_feat_size)
        :param past_feat: (NA x past_feat_size)
        :param future_feat: (NA x future_feat_size)
        '''
        NA, _ = past_feat.size()
        posterior_in_feat = torch.cat([past_feat, future_feat, map_feat, scene_graph.sem], dim=-1)
        # set node features in scene graph
        scene_graph.x = posterior_in_feat
        # and set current position to be last step of past (to determine relative transforms)
        scene_graph.pos = scene_graph.past[:, -1, :4]

        posterior_z = self.posterior_net(scene_graph)
        # split into mu and sigma and make NA x z_size
        mean, logvar = posterior_z[:, :self.z_size], posterior_z[:, self.z_size:]
        var = torch.exp(logvar)
        return mean, var

    def prior(self, scene_graph, map_feat, past_feat):
        '''
        :param scene_graph: pytorch geom Batch of graphs (NOTE will be used in place...)

        :param map_feat: (NA x map_feat_size)
        :param past_feat: (NA x past_feat_size)

        :return: mean, var both (NA, z_size)
        '''
        NA, _ = past_feat.size()
        prior_in_feat = torch.cat([past_feat, map_feat, scene_graph.sem], dim=-1)
        # set node features in scene graph
        scene_graph.x = prior_in_feat
        # and set current position to be last step of past (to determine relative transforms)
        scene_graph.pos = scene_graph.past[:, -1, :4]

        prior_z = self.prior_net(scene_graph)
        # split into mu and sigma and make NA x z_size
        mean, logvar = prior_z[:, :self.z_size], prior_z[:, self.z_size:]
        var = torch.exp(logvar)
        return mean, var

    def decoder(self, scene_graph, map_feat, past_feat, z, map_idx, map_env,
                ext_future=None,
                nfuture=None):
        '''
        :param scene_graph: pytorch geom Batch of graphs (NOTE will be used in place...)
        :param map_feat: (NA x map_feat_size) of past[-1] viewpoint
        :param past_feat: (NA x past_feat_size)
        :param z: (NA x z_size) or (NA x NS x z_size) if wanting to decode multiple samples
        :param map_idx: (B,) indices of map used for each sub graph
        :param map_env: MapEnv object to query map crops from
        :param ext_future: if not None, size (B x FT x 4), indicates that the first node in each graph
                            is an external agent with the given true observed future. This means rollout
                            will use this external future rather than its own predictions at each step.
                            Assumed all entires are non-null. NORMALIZED
        :param nfuture: if not None, overrides self.FT and rolls out future for nfuture steps

        :return: future predicted trajectories of all agents (NA x FT x 4) in GLOBAL frame
        '''
        return self.autoregressive_decoder(scene_graph, map_feat, past_feat, z, map_idx, map_env,
                                            ext_future=ext_future,
                                            nfuture=nfuture)

    def autoregressive_decoder(self, scene_graph, map_feat, past_feat, z, map_idx, map_env,
                                ext_future=None,
                                nfuture=None):
        NA = map_feat.size(0)
        FT = self.FT if nfuture is None else nfuture
        # initialze with last step of input sequence (in global frame)
        prev_state = scene_graph.past[:, -1, :] if self.output_bicycle else scene_graph.past[:, -1, :4]
        traj_out = []
        cur_map_feat = map_feat
        cur_past_feat = past_feat
        cur_sem = scene_graph.sem
        cur_lw = scene_graph.lw
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:,0].unsqueeze(1)
        ego_inds = scene_graph.ptr[:-1]
        # init positions to last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        zsize = z.size()
        NS = None
        if len(z.size()) == 3:
            # have multiple samples
            NS = zsize[1]
            scene_graph.pos = scene_graph.pos.unsqueeze(1).expand(NA, NS, scene_graph.pos.size(1))
            cur_past_feat = cur_past_feat.unsqueeze(1).expand(NA, NS, past_feat.size(1))
            cur_map_feat = cur_map_feat.unsqueeze(1).expand(NA, NS, map_feat.size(1))
            cur_sem = cur_sem.unsqueeze(1).expand(NA, NS, scene_graph.sem.size(1))
            cur_lw = cur_lw.unsqueeze(1).expand(NA, NS, scene_graph.lw.size(1))
            cur_veh_len = cur_veh_len.unsqueeze(1).expand(NA, NS, 1).reshape(NA*NS, 1)
            prev_state = prev_state.unsqueeze(1).expand(NA, NS, prev_state.size(1)).reshape(NA*NS, -1)
            if ext_future is not None:
                ext_future = ext_future.unsqueeze(1).expand(NA, NS, -1, 4).reshape(NA*NS, -1, 4)
                ego_inds = ego_inds.unsqueeze(1).expand(NA, NS).reshape(NA*NS)
        mult_samp = NS is not None
        # init RNN hidden state from past encoder
        if mult_samp:
            cur_mem_state = cur_past_feat.reshape(NA*NS, past_feat.size(1)).unsqueeze(0).expand(self.num_memory_layers, NA*NS, self.past_feat_size).contiguous()
        else:
            cur_mem_state = past_feat.unsqueeze(0).expand(self.num_memory_layers, NA, self.past_feat_size).contiguous()
        for t in range(FT):
            # update scene graph node features with z + cur_past_feat + cur_map_feat + sem_class
            decoder_in_feat = torch.cat([cur_past_feat, cur_map_feat, cur_sem, z], dim=-1)
            decoder_in_feat = torch.cat([decoder_in_feat, cur_lw], dim=-1)
            scene_graph.x = decoder_in_feat
            # run scene graph decoding
            decoder_out = self.decoder_net(scene_graph) # (NA, traj_dim)

            if mult_samp:
                decoder_out = decoder_out.reshape(NA*NS, -1)

            cur_state_local = None # in the frame of t-1 (NA, 4)
            cur_state_global = None # in the global frame (NA, 4)
            cur_bike_state = None
            if self.output_bicycle:
                # acceleration and heading
                bsize = NA if not mult_samp else NA*NS
                dynamics_out = decoder_out.view(bsize, 1, 1, 2) # input to sim must be (B x _ x T x 2)
                # unnormalize
                a_out = dynamics_out[:,:,:,0]*self.bicycle_params['a_stats'][1] + self.bicycle_params['a_stats'][0]
                ddh_out = dynamics_out[:,:,:,1]*self.bicycle_params['ddh_stats'][1] + self.bicycle_params['ddh_stats'][0]
                # simulate forward
                init_state = self.normalizer.unnormalize(prev_state)
                cur_bike_state = self.sim_traj(init_state.unsqueeze(1), a_out, ddh_out, cur_veh_len)[:,0,0]
                cur_bike_state = self.normalizer.normalize(cur_bike_state)

                cur_state_global = cur_bike_state[:, :4]
                # also need local frame
                cur_state_local = transform2frame(prev_state[:,:4], cur_state_global.unsqueeze(1))[:,0]
            else:
                # normalize heading to be unit vector
                heading_mag = torch.norm(decoder_out[:, 2:], dim=-1, keepdim=True)
                cur_state_local = torch.cat([decoder_out[:, :2], decoder_out[:, 2:] / heading_mag], dim=-1)
                # transform output into global frame from frame of t-1
                cur_state_global = transform2frame(prev_state,
                                                   cur_state_local.unsqueeze(1),
                                                   inverse=True)[:, 0, :]

            # save for output
            traj_out.append(cur_state_global)

            if ext_future is not None:
                # update both local and global to be the given future
                # global is used to crop map pos
                cur_state_global = cur_state_global.clone() # don't want to replace output traj - only use for input
                cur_state_global[ego_inds] = ext_future[:, t]
                # local is used to update past feature
                cur_state_local = cur_state_local.clone()
                cur_state_local[ego_inds] = transform2frame(prev_state[ego_inds][:, :4],
                                                            cur_state_global[ego_inds].unsqueeze(1))[:, 0, :]

            # update prev state
            #   NOTE: in case of ext_future, this will use the external input rather than predicted output
            if self.output_bicycle:
                prev_state = cur_bike_state
            else:
                prev_state = cur_state_global

            if t < FT - 1:
                # update past feat using memory
                cur_past_feat, cur_mem_state = self.decoder_memory(cur_state_local.unsqueeze(1), # input
                                                                    cur_mem_state) # init hidden state
                cur_past_feat = cur_past_feat[:,0]
                if mult_samp:
                    cur_past_feat = cur_past_feat.reshape(NA, NS, self.past_feat_size)
                    cur_past_feat.size()

                # crop and encode map around new position
                scene_graph.pos = cur_state_global.detach() if not mult_samp else cur_state_global.detach().reshape(NA, NS, -1)
                cur_map_feat = self.encode_map(scene_graph, map_idx, map_env)

                # update positions for relative transformations during next GNN pass
                scene_graph.pos = cur_state_global if not mult_samp else cur_state_global.reshape(NA, NS, -1)

        # return all outputs in global frame
        traj_out = torch.stack(traj_out, dim=1)
        if mult_samp:
            traj_out = traj_out.reshape(NA, NS, FT, -1)
        return traj_out

    def rsample(self, mean, var):
        '''
        Return gaussian sample of (mu, var) using reparameterization trick.
        '''
        eps = torch.randn_like(mean)
        z = mean + eps*torch.sqrt(var)
        return z

    def sim_traj(self, init_state, a, ddh, vehicle_len):
        '''
        Everything is assumed to be UNNORMALIZED.
        :param init_state: (B, NA, 6)
        :param a: acceleration profile (B, NA, FT)
        :param ddh: yaw accel profile (B, NA, FT)
        :param vehicle_len: length of vehicles (B, NA)
        '''
        cur_kinematics = kinematics2angle(init_state)
        sim_steps = a.size(-1)
        kin_seq = []
        for t in range(sim_steps):
            cur_kinematics = car_dynamics(cur_kinematics, a[:,:,t], ddh[:,:,t],
                                        self.bicycle_params['dt'], 0, 1, 2, 3,
                                        4, vehicle_len, self.bicycle_params['maxhdot'],
                                        self.bicycle_params['maxs'])
            kin_seq.append(kinematics2vec(cur_kinematics))
        
        traj_out = torch.stack(kin_seq, dim=2)
        return traj_out


