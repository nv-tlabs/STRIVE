# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch


def kinematics2angle(kinematics):
    '''
    Converts heading vector to angle.
    :param kinematics: B x T x 6 (x,y,hx,hy,s,hdot)
    '''
    hsin = kinematics[:, :, 3:4]
    hcos = kinematics[:, :, 2:3]
    new_heading = torch.atan2(hsin, hcos)
    new_kinematics = torch.cat([kinematics[:, :, :2], new_heading, kinematics[:, :, 4:]], dim=2)
    return new_kinematics

def kinematics2vec(kinematics):
    '''
    Converts heading angle to vector.
    :param kinematics: B x T x 5 (x,y,h,s,hdot)
    '''
    # new kinematics with heading unit vector rather than angle
    hx = kinematics[:, :, 2].cos()
    hy = kinematics[:, :, 2].sin()
    hvec = torch.stack([hx, hy], dim=2)
    new_kinematics = torch.cat([kinematics[:, :, :2], hvec, kinematics[:, :, 3:]], dim=-1)
    return new_kinematics

def pairwise_transforms(poses):
    '''
    Gets all pairs of transforms (relative heading and position) between the given
    poses.

    :param poses: (B x N x 3) or (B x N x 4) of (x,y,h) or (x,y,hx,hy)

    :return: (B x N x N x 3) or (B x N x N x 4) where (b, i, j) is the pose of j in the frame of i
    '''
    B, N, D = poses.size()

    # build rotation matrices
    if D == 3:
        # get from angle
        poses_hcos = poses[:, :, 2].cos()
        poses_hsin = poses[:, :, 2].sin()
    else:
        poses_hcos = poses[:, :, 2]
        poses_hsin = poses[:, :, 3]
    # columns of pairwise transform mat
    Rp = torch.stack([poses_hcos, -poses_hsin, poses_hsin, poses_hcos], dim=2)
    Rp = Rp.view(B, N, 2, 2).unsqueeze(1).expand(B, N, N, 2, 2)
    # rows of pairwise transform mat
    Rf = torch.stack([poses_hcos, poses_hsin, -poses_hsin, poses_hcos], dim=2)
    Rf = Rf.view(B, N, 2, 2).unsqueeze(2).expand(B, N, N, 2, 2)

    # transform
    R_local = torch.matmul(Rp, Rf) # B x N x N x 2 x 2
    local_hcos = R_local[:, :, :, 0, 0]
    local_hsin = R_local[:, :, :, 1, 0]
    if D == 3:
        local_h = torch.atan2(local_hsin, local_hcos)
        local_h = local_h.unsqueeze(-1)
    else:
        local_h = torch.stack([local_hcos, local_hsin], dim=3)

    # now translation
    poses_t = poses[:, :, :2].unsqueeze(1).expand(B, N, N, 2)
    frame_t = poses_t.transpose(1, 2)
    local_t = poses_t - frame_t
    local_t = torch.matmul(Rf, local_t.reshape((B, N, N, 2, 1)))[:, :, :, :, 0]

    # all together
    local_poses = torch.cat([local_t, local_h], dim=-1)
    return local_poses


def transform2frame(frame, poses, inverse=False):
    '''
    Transform the given poses into the local frame of the given frame.
    All inputs are in the global frame unless inverse=True.

    :param frame: B x 3 where each row is (x, y, h) or B x 4 with (x,y,hx,hy) i.e. heading as a vector
    :param poses: to transform (B x N x 3) or (B x N x 4)
    :param inverse: if true, poses are assumed already in the local frame of frame, and instead transforms
                    back to the global frame based on frame.

    :return: poses (B x N x 3) or (B x N x 4), but in the local frame
    '''
    B, N, D = poses.size()

    # build rotation matrices
    # for frame
    if D == 3:
        # get from angle
        frame_hcos = frame[:, 2].cos()
        frame_hsin = frame[:, 2].sin()
    else:
        frame_hcos = frame[:, 2]
        frame_hsin = frame[:, 3]
    Rf = torch.stack([frame_hcos, frame_hsin, -frame_hsin, frame_hcos], dim=1)
    Rf = Rf.reshape((B, 1, 2, 2)).expand(B, N, 2, 2)
    # and for poses
    if D == 3:
        # get from angle
        poses_hcos = poses[:, :, 2].cos()
        poses_hsin = poses[:, :, 2].sin()
    else:
        poses_hcos = poses[:, :, 2]
        poses_hsin = poses[:, :, 3]
    Rp = torch.stack([poses_hcos, -poses_hsin, poses_hsin, poses_hcos], dim=2)
    Rp = Rp.reshape((B, N, 2, 2))

    # compute relative rotation
    if inverse:
        Rp_local = torch.matmul(Rp, Rf.transpose(2, 3))
    else:
        Rp_local = torch.matmul(Rp, Rf)
    local_hcos = Rp_local[:, :, 0, 0]
    local_hsin = Rp_local[:, :, 1, 0]
    if D == 3:
        local_h = torch.atan2(local_hsin, local_hcos)
        local_h = local_h.unsqueeze(-1)
    else:
        local_h = torch.stack([local_hcos, local_hsin], dim=2)

    # now translation
    frame_t = frame[:, :2].reshape((B, 1, 2))
    poses_t = poses[:, :, :2]
    if inverse:
        local_t = torch.matmul(Rf.transpose(2, 3), poses_t.reshape((B, N, 2, 1)))[:, :, :, 0]
        local_t = local_t + frame_t
    else:
        local_t = poses_t - frame_t
        local_t = torch.matmul(Rf, local_t.reshape((B, N, 2, 1)))[:, :, :, 0]

    # all together
    local_poses = torch.cat([local_t, local_h], dim=-1)
    return local_poses
