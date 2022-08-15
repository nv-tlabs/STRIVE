# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, re

class Struct(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, Struct(b) if isinstance(b, dict) else b)

    def __repr__(self):
        return '<%s>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def dict2obj(dict):
    '''
    Converts a dictionary to an object.
    :param dict: dictionary
    :return: object with same values
    '''
    return Struct(dict)

def mkdir(path):
    '''
    Makes a directory at the given path if doesn't already exist.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def log(log_out, write_str):
    with open(log_out, 'a') as f:
        f.write(str(write_str) + '\n')
    print(write_str)