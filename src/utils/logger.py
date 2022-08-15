# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, re, datetime, shutil

class Logger(object):
    '''
    "Static" class to handle logging.
    '''
    log_file = None

    @staticmethod
    def init(log_path):
        Logger.log_file = log_path

    @staticmethod
    def log(write_str):
        print(write_str)
        if not Logger.log_file:
            print('Logger must be initialized before logging!')
            return
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with open(Logger.log_file, 'a') as f:
            f.write(time_str + '  ')
            f.write(str(write_str) + '\n')

def throw_err(err_str):
    '''
    Logs and throws a runtime error.
    '''
    Logger.log('ERROR: %s' % (err_str))
    raise RuntimeError(err_str)