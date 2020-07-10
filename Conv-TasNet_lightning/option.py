# -*- encoding: utf-8 -*-
'''
@Filename    :option.py
@Time        :2020/07/10 23:23:10
@Author      :Kai Li
@Version     :1.0
'''

import os
import yaml

def parse(opt_path, is_train=True):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    with open(opt_path,mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)
    # Export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])

    # is_train into option
    opt['is_train'] = is_train

    return opt


if __name__ == "__main__":
    parse('./train/train.yml')