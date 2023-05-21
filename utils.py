import torch
import torch.nn as nn
import random
import numpy as np
import os
import GPUtil
import time

def compute_5_poly(adj, poly_list):
    poly_arr = []
    for poly in poly_list:
        poly_arr.append(torch.einsum(poly, adj,adj,adj,adj,adj))
    return torch.cat(poly_arr,dim=0)

def compute_6_poly(adj, poly_list):
    poly_arr = []
    n= adj.shape[-1]
    for poly in poly_list:
        if poly[-2:] == 'yz':
            poly_arr.append((torch.einsum(poly[:-2], adj,adj,adj,adj,adj,adj).view(1,1,1).repeat(1,n,n)))
        elif poly[-1] == 'z':
            poly_arr.append(torch.diag_embed(torch.einsum(poly[:-1], adj,adj,adj,adj,adj,adj)))
        else:
            poly_arr.append(torch.einsum(poly, adj,adj,adj,adj,adj,adj))
    return torch.cat(poly_arr,dim=0)

def compute_poly(adj, deg, poly_list):
    if deg == 5:
        poly = compute_5_poly(adj,poly_list)
    elif deg == 6:
        poly = compute_6_poly(adj.to('cuda:1'),poly_list)
    return poly


def set_gpu(gpu):
    if gpu[0] == '-':
        deviceIDs = []
        count = 0
        num_gpus = int(gpu[1:])
        print('searching for available gpu...')
        deviceIDs = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                            excludeUUID=[])
        print(deviceIDs)
        while len(deviceIDs) < num_gpus:
            time.sleep(60)
            count += 60
            print('Pending... | {} mins |'.format(count//60))
            deviceIDs = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                            excludeUUID=[])
            print(deviceIDs)
        gpu = deviceIDs[0:num_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)[1:-1]#.replace(' ','')

def get_name(cfg):
    name = cfg.dataset + '_poly_ppgn_' + cfg.model + '_'
    if cfg.use_poly_data:
        name = name + '{}_expressive_'.format(cfg.d_expressive)
        if cfg.normalize_features:
            name = name + 'feature_normalized' + '_'
    else:
        name = name + 'baseline_'
    name = name + 'depth_{}_hidden_dim_{}_'.format(cfg.depth,cfg.hidden_dim)
    if cfg.subset and (cfg.dataset == 'ZINC'):
        name = name + 'subset'
    else:
        name = name + 'full'
    return name

def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    return

def update_cfg_dims(cfg):
    if cfg.use_poly_data:
        if cfg.d_expressive == 5:
            cfg.input_dim += 1
        elif cfg.d_expressive == 6:
            if cfg.dataset == 'SR':
                cfg.input_dim += 12
            else:
                cfg.input_dim += 9
    return cfg