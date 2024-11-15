import numpy as  np
import torch
import torch_scatter

def BEV_fusion(xyz, feats_2d, feats_3d, max_space=[5000,5000,5000], min_space=[0,0,0], grid_size=[480,360,32]):
    max_bound = np.asarray(max_space)
    min_bound = np.asarray(min_space)
    cur_grid_size = np.asarray(grid_size)
    crop_range = max_bound - min_bound
    intervals = crop_range/(cur_grid_size-1)
    grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals))[:,:2]
    unq, unq_inv, unq_cnt = torch.unique(grid_ind, return_inverse=True, return_counts=True, dim=0)
    unq = unq.type(torch.int64)
    unq_inv = unq_inv.cuda()
    pooled_feats_3d = torch_scatter.scatter_max(feats_3d, unq_inv, dim=0)[0]
    pooled_feats_2d = torch_scatter.scatter_max(feats_2d, unq_inv, dim=0)[0]
    cat_feats = torch.cat([pooled_feats_3d, pooled_feats_2d], dim=1)
    return cat_feats, pooled_feats_3d, unq_inv, unq_cnt

