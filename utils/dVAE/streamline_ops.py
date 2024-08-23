
import multiprocessing
import numpy as np 
from scipy.interpolate import CubicSpline



def bicubic_interpolate_single_streamline(label, sub_id, feature,ras_feature, indices, num_point_per_fiber):
    original_x = np.arange(feature.shape[0])
    new_x = np.linspace(0, original_x[-1], num_point_per_fiber)
    num_dims = feature.shape[1]
    interpolated_streamline = np.zeros((num_point_per_fiber, num_dims))

    for dim in range(num_dims):
        cs = CubicSpline(original_x, feature[:, dim], bc_type='natural')
        interpolated_streamline[:, dim] = cs(new_x)

    return label, sub_id, interpolated_streamline,ras_feature, indices 

def bicubic_interpolate(labels, subject_ids, features, ras_feature, indices, num_point_per_fiber):
    """
    Interpolates a set of streamlines using bicubic interpolation with multiprocessing

    Args:
        labels (numpy.ndarray): An array of labels.
        subject_ids (numpy.ndarray): An array of subject IDs.
        features (numpy.ndarray): An array of features.
        ras_feature (numpy.ndarray): An array of RAS features or original streamlines which 
        won't be interpolated 
        indices (numpy.ndarray, optional): An array of indices.

    Returns:
        tuple: A tuple containing the interpolated labels, subject IDs, features,
               RAS features, and indices.

    """
    arg_streamlines = [(labels[i],subject_ids[i] ,features[i],ras_feature[i], indices[i], num_point_per_fiber) for i in range(len(features))]
    with multiprocessing.Pool() as pool:
        result_list = pool.starmap(bicubic_interpolate_single_streamline, arg_streamlines)
    labels, subject_ids, features, ras_feature, indices = zip(*result_list)
    return np.array(labels), np.array(subject_ids), np.array(features),np.array(ras_feature), np.array(indices)


def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def shuffle(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]
    return data, labels, idx

import logging
import time
import os
import sys


def create_logger(final_output_path, description=None):
    if description is None:
        log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_file = '{}_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'), description)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

def interpolate_streamline(streamlines, pps):
    """
    streamlines: np.array
    pps: int (points per streamline)
    """
    a,b,c = np.shape(streamlines)
    resampled_streamlines = np.zeros((a,pps,c))

    for i,streamline in enumerate(streamlines):
        old_points = np.linspace(0, 1, streamline.shape[0])
        new_points = np.linspace(0, 1, pps)
        
        # Interpolation functions for x, y, z
        x_interp = CubicSpline(old_points, streamline[:, 0], bc_type='natural')
        y_interp = CubicSpline(old_points, streamline[:, 1], bc_type='natural')
        z_interp = CubicSpline(old_points, streamline[:, 2], bc_type='natural')
        
        # Resample streamline
        resampled_streamline = np.vstack([
            x_interp(new_points),
            y_interp(new_points),
            z_interp(new_points)
        ]).T
        
        resampled_streamlines[i] = resampled_streamline
    return resampled_streamlines

def cal_local_feat(cur_feat, k_ds_rate=0.1, k =20 , use_endpoints_dist = True, cal_equiv_dist=False):
    import torch
    
    """ Calculate the local feature for all streamlines in cur_feat
    Args:
        cur_feat: [n_fiber, n_feat, n_point]
        k_ds_rate: the rate of downsample the fibers to calculate the distance matrix. 1 means no downsample
        k: the number of nearest neighbor streamlines (local)
        use_endpoints_dist (bool): whether to use the distance between endpoints to calculate the distance matrix
        cal_equiv_dist (bool): whether to calculate the equivalent distance matrix

    Returns:
        cur_local_feat: [n_fiber*k, n_feat, n_point], features of the k nearest neighbor streamlines
    """
    # local feat
    # [n_fiber, k], [n_fiber, k], [n_fiber,n_feat,n_point], [n_fiber,n_feat,n_point]
    near_idx, near_flip_mask, ds_cur_feat, ds_cur_feat_equiv = dist_mat_knn(
                        torch.from_numpy(cur_feat), k_ds_rate, k, use_endpoints_dist, cal_equiv_dist)
    
    #selecting downsampled local features from org, by selecting the near index streamlines
    cur_local_feat_org = ds_cur_feat[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    cur_local_feat_equiv = ds_cur_feat_equiv[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    near_flip_mask = near_flip_mask.reshape(-1)[:,None,None]  # [n_fiber*k, 1, 1]
    near_nonflip_mask = 1-near_flip_mask
    cur_local_feat = cur_local_feat_org*near_nonflip_mask + cur_local_feat_equiv*near_flip_mask  # [n_fiber*k, n_feat, n_point]

    return cur_local_feat 


def dist_mat_knn(brain_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    import torch
    from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints   

    """ 
        calculate the distance between streamlines (fibers) and then find the neighbor streamlines (fibers)
        input (self) 
            brain_feat: [n_fiber, n_feat, n_point]
            k_ds_rate (float): the rate of downsample the fibers to calculate the distance matrix
            k (int): the number of nearest neighbors
            use_endpoints_dist (bool): whether to use endpoints distance to calculate the nearest neighbor streamlines (fibers)
            cal_equiv_dist (bool): whether to calculate the equivalent distance
        output 
            idx (the nearest idx): [n_fiber, k]
            flip_mask (the flip mask): [n_fiber, k]. whether the nearest fiber feature should use flipped fiber (reverse order)
            ds_brain_feat (the downsampled brain_feat): [n_fiber, n_feat, n_point]
            ds_brain_feat_equiv (the downsampled equivalent (reverse order) brain_feat): [n_fiber, n_feat, n_point]
    """
        
    # calculate the distance matrix. Take the minus since we wanna find the smallest distance using topk.
    if 0 < k_ds_rate < 1:
        num_ds_feat =  int(brain_feat.shape[0]*k_ds_rate)
        ds_indices = np.random.choice(brain_feat.shape[0], size=num_ds_feat, replace=False)
        downsample_feat = brain_feat[ds_indices,:,:]  # [n_ds_fiber, n_point, n_feat]
    else:
        downsample_feat = brain_feat
    if use_endpoints_dist:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation_endpoints(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber) for dist_mat
    else:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber)
    
    topk_idx = dist_mat.topk(k=k, largest=False, dim=-1)[1]   # (N_fiber, k). largest is False then the k smallest elements are returned.
    near_idx = topk_idx[:,:]   # (N_fiber, k)
    # print(dist_mat[0,near_idx[0,:]])
    near_flip_mask = torch.gather(flip_mask, dim=1, index=near_idx) # (N_fiber, k). The flip mask of the info fibers (neighbor).

    return near_idx.numpy(), near_flip_mask.numpy(), ds_brain_feat.numpy(), ds_brain_feat_equiv.numpy()
