from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import pickle
import os
from scipy.interpolate import CubicSpline
from dipy.segment.fss import FastStreamlineSearch, nearest_from_matrix_row
from nibabel.streamlines.array_sequence import ArraySequence
import multiprocessing
from utils.funcs import obtain_TractClusterMapping, cluster2tract_label,\
    get_rot_axi, array2vtkPolyData, makepath
from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints   


class HCP_Data(data.Dataset):
    def __init__(self, root, out_path, logger, split='train', num_fiber_per_brain=10000,num_point_per_fiber=15, 
                 use_tracts_training=False, k=0, cal_equiv_dist=False, k_ds_rate=0.1, sample_pts=150):        
        self.root = root
        self.out_path = out_path
        self.split = split
        self.logger = logger
        self.num_fiber = num_fiber_per_brain
        self.num_point = num_point_per_fiber
        self.use_tracts_training = use_tracts_training
        self.k = k
        self.sample_points=sample_pts
        self.k_ds_rate=k_ds_rate  
        self.save_aug_data = True
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = False
        self.logger.info('cal_equiv_dist: {}, use_endpoints_dist: {}'
                    .format(self.cal_equiv_dist, self.use_endpoints_dist))
                
        # load data
        with open(os.path.join(root, '{}.pickle'.format(split)), 'rb') as file:
            # Load the data from the file
            data_dict = pickle.load(file)
            
        self.features = data_dict['feat']
        self.labels = data_dict['label']
        self.label_names = data_dict['label_name']
        self.subject_ids = data_dict['subject_id']
        self.ras_feat = data_dict['cnn_embed'].numpy()
        
        # bicubic interpolation
        self.labels, self.subject_ids, self.features,self.ras_feat = bicubic_interpolate(self.labels, self.subject_ids, self.features,self.ras_feat)
        
        self.logger.info('Load {} data'.format(self.split))
        
        # calculate brain-level features  
        # [n_subject, n_fiber, n_point, n_feat], labels [n_subject, n_fiber, n_point or 1], ras_feat [n_subject, n_fiber, 256]
        self.brain_features, self.brain_labels,self.brain_feat_ras = self._cal_brain_feat()
        
        # calculate hyperlocal features representations 
        # [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k], [n_subject, n_point, n_feat, k_global], [n_subject*n_fiber, 1], [n_subject*n_fiber, 256]
        self.org_feat, self.org_label, self.hyperlocal_feat, self.new_subidx,self.new_feat_ras = self._cal_hyperlocal_feat()
        

    def __getitem__(self, index):

        streamline = self.org_feat[index]
        label = self.org_label[index]
        new_subidx = self.new_subidx[index]
        hyperlocal_streamlines=self.hyperlocal_feat[index]
        hyperlocal_pc=hyperlocal_streamlines.reshape(-1,3)
        hyperlocal_pc=np.concatenate((streamline,hyperlocal_pc),axis=0)
        feat_ras=self.new_feat_ras[index]
        if(hyperlocal_pc.shape[0]>=self.sample_points):
            selct_idx = np.random.choice(hyperlocal_pc.shape[0], self.sample_points, replace=False)
            cluster_data = hyperlocal_pc[selct_idx]
        else:
            select_idx = np.random.choice(hyperlocal_pc.shape[0], self.sample_points, replace=True)
            cluster_data = hyperlocal_pc[select_idx]
        if streamline.dtype == 'float32':
            streamline = torch.from_numpy(streamline)
            cluster_data = torch.from_numpy(cluster_data)
            feat_ras = torch.from_numpy(feat_ras)
        else:
            streamline = torch.from_numpy(streamline.astype(np.float32))
            cluster_data = torch.from_numpy(cluster_data.astype(np.float32))
            feat_ras = torch.from_numpy(feat_ras.astype(np.float32))

        if label.dtype == 'int64':
            label = torch.from_numpy(label)
            new_subidx = torch.from_numpy(new_subidx)
        else:
            label = torch.from_numpy(label.astype(np.int64))
            new_subidx = torch.from_numpy(new_subidx.astype(np.int64))

        return streamline, label, cluster_data, new_subidx,feat_ras

    def __len__(self):
        return self.org_feat.shape[0]


    def _cal_brain_feat(self):
        """Get data in both brain and streamline level.
           Brain features are used for calculating the hyperlocal representation.

        Args (self):
            features (array):[n_fiber, n_point, n_feat] original feature in fiber-level 
            subject_ids (array): [n_fiber] subject id for streamlines
            num_fiber_per_brain (int): number of fiber per brain
            num_point_per_fiber (int): number of point per fiber
            use_tracts_training (bool): whether to use tract label for training
        Returns:
            brain_feature (array):[n_subject, n_fiber, n_point, n_feat] feature in brain-level 
            brain_label (array): [n_subject, n_fiber, n_point or 1] label in brain-level
        """ 
        
        num_feat_per_point = self.features.shape[2]
        unique_subject_ids = np.unique(self.subject_ids)
        num_subject = len(unique_subject_ids)
        
        brain_features = np.zeros((num_subject, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
        brain_labels = np.zeros((num_subject, self.num_fiber,1), dtype=np.int64)
        brain_featras= np.zeros((num_subject, self.num_fiber, 256),dtype=np.float32)
                
                
        for i_subject, unique_id in enumerate(unique_subject_ids):  # for each subject
            cur_idxs = np.where(self.subject_ids == unique_id)[0]
            np.random.shuffle(cur_idxs)
            cur_select_idxs = cur_idxs[:self.num_fiber]     #selecting  only predefined number of streamlines per brain 
            cur_features = self.features[cur_select_idxs,:,:]  # [num_fiber_per_brain, num_point_per_fiber, num_feat_per_point]
            cur_labels = self.labels[cur_select_idxs, None]
            cur_feat_ras= self.ras_feat[cur_select_idxs,:]
            brain_features[i_subject,:,:,:] = cur_features
            brain_featras[i_subject:,:] = cur_feat_ras
            
            if self.use_tracts_training:
                # map cluster label to tract label
                ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']} 
                cur_labels = cluster2tract_label(cur_labels, ordered_tract_cluster_mapping_dict, output_lst=False)

            brain_labels[i_subject,...] = cur_labels

        return brain_features, brain_labels, brain_featras
        
        
    def _cal_hyperlocal_feat(self):
        """
        Calculate hyper local representations
        Args (self):
            n_subject=num_unique_subjects*self.aug_times
            brain_feature (array):[n_subject, n_fiber, n_point, n_feat] feature in brain-level 
            brain_label (array): [n_subject, n_fiber, n_point or 1] label in brain-level
            k (int, optional): How many k nearest neighbors are needed.
        Returns:
            fiber_feat (array): [n_subject*n_fiber, n_point, n_feat] feature in fiber-level
            fiber_label (array): [n_subject*n_fiber, n_point or 1] label in fiber-level
            hyperlocal_feat (array): [n_subject*n_fiber, n_point, n_feat, k] k nearest neighbor streamline features  
        """                   
        
        num_subjects = self.brain_features.shape[0]
        num_feat_per_point = self.brain_features.shape[-1]
        if self.k>0:
            local_feat = np.zeros((*self.brain_features.shape, self.k), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, k]
        else: # will be discarded later in the training
            local_feat = np.zeros((*self.brain_features.shape, 1), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, 1]
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, 1)  # [n_subject*n_fiber, n_point, n_feat, 1]

        # calculate new sub idx no where what the value of k and k_global are.
        new_subidx = np.zeros((num_subjects, self.num_fiber), dtype=np.int64)  # [n_subject, n_fiber]

        for cur_idx in range(num_subjects):
            cur_feat = self.brain_features[cur_idx,...]  # [n_fiber,n_point,n_feat]
            cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
            if self.k>0:
                # local feat
                cur_hyperlocal_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
                cur_hyperlocal_feat = cur_hyperlocal_feat.reshape(self.num_fiber, self.k, num_feat_per_point, self.num_point)  # [n_fiber, k, n_feat, n_point]
                cur_hyperlocal_feat = np.transpose(cur_hyperlocal_feat,(0,3,2,1))  # [n_fiber, n_point, n_feat, k]

            cur_subidx = np.ones((cur_feat.shape[0]), dtype=np.int64)*cur_idx   # [n_fiber,]
            if self.k>0:
                local_feat[cur_idx,...] = cur_hyperlocal_feat
            new_subidx[cur_idx,...] = cur_subidx
        
        if self.k>0:
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, self.k)  # [n_subject*n_fiber, n_point, n_feat, k]        

        new_subidx = new_subidx.reshape(-1, 1)  # [n_subject*n_fiber, 1]
        
        # original features and labels
        fiber_feat = self.brain_features.reshape(-1, self.num_point, num_feat_per_point)  # [n_subject*n_fiber, n_point, n_feat]
        fiber_label = self.brain_labels.reshape(-1, 1)  # [n_subject*n_fiber, 1]
        fibre_feat_ras=self.brain_feat_ras.reshape(-1, 256)
        return fiber_feat, fiber_label,local_feat, new_subidx,fibre_feat_ras
    
class RealData(data.Dataset):
    def __init__(self, feat, args, logger=None):
        self.logger = logger
        self.num_point = args.num_point_per_fiber
        self.use_tracts_training = args.use_tracts_training
        self.k = args.k
        self.sample_points=args.sample_pts
        self.k_ds_rate=args.k_ds_rate  
        self.cal_equiv_dist = args.cal_equiv_dist
        self.use_endpoints_dist = False
        self.k = args.k
        self.k_global = args.k_global

        self.features = feat.astype(np.float32) 
         # self.labels=np.zeros(feat.shape[0])
        # self.ras_feat=np.zeros(feat.shape[0],256)

        # bicubic interpolation
        self.features = bicubic_interpolate_single_arg(self.features)
    
        # calculate hyperlocal features representations 
        # [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k], [n_subject, n_point, n_feat, k_global], [n_subject*n_fiber, 1], [n_subject*n_fiber, 256]
        self.hyperlocal_feat=self.calhyperloc()
            
    def __getitem__(self, index):
        streamline = self.features[index]
        # label = 0
        # new_subidx = 0
        hyperlocal_streamlines=self.hyperlocal_feat[index]
        hyperlocal_pc=hyperlocal_streamlines.reshape(-1,3)
        hyperlocal_pc=np.concatenate((streamline,hyperlocal_pc),axis=0)
        feat_ras=self.ras_feat[index]
        if(hyperlocal_pc.shape[0]>=self.sample_points):
            selct_idx = np.random.choice(hyperlocal_pc.shape[0], self.sample_points, replace=False)
            cluster_data = hyperlocal_pc[selct_idx]
        else:
            select_idx = np.random.choice(hyperlocal_pc.shape[0], self.sample_points, replace=True)
            cluster_data = hyperlocal_pc[select_idx]
        if streamline.dtype == 'float32':
            streamline = torch.from_numpy(streamline)
            cluster_data = torch.from_numpy(cluster_data)
            feat_ras = torch.from_numpy(feat_ras)
        else:
            streamline = torch.from_numpy(streamline.astype(np.float32))
            cluster_data = torch.from_numpy(cluster_data.astype(np.float32))
            feat_ras = torch.from_numpy(feat_ras.astype(np.float32))

        if label.dtype == 'int64':
            label = torch.from_numpy(label)
            new_subidx = torch.from_numpy(new_subidx)
        else:
            label = torch.from_numpy(label.astype(np.int64))
            new_subidx = torch.from_numpy(new_subidx.astype(np.int64))

        return streamline, label, cluster_data, new_subidx,feat_ras

    def __len__(self):
        return self.feat.shape[0]
    
    def _cal_hyperlocal_feat(self):
        if self.k>0:
            local_feat = np.zeros((*self.features.shape, self.k), dtype=np.float32) # [n_fiber, n_point, n_feat, k]
        else: # will be discarded later in the training
            local_feat = np.zeros((*self.features.shape, 1), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, 1]

        num_feat_per_point = self.features.shape[-1]
        cur_feat = self.features # [n_fiber,n_point,n_feat]
        cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
        if self.k>0:
            # local feat
            cur_hyperlocal_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
            cur_hyperlocal_feat = cur_hyperlocal_feat.reshape(self.num_fiber, self.k, num_feat_per_point, self.num_point)  # [n_fiber, k, n_feat, n_point]
            cur_hyperlocal_feat = np.transpose(cur_hyperlocal_feat,(0,3,2,1))  # [n_fiber, n_point, n_feat, k]

        if self.k>0:
            local_feat= cur_hyperlocal_feat

        return local_feat
    
    
def cal_hyperlocal(cur_feat,local_feat,radius=6):#[n_fiber,n_feat,n_point]   [n_fiber, k, n_feat, n_point]
    cur_feat = np.transpose(cur_feat,(0,2,1))
    local_feat = np.transpose(local_feat,(0,1,3,2))
    num_streamlines=cur_feat.shape[0]
    hyper_local_feat=np.zeros((num_streamlines,150,3),dtype=np.float32)
    for i in range(num_streamlines):
        cur_local_feat = local_feat[i,...]
        cur_query=cur_feat[i,...]
        cur_hyper_local_feat = hyperlocal_fn(radius, cur_query, cur_local_feat,plot=False,interactive=False)
        if(cur_hyper_local_feat.shape[0]==0):
            cur_local_pc=cur_local_feat.reshape(-1,3)
            selected_idx = np.random.randint(0, cur_local_pc.shape[0], 150)
            localpc=cur_local_pc[selected_idx]
        else:
            cur_hyper_local_pc=cur_hyper_local_feat.reshape(-1,3)
            if(cur_hyper_local_pc.shape[0]>150):
                selected_idx = np.random.randint(0, cur_hyper_local_pc.shape[0], 150)
                localpc=cur_hyper_local_pc[selected_idx]
            else:
                localpc=cur_hyper_local_pc
                cur_local_pc=cur_local_feat.reshape(-1,3)
                selected_idx=np.random.randint(0, cur_local_pc.shape[0], 150-cur_hyper_local_pc.shape[0])
                k_kocal=cur_local_pc[selected_idx]
                localpc=np.concatenate((localpc,k_kocal),axis=0)
        hyper_local_feat[i,...]=localpc


    return hyper_local_feat

def hyperlocal_fn(radius, 
                            query,
                            ref_streamlines,
                            plot = False,
                            interactive = False):

    ref_streamlines = ArraySequence(ref_streamlines)
    query = ArraySequence(query[None, ...])
    """ To get bundle specific streamlines, or very similar streamlines based on MDF distance and other
    norm based distance. 
    Fast Streamline Search [StOnge2022]
    Parameters
    @query: string of the path where the query streamline is stored
    @hcp_ref_streamlines,: string of the path where the wbt is stored

    Returns: a set of streamlines very similar to the query streamline from the ref_streamlines,
    @ids_s: ids of streamlines from the source
    @ids_ref: ids of the reference streamlines
    @nn_dist: nearest neighbour distance 
    @ref_streamlines,: reference ref_streamlines, or the WBT of the test subject in the same registration space as the test streamline"""

    fs_tree = FastStreamlineSearch(ref_streamlines=query,
                                max_radius=radius)
    coo_mdist_mtx = fs_tree.radius_search(ref_streamlines, radius=radius)
    try:
        ids_s = np.unique(coo_mdist_mtx.row)    
        ids_ref = np.unique(coo_mdist_mtx.col)
        nn_s, nn_ref, nn_dist = nearest_from_matrix_row(coo_mdist_mtx)
        del nn_s, nn_ref

        ref_streamlines=np.array(ref_streamlines)
        return ref_streamlines[ids_s]

    except Exception as e:
        return np.array([[[]]])

def cal_local_feat(cur_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):

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

        
def bicubic_interpolate_single_streamline(label, sub_id, feature,ras_feature, num_point_per_fiber=40):
    original_x = np.arange(feature.shape[0])
    new_x = np.linspace(0, original_x[-1], num_point_per_fiber)
    num_dims = feature.shape[1]
    interpolated_streamline = np.zeros((num_point_per_fiber, num_dims))

    for dim in range(num_dims):
        cs = CubicSpline(original_x, feature[:, dim])
        interpolated_streamline[:, dim] = cs(new_x)

    return label, sub_id, interpolated_streamline,ras_feature

def bicubic_interpolate_single_streamline_single_arg(feature, num_point_per_fiber=40):
    original_x = np.arange(feature.shape[0])
    new_x = np.linspace(0, original_x[-1], num_point_per_fiber)
    num_dims = feature.shape[1]
    interpolated_streamline = np.zeros((num_point_per_fiber, num_dims))

    for dim in range(num_dims):
        cs = CubicSpline(original_x, feature[:, dim])
        interpolated_streamline[:, dim] = cs(new_x)

    return interpolated_streamline

def bicubic_interpolate(labels, subject_ids, features,ras_feature):
    arg_streamlines = [(labels[i],subject_ids[i] ,features[i],ras_feature[i]) for i in range(len(features))]
    with multiprocessing.Pool() as pool:
        result_list = pool.starmap(bicubic_interpolate_single_streamline, arg_streamlines)
    labels, subject_ids, features,ras_feature = zip(*result_list)
    return np.array(labels), np.array(subject_ids), np.array(features),np.array(ras_feature)

def bicubic_interpolate_single_arg(features):
    with multiprocessing.Pool() as pool:
        new_features = pool.starmap(bicubic_interpolate_single_streamline, features)
    return np.array(new_features)
