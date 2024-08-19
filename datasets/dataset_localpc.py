#%%
from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import pickle
import os
import whitematteranalysis as wma
from pytorch3d.transforms import RotateAxisAngle, Scale, Translate
from scipy.interpolate import CubicSpline
# from dipy.segment.fss import FastStreamlineSearch, nearest_from_matrix_row
# from nibabel.streamlines.array_sequence import ArraySequence
import multiprocessing
# import utils.tract_feat as tract_feat
from utils.funcs import obtain_TractClusterMapping, cluster2tract_label,\
    get_rot_axi, array2vtkPolyData, makepath
from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints   



def bicubic_interpolate_single_streamline(label, sub_id, feature,ras_feature, num_point_per_fiber=40):
    original_x = np.arange(feature.shape[0])
    new_x = np.linspace(0, original_x[-1], num_point_per_fiber)
    num_dims = feature.shape[1]
    interpolated_streamline = np.zeros((num_point_per_fiber, num_dims))

    for dim in range(num_dims):
        cs = CubicSpline(original_x, feature[:, dim])
        interpolated_streamline[:, dim] = cs(new_x)

    return label, sub_id, interpolated_streamline,ras_feature

def bicubic_interpolate(labels, subject_ids, features,ras_feature):
    arg_streamlines = [(labels[i],subject_ids[i] ,features[i],ras_feature[i]) for i in range(len(features))]
    with multiprocessing.Pool() as pool:
        result_list = pool.starmap(bicubic_interpolate_single_streamline, arg_streamlines)
    labels, subject_ids, features,ras_feature = zip(*result_list)
    return np.array(labels), np.array(subject_ids), np.array(features),np.array(ras_feature)


class unrelatedHCP_PatchData(data.Dataset):
    def __init__(self, root, out_path, logger, split='train', num_fiber_per_brain=10000,num_point_per_fiber=15, 
                 use_tracts_training=False, k=0, k_global=0, rot_ang_lst=[0,0,0], scale_ratio_range=[0,0], trans_dis=0.0,
                 aug_axis_lst=['LR','AP', 'SI'], aug_times=10, cal_equiv_dist=False, k_ds_rate=0.1, recenter=False, include_org_data=False,sample_pts=150):        
        self.root = root
        self.out_path = out_path
        self.split = split
        self.logger = logger
        self.num_fiber = num_fiber_per_brain
        self.num_point = num_point_per_fiber
        self.use_tracts_training = use_tracts_training
        self.k = k
        self.k_global = k_global
        self.sample_points=sample_pts
        # Augmentations, which can be replaced by the spherical coordinates
        self.rot_ang_lst = rot_ang_lst
        self.scale_ratio_range = scale_ratio_range
        self.trans_dis = trans_dis
        self.aug_axis_lst = aug_axis_lst
        self.aug_times = aug_times


        self.k_ds_rate=k_ds_rate  
        self.recenter = recenter
        self.include_org_data = include_org_data
        
        # data save for debugging
        self.save_aug_data = True
        
        # algorithm tests
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
        
        # calculate brain-level features  [n_subject, n_fiber, n_point, n_feat], labels [n_subject, n_fiber, n_point or 1]
        self.brain_features, self.brain_labels,self.brain_feat_ras = self._cal_brain_feat()
        
        # calculate local global features/representations [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k]
        # [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k], [n_subject, n_point, n_feat, k_global], [n_subject*n_fiber, 1]
        self.org_feat, self.org_label, self.local_feat, self.new_subidx,self.new_feat_ras = self._cal_info_feat()
        

    def __getitem__(self, index):
        point_set = self.org_feat[index]
        label = self.org_label[index]
        new_subidx = self.new_subidx[index]
        local_str=self.local_feat[index]
        local_pc=local_str.reshape(-1,3)
        local_pc=np.concatenate((local_pc,local_pc),axis=0)
        feat_ras=self.new_feat_ras[index]
        # print("In dataloader, shape of local_feat",local_pc.shape)
        if(local_pc.shape[0]>=self.sample_points):
            selct_idx = np.random.choice(local_pc.shape[0], self.sample_points, replace=False)
            klocal_point_set = local_pc[selct_idx]
        else:
            select_idx = np.random.choice(local_pc.shape[0], self.sample_points, replace=True)
            klocal_point_set = local_pc[select_idx]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
            feat_ras = torch.from_numpy(feat_ras)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            feat_ras = torch.from_numpy(feat_ras.astype(np.float32))
            # print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(label)
            new_subidx = torch.from_numpy(new_subidx)
        else:
            label = torch.from_numpy(label.astype(np.int64))
            new_subidx = torch.from_numpy(new_subidx.astype(np.int64))
            # print('Label is not in int64 format')

        return point_set, label, klocal_point_set, new_subidx,feat_ras

    def __len__(self):
        return self.org_feat.shape[0]


    def _cal_brain_feat(self):
        """Process data for classification and segmentation. Get data in both brain and streamline level.
           Brain features are used for calculating the local-global representation.

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
        
        # if self.aug_times > 0: # augmented data
        #     brain_features = np.zeros((num_subject*self.aug_times, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
        #     brain_labels = np.zeros((num_subject*self.aug_times, self.num_fiber,1), dtype=np.int64)
        #     aug_matrices = np.zeros((num_subject, self.aug_times, 4, 4), dtype=np.float32)
        # else:  # non-augmented data
        brain_features = np.zeros((num_subject, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
        brain_labels = np.zeros((num_subject, self.num_fiber,1), dtype=np.int64)
        brain_featras= np.zeros((num_subject, self.num_fiber, 256),dtype=np.float32)
                
                
        for i_subject, unique_id in enumerate(unique_subject_ids):  # for each subject
            cur_idxs = np.where(self.subject_ids == unique_id)[0]
            np.random.shuffle(cur_idxs)
            cur_select_idxs = cur_idxs[:self.num_fiber] #selectingtranspose only predefined number of streamlines per brain 
            cur_features = self.features[cur_select_idxs,:,:]  # [num_fiber_per_brain, num_point_per_fiber, num_feat_per_point]
            cur_labels = self.labels[cur_select_idxs, None]
            cur_feat_ras= self.ras_feat[cur_select_idxs,:]
            
            # if self.aug_times > 0:
            #     # Augmentation the brain. Note that torch.from_numpy and .numpy() return data sharing the same memory location
            #     cur_features = torch.from_numpy(cur_features)  # numpy to tensor
            #     aug_features = np.zeros((self.aug_times, *cur_features.shape))  # (aug_times, num_fiber_per_brain, num_point_per_fiber, num_feat_per_point)  
            #     for i_aug in range(self.aug_times):
            #         trot = None
            #         cur_angles = []
            #         # rotations
            #         for i, rot_ang in enumerate(self.rot_ang_lst):
            #             angle = ((torch.rand(1) - 0.5)*2*rot_ang).item()  # random angle between [-rot_ang, rot_ang] 
            #             rot_axis_name = get_rot_axi(self.aug_axis_lst[i])
            #             cur_trot = RotateAxisAngle(angle=angle, axis=rot_axis_name, degrees=True)  #  rotate around the axis by the angle
            #             cur_angles.append(round(angle,1))
            #             if trot is None:
            #                 trot = cur_trot
            #             else:
            #                 trot = trot.compose(cur_trot)
                            
            #         # scales
            #         if self.scale_ratio_range[0] == 0 and self.scale_ratio_range[1] == 0:
            #             scale_r = 1.0
            #         else:
            #             scale_r = torch.distributions.Uniform(1-self.scale_ratio_range[0], 1+self.scale_ratio_range[1]).sample().item()  # random scale between [1-scale_ratio_range[0], 1+scale_ratio_range[1]]
            #         cur_trot = Scale(scale_r) 
            #         trot = trot.compose(cur_trot)
                        
            #         # translations
            #         LR_trans = ((torch.rand(1) - 0.5)*2*self.trans_dis).item() # random translation between [-trans_dis, +trans_dis]
            #         AP_trans = ((torch.rand(1) - 0.5)*2*self.trans_dis).item() # random translation between [-trans_dis, +trans_dis] 
            #         SI_trans = ((torch.rand(1) - 0.5)*2*self.trans_dis).item() # random translation between [-trans_dis, +trans_dis]
            #         cur_trot = Translate(LR_trans, AP_trans, SI_trans)
            #         trot = trot.compose(cur_trot)
                        
            #         aug_matrices[i_subject,i_aug,:,:] = np.array(trot.get_matrix())
            #         aug_feat = trot.transform_points(cur_features.float()).numpy()  # rotate and then convert tensor to numpy
                    
            #         scale_r, LR_trans, AP_trans, SI_trans = round(scale_r,3),round(LR_trans,1),round(AP_trans,1),round(SI_trans,1)
            #         if self.recenter:
            #             aug_feat = center_tractography(self.root, aug_feat)
            #             # self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {} (centered). Aug axis order: {}'
            #                             # .format(i_subject, unique_id, i_aug, cur_angles, scale_r, [LR_trans, AP_trans, SI_trans], self.aug_axis_lst))
            #         # else:
            #             # self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {}. Aug axis order: {}'
            #                             # .format(i_subject, unique_id, i_aug, cur_angles, scale_r, [LR_trans, AP_trans, SI_trans], self.aug_axis_lst))
            #         aug_features[i_aug,...] = aug_feat
            #         # save augmented data
            #         if self.save_aug_data and i_subject < 5: # only save the first 5 subjects
            #             aug_data_save_path = os.path.join(self.out_path,'AugmentedData',self.split)
            #             makepath(aug_data_save_path)
            #             aug_feat_pd = array2vtkPolyData(aug_feat)
            #             if self.recenter:
            #                 aug_feat_name = 'SubID{}Aug{}_RotR{}A{}S{}_Scale{}_TransR{}A{}S{}_Recenter'\
            #                     .format(i_subject, i_aug, cur_angles[0],cur_angles[1],cur_angles[2],
            #                             scale_r, LR_trans, AP_trans, SI_trans)
            #             else:
            #                 aug_feat_name = 'SubID{}Aug{}_RotR{}A{}S{}_Scale{}_TransR{}A{}S{}'\
            #                     .format(i_subject, i_aug, cur_angles[0],cur_angles[1],cur_angles[2],
            #                             scale_r, LR_trans, AP_trans, SI_trans)                           
            #             aug_feat_name = aug_feat_name.replace('.', '`') + '.vtk'
            #             wma.io.write_polydata(aug_feat_pd, os.path.join(aug_data_save_path,aug_feat_name))  
            #             print('Save augmented data to {}'.format(os.path.join(aug_data_save_path,aug_feat_name)))
    
            #     brain_features[i_subject*self.aug_times:(i_subject+1)*self.aug_times, :,:,:] = aug_features
            #     # the brain features get increased by 30 times due to augmentation  
            
            # else:
            brain_features[i_subject,:,:,:] = cur_features
            brain_featras[i_subject:,:] = cur_feat_ras
            
            if self.use_tracts_training:
                # map cluster label to tract label
                ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']} 
                cur_labels = cluster2tract_label(cur_labels, ordered_tract_cluster_mapping_dict, output_lst=False)
            # if self.aug_times > 0:
            #     brain_labels[i_subject*self.aug_times:(i_subject+1)*self.aug_times,...] = cur_labels[None,...].repeat(self.aug_times, axis=0)  # [aug_times, num_fiber_per_brain, num_point_per_fiber or 1]
            # else:
            brain_labels[i_subject,...] = cur_labels
        
        # if self.aug_times > 0:      
        #     # save augmentation matrices
        #     np.save(os.path.join(self.out_path, '{}_aug_matrices.npy'.format(self.split)), aug_matrices)
        #     if self.include_org_data:
        #         assert self.num_fiber == 10000 # only support 10000 fibers for now, since each original brain has 10000 fibers
        #         org_features = self.features.reshape(num_subject, self.num_fiber, self.num_point, num_feat_per_point)
        #         org_labels = self.labels.reshape(num_subject, self.num_fiber, 1) 
        #         brain_features = np.concatenate((brain_features, org_features), axis=0)
        #         brain_labels = np.concatenate((brain_labels, org_labels), axis=0)
        #         # self.logger.info('Include {} original data in the {} data.'.format(org_features.shape[0], self.split))
        
        return brain_features, brain_labels, brain_featras
        
        
    def _cal_info_feat(self):
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
            local_feat (array): [n_subject*n_fiber, n_point, n_feat, k] k nearest neighbor streamline feature (local)
            global_feat (array): [n_subject, n_point, n_feat, 1] randomly selected streamline feature (global)
        """                   
        
        num_subjects = self.brain_features.shape[0]
        num_feat_per_point = self.brain_features.shape[-1]
        if self.k>0:
            local_feat = np.zeros((*self.brain_features.shape, self.k), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, k]
        else: # will be discarded later in the training
            local_feat = np.zeros((*self.brain_features.shape, 1), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, 1]
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, 1)  # [n_subject*n_fiber, n_point, n_feat, 1]
        # if self.k_global>0:
        #     global_feat = np.zeros((num_subjects, self.num_point, num_feat_per_point, self.k_global), dtype=np.float32) # [n_subject, n_point, n_feat, k_global]
        # else:
        #     global_feat = np.zeros((num_subjects, self.num_point, num_feat_per_point, 1), dtype=np.float32) # [n_subject, n_point, n_feat, 1]

        # calculate new sub idx no where what the value of k and k_global are.
        new_subidx = np.zeros((num_subjects, self.num_fiber), dtype=np.int64)  # [n_subject, n_fiber]
        # iterate over augmented subjects
        for cur_idx in range(num_subjects):
            cur_feat = self.brain_features[cur_idx,...]  # [n_fiber,n_point,n_feat]
            cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
            if self.k>0:
                # local feat
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
                cur_local_feat = cur_local_feat.reshape(self.num_fiber, self.k, num_feat_per_point, self.num_point)  # [n_fiber, k, n_feat, n_point]
                cur_local_feat = np.transpose(cur_local_feat,(0,3,2,1))  # [n_fiber, n_point, n_feat, k]
            # if self.k_global>0:
            #     # global feat
            #     random_idx = np.random.randint(0, cur_feat.shape[0], self.k_global)
            #     cur_global_feat = cur_feat[random_idx,...]  # [k_global, n_feat, n_point]. This is the random feature for all fibers in a test subject
            #     cur_global_feat = cur_global_feat.transpose(2,1,0)  # [n_point, n_feat, k_global]
            # new sub idx
            cur_subidx = np.ones((cur_feat.shape[0]), dtype=np.int64)*cur_idx   # [n_fiber,]
            # if self.aug_times >0:
            #     self.logger.info('Subject {} Aug {} with {} fibers feature calculation time: {:.2f} s'
            #                     .format(cur_idx//self.aug_times, cur_idx%self.aug_times, self.num_fiber, time_end-time_start))
            # else:
            #     self.logger.info('Subject {} (No Aug) with {} fibers feature calculation time: {:.2f} s'
            #                     .format(cur_idx, self.num_fiber, time_end-time_start))
            if self.k>0:
                local_feat[cur_idx,...] = cur_local_feat
            # if self.k_global>0:
            #     global_feat[cur_idx,...] = cur_global_feat
            new_subidx[cur_idx,...] = cur_subidx
        
        if self.k>0:
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, self.k)  # [n_subject*n_fiber, n_point, n_feat, k]        

        new_subidx = new_subidx.reshape(-1, 1)  # [n_subject*n_fiber, 1]
        
        # original features and labels
        fiber_feat = self.brain_features.reshape(-1, self.num_point, num_feat_per_point)  # [n_subject*n_fiber, n_point, n_feat]
        fiber_label = self.brain_labels.reshape(-1, 1)  # [n_subject*n_fiber, 1]
        fibre_feat_ras=self.brain_feat_ras.reshape(-1, 256)
        return fiber_feat, fiber_label,local_feat, new_subidx,fibre_feat_ras
    
# def cal_hyperlocal(cur_feat,local_feat,radius=6):#[n_fiber,n_feat,n_point]   [n_fiber, k, n_feat, n_point]
#     cur_feat = np.transpose(cur_feat,(0,2,1))
#     local_feat = np.transpose(local_feat,(0,1,3,2))
#     num_streamlines=cur_feat.shape[0]
#     hyper_local_feat=np.zeros((num_streamlines,150,3),dtype=np.float32)
#     for i in range(num_streamlines):
#         cur_local_feat = local_feat[i,...]
#         cur_query=cur_feat[i,...]
#         cur_hyper_local_feat = hyperlocal_fn(radius, cur_query, cur_local_feat,plot=False,interactive=False)
#         if(cur_hyper_local_feat.shape[0]==0):
#             cur_local_pc=cur_local_feat.reshape(-1,3)
#             selected_idx = np.random.randint(0, cur_local_pc.shape[0], 150)
#             localpc=cur_local_pc[selected_idx]
#         else:
#             cur_hyper_local_pc=cur_hyper_local_feat.reshape(-1,3)
#             if(cur_hyper_local_pc.shape[0]>150):
#                 selected_idx = np.random.randint(0, cur_hyper_local_pc.shape[0], 150)
#                 localpc=cur_hyper_local_pc[selected_idx]
#             else:
#                 localpc=cur_hyper_local_pc
#                 cur_local_pc=cur_local_feat.reshape(-1,3)
#                 selected_idx=np.random.randint(0, cur_local_pc.shape[0], 150-cur_hyper_local_pc.shape[0])
#                 k_kocal=cur_local_pc[selected_idx]
#                 localpc=np.concatenate((localpc,k_kocal),axis=0)
#         hyper_local_feat[i,...]=localpc


#     return hyper_local_feat

# def hyperlocal_fn(radius, 
#                             query,
#                             ref_streamlines,
#                             plot = False,
#                             interactive = False):
#     # ref_streamlines = bicubic_interpolate(ref_streamlines, nppf= 100)
#     # query = bicubic_interpolate(query)
#     ref_streamlines = ArraySequence(ref_streamlines)
#     query = ArraySequence(query[None, ...])
#     """ To get bundle specific streamlines, or very similar streamlines based on MDF distance and other
#     norm based distance. 
#     Fast Streamline Search [StOnge2022]
#     Parameters
#     @query: string of the path where the query streamline is stored
#     @hcp_ref_streamlines,: string of the path where the wbt is stored

#     Returns: a set of streamlines very similar to the query streamline from the ref_streamlines,
#     @ids_s: ids of streamlines from the source
#     @ids_ref: ids of the reference streamlines
#     @nn_dist: nearest neighbour distance 
#     @ref_streamlines,: reference ref_streamlines, or the WBT of the test subject in the same registration space as the test streamline"""

#     fs_tree = FastStreamlineSearch(ref_streamlines=query,
#                                 max_radius=radius)
#     coo_mdist_mtx = fs_tree.radius_search(ref_streamlines, radius=radius)
#     # logger2 = create_logger('my_logger', '/scratch/jankita.scee.iitmandi/TractCloud/utils/hyperlocal.log')

#     # Log some messages

#     # Extract indices of streamlines with an similar ones in the reference
#     try:
#         ids_s = np.unique(coo_mdist_mtx.row)    
#         ids_ref = np.unique(coo_mdist_mtx.col)
#         nn_s, nn_ref, nn_dist = nearest_from_matrix_row(coo_mdist_mtx)
#         del nn_s, nn_ref

#         # ref_streamlines[id_s] are the similar streamlines for query streamline 
#         ref_streamlines=np.array(ref_streamlines)
#         # logger2.info(f"similar streamlines for query streamline: {len(ids_s)}")
#         return ref_streamlines[ids_s]

#     except Exception as e:
#         return np.array([[[]]])
#     # return ids_s, ids_ref, nn_dist, ref_streamlines

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

        
def center_tractography(input_path, feat_RAS, out_path=None, logger=None, tractography_name=None,save_data=False):
    """Recenter the tractography to atlas center
        feat_RAS: [n_fiber, n_point, n_feat]"""
    HCP_center = np.load(os.path.join(input_path, 'HCP_mass_center.npy'))  # (15(n_point),3(n_feat)) from 100 unrelated HCP subjects (atlas). The calculation function is in func_intra.py
    test_subject_center = np.mean(feat_RAS, axis=0)
    displacement = HCP_center - test_subject_center
    c_feat_RAS = feat_RAS + displacement  # recenter the tractography to HCP atlas center
    if save_data:
        recenter_path = os.path.join(out_path, 'recentered_tractography')
        makepath(recenter_path)
        feat_RAS_pd = array2vtkPolyData(c_feat_RAS)
        wma.io.write_polydata(feat_RAS_pd, os.path.join(recenter_path, 'recentered_{}'.format(tractography_name)))
        # logger.info('Saved recentered tractography to {}'.format(recenter_path))
    return c_feat_RAS

import logging

def create_logger(name, log_file, level=logging.DEBUG):
    """
    Creates a logger with the specified name, log file, and logging level.

    Args:
    - name (str): The name of the logger.
    - log_file (str): The file to which logs should be written.
    - level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create file handler to log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create console handler to log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# # Example usage


# %% 