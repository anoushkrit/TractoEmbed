#%%

from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import pickle
import sys
import os
sys.path.append('../')
from utils.config import *
from utils.streamline_ops import *
from utils.streamline_ops import bicubic_interpolate
from utils.funcs import obtain_TractClusterMapping, cluster2tract_label
#%%
class LocalFeatLoader(data.Dataset):
    def __init__(self,
                 data_dict, 
                 k_local, 
                 k_global= 0, 
                 interpolate = (False, 40),
                 cal_equiv_dist=False, 
                 use_endpoints_dist=False, 
                 rough_num_fiber_each_iter=10000, 
                 k_ds_rate=0.1,
                 shuffle_streamlines = False):
        self.interpolate  = interpolate
        try: 
            self.feat = data_dict['feat'].astype(np.float32) 
            self.labels = data_dict['label']
            self.label_names = data_dict['label_name']
            self.subject_ids = data_dict['subject_id']
        except Exception as e:
            print(f'Error: {e}')
            assert False, "Error: training data is not in the pickle format"
        

        self.k_local= k_local
        self.k_global = k_global 
        self.cal_equiv_dist = cal_equiv_dist 
        self.use_endpoints_dist = use_endpoints_dist 
        self.rough_num_fiber_each_iter = rough_num_fiber_each_iter 
        self.k_ds_rate=k_ds_rate # k downsampling rate, default: 1
        
        num_fiber = self.feat.shape[0]
        if self.interpolate[1] is not None: 
            num_point = self.interpolate[1]
        else: 
            num_point = self.feat.shape[1]
        num_feat_per_point = self.feat.shape[2]
        
        if shuffle_streamlines == True:
            self.streamlines, self.labels, self.indices = shuffle(self.streamlines, self.labels)
        else: 
            self.indices = np.arange(len(self.labels))

        if self.interpolate[0]: 
            #TODO: Remove hardcoded 40 points per streamline
            self.labels, self.subject_ids , self.feat, _ , self.indices= bicubic_interpolate(labels = self.labels,
                                                                    subject_ids= self.subject_ids ,
                                                                    features = self.feat,
                                                                    ras_feature=self.feat,
                                                                    indices = self.indices,
                                                                    num_point_per_fiber=self.interpolate[1])
        # global feature
        if self.k_global==0:
            self.global_feat = np.zeros((1, num_point, num_feat_per_point, 1), dtype=np.float32) # [1, n_point, n_feat, 1]
            # if k_global is 0, then this creates a global feature of shape [1, n_point, n_feat, 1] where npoint = 15, and num_feat_per_point = 3 
        else:
            random_idx = np.random.randint(0, num_fiber, self.k_global)  # selecting k_global number of streamlines from the total fibers: num_fiber
            self.global_feat = self.feat[random_idx,...]  # [k_global, n_point, n_feat]. This is the global (random) feature for all fibers in a test subject
            self.global_feat = self.global_feat.transpose(1,2,0)[None,:,:,:].astype(np.float32)  # [1, n_point, n_feat, k_global] 
        

        if self.k_local==0:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, 1), dtype=np.float32) # [n_fiber, n_point, n_feat, 1]
        else:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, self.k_local), dtype=np.float32)  # [n_fiber, k, n_point, n_feat]
            num_iter = num_fiber // self.rough_num_fiber_each_iter
            self.num_fiber_each_iter = (num_fiber // num_iter) + 1

            for i_iter in range(num_iter): # type: ignore
                # per iteration calculating feature and slicing the self.feat
                cur_feat = self.feat[i_iter*self.num_fiber_each_iter:(i_iter+1)*self.num_fiber_each_iter,...]  # [n_fiber, n_point, n_feat]
                cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k_local, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
                cur_local_feat = cur_local_feat.reshape(cur_feat.shape[0], self.k_local, num_feat_per_point, num_point)  # [n_fiber, k, n_feat, n_point]
                cur_local_feat = np.transpose(cur_local_feat,(0,3,2,1))  # [n_fiber, k, n_feat, n_point]->[n_fiber, n_point, n_feat, k]
                self.local_feat[i_iter*self.num_fiber_each_iter:(i_iter+1)*self.num_fiber_each_iter,...] = cur_local_feat
        
    def __getitem__(self, index):
        point_set = self.feat[index]    # [n_point, n_feat]
        klocal_point_set = self.local_feat[index]   # [n_point, n_feat, k]
        cluster_label = self.labels[index]
        tract_cluster_mapping = obtain_TractClusterMapping()
        if type(cluster_label) is not list: 
            tract_label = cluster2tract_label([cluster_label], tract_cluster_mapping, output_lst = False)
        else: 
            tract_label = cluster2tract_label(cluster_label, tract_cluster_mapping, output_lst= False)

        subject_id = self.subject_ids[index]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            # print('Feature is not in float32 format')
        if len(tract_label) == 1: 
            tract_label = tract_label[0]
            tract_label_name = list(tract_cluster_mapping.keys())[tract_label]

        else: 
            tract_label_name = []
            for t in tract_label:
                tract_label_name.append(list(tract_cluster_mapping.keys())[t])
            tract_label = tract_label.tolist()
        return point_set, klocal_point_set, cluster_label, tract_label, tract_label_name, subject_id

    def __len__(self):
        return self.feat.shape[0]
#%%

if __name__ == '__main__':
    sys.path.append('../')
    method="s2LocalFeatLoader"
    input_path = "/neuro/tractcloud-data/TrainData_800clu800ol/"
    out_path = f"/neuro/tracto/TractoBERT/ModelWeights/{method}"
    num_fiber_per_brain = 10000
    num_point_per_fiber = 15
    use_tracts_training=False

    os.makedirs(out_path, exist_ok=True)

    logger = create_logger(out_path)
    cal_equiv_dist = False
    recenter=False
    include_org_data=False
    sample_pts=150
    k_local = 19
    k_global = 0
    interpolation_values = (False, 15)

    split = "train"
    with open(os.path.join(input_path, '{}.pickle'.format(split)), 'rb') as file:
        data_dict = pickle.load(file)

    features = data_dict['feat']
    labels = data_dict['label']
    label_names = data_dict['label_name']
    subject_ids = data_dict['subject_id']


    lfl = LocalFeatLoader( 
            data_dict = data_dict, 
            k_local = k_local, 
            k_global= k_global, 
            interpolate = interpolation_values,
            cal_equiv_dist=False, 
            use_endpoints_dist=False, 
            rough_num_fiber_each_iter=num_fiber_per_brain, 
            k_ds_rate=0.1,
            shuffle_streamlines=False)
    #%%
#point_set, klocal_point_set, cluster_label, tract_label, tract_label_name, subject_id
    # lfl[10]
    
    # lfl[100]
    num = 999+1
    klocal_point_set = lfl[num][1]
    point_set = lfl[num][0] 
    pcd = torch.concat((point_set.unsqueeze(0), klocal_point_set.transpose(2,1).transpose(0,1)), dim = 0).reshape(-1,3)

    # plotting the patches here using plotly
    import plotly.graph_objects as go

    fig = go.Figure()

    x_values = pcd[:, 0]  # X-axis values
    y_values = pcd[:, 1]  # Y-axis values
    z_values = pcd[:, 2]  # Z-axis values

    # Add the line to the plot
    fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, 
                                mode='markers',  marker=dict(size=2)))

    # Update the layout
    fig.update_layout(
        title='3D Plot of Multiple Lines',
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        legend_title='Lines'
    )

    # Show the plot
    fig.show()
# %%
