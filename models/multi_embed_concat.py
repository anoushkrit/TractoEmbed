"""
Created by

@Author : 
@Contact : 
"""

from __future__ import print_function
import yaml
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from dvae_TC import DiscreteVAE
from collections import namedtuple
from pointnet import PointNetfeat

class MultiEmbed(nn.Module):
    def __init__(self, args, k=0, k_global=0, num_classes=2, feature_transform=False, first_feature_transform=False):
        super(MultiEmbed, self).__init__()
        self.feature_transform = feature_transform
        self.first_feature_transform = first_feature_transform
        self.k = k
        self.k_global = k_global
        self.args=args
        if(args.use_pointnet):
            self.feat = PointNetfeat( args = args, k=k, k_global=k_global, global_feat=True, feature_transform=feature_transform, first_feature_transform=first_feature_transform)
        if(args.dVAE=="gold"):
            config_path="/scratch/jankita.scee.iitmandi/TractCloud/dvae_local/gold/config.yaml"
            path="/scratch/jankita.scee.iitmandi/TractCloud/dvae_local/gold/ckpt-best.pth"
        elif(args.dVAE=="hyperlocal"):
            config_path="/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/param_trial/dvae/tract_classification/hyperlocal_fss_1024/config.yaml"
            path="/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/param_trial/dvae/tract_classification/hyperlocal_fss_1024/ckpt-best.pth"
        elif(args.dVAE=="goldy"):
            config_path="/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/param_trial/dvae/tract_classification/GOLD_dvae_16pts/config.yaml"
            path="/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/param_trial/dvae/tract_classification/GOLD_dvae_16pts/ckpt-best.pth"
        else:
            config_path="/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/param_trial/TCdvae/tract_classification/B2e15_SL15/config.yaml"  #new local dvae
            path="/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/param_trial/TCdvae/tract_classification/B2e15_SL15/ckpt-best.pth"
        
        if(args.use_dvae):
            with open(config_path,'r') as file:
                config=yaml.safe_load(file)
            
            Config = namedtuple('Config', config["model"].keys())
            config_obj = Config(**config["model"])

            self.dvae = DiscreteVAE(config_obj).cuda()
        
            #new local dvae
            checkpoint = torch.load(path)
            state_dict = checkpoint['base_model']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value

            self.dvae.load_state_dict(new_state_dict)
            if(args.dVAE=="hyperlocal"):
                self.dvae_fc1=nn.Linear(4096, 2048)
                self.dvae_fc2=nn.Linear(2048, 1024)
                self.dvae_bn1 = nn.BatchNorm1d(2048) 
            else:
                self.dvae_fc1=nn.Linear(16384, 4096)
                self.dvae_fc2=nn.Linear(4096, 1024)
                self.dvae_bn1 = nn.BatchNorm1d(4096) 
        
        if(args.use_dvae & args.use_cnn & args.use_pointnet):
            self.fc1 = nn.Linear(2304, 1024) # 2048, 512
            self.fc2 = nn.Linear(1024, 512) # 2048, 256
            self.fc3 = nn.Linear(512, num_classes) # 256
            self.dropout = nn.Dropout(p=0.3)
            # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
            self.bn1 = nn.BatchNorm1d(1024) # 2048
            self.bn2 = nn.BatchNorm1d(512) #256
            self.relu = nn.ReLU()
        elif(args.use_dvae & args.use_pointnet):
            self.fc1 = nn.Linear(2048, 1024) # 2048, 512
            self.fc2 = nn.Linear(1024, 512) # 2048, 256
            self.fc3 = nn.Linear(512, num_classes) # 256
            self.dropout = nn.Dropout(p=0.3)
            # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
            self.bn1 = nn.BatchNorm1d(1024) # 2048
            self.bn2 = nn.BatchNorm1d(512) #256
            self.relu = nn.ReLU()
        elif((args.use_cnn & args.use_pointnet) | (args.use_cnn & args.use_dvae)):
            self.fc1 = nn.Linear(1280, 1024) # 2048, 512
            self.fc2 = nn.Linear(1024, 512) # 2048, 256
            self.fc3 = nn.Linear(512, num_classes) # 256
            self.dropout = nn.Dropout(p=0.3)
            # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
            self.bn1 = nn.BatchNorm1d(1024) # 2048
            self.bn2 = nn.BatchNorm1d(512) #256
            self.relu = nn.ReLU()
        elif(args.use_dvae):
            self.fc1 = nn.Linear(1024, 512) # 2048, 512
            self.fc2 = nn.Linear(512, 256) # 2048, 256
            self.fc3 = nn.Linear(256, num_classes) # 256
            self.dropout = nn.Dropout(p=0.3)
            # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
            self.bn1 = nn.BatchNorm1d(512) # 2048
            self.bn2 = nn.BatchNorm1d(256) #256
            self.relu = nn.ReLU()
        elif(args.use_cnn ):
            self.fc1 = nn.Linear(256, 512) # 2048, 512
            self.fc2 = nn.Linear(512, 1024) # 2048, 256
            self.fc3 = nn.Linear(1024, num_classes) # 256
            self.dropout = nn.Dropout(p=0.3)
            # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
            self.bn1 = nn.BatchNorm1d(512) # 2048
            self.bn2 = nn.BatchNorm1d(1024) #256
            self.relu = nn.ReLU()
        
        else:
            self.fc1 = nn.Linear(1024, 512) # 2048, 512
            self.fc2 = nn.Linear(512, 256) # 2048, 256
            self.fc3 = nn.Linear(256, num_classes) # 256
            self.dropout = nn.Dropout(p=0.3)
            # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
            self.bn1 = nn.BatchNorm1d(512) # 2048
            self.bn2 = nn.BatchNorm1d(256) #256
            self.relu = nn.ReLU()
        


    def forward(self, x, info_point_set,ras_feat):
        """x (num_fiber, 3, num_points)"""

        if(self.args.use_dvae):
            # y=torch.cat((x,info_point_set),dim=2)
            y=info_point_set
            y = y.transpose(2, 1).cuda().to(torch.float32)
            y=y.contiguous()
            # print("shape of y",y.shape, y.dtype)
            y = self.dvae.latent_shape(y) #for not training dvae
            # y = self.dvae(y)   # forward pass of dvae
            # print("shape of y after dvae latent shape",y.shape)
            # del y
            y=y.reshape(y.shape[0],y.shape[1]*y.shape[2]) # flatten (bs,16384)
            # print("shape of y after reshape",y.shape)
            y=F.relu(self.dvae_bn1(self.dvae_fc1(y)))
            y=self.dvae_fc2(y)
        if(self.args.use_pointnet):
            # print("pointnet true")
            x, trans, trans_feat = self.feat(x, info_point_set)
        else:
            trans, trans_feat=None,None

        # print("shape of y after dvae fc",y.shape)
        # print("shape of x before adding dvae",x.shape)
        if(self.args.use_dvae & self.args.use_pointnet):
            x = torch.concat((x,y),dim=1)
        elif(self.args.use_dvae):
            x=y

        if((self.args.use_cnn & self.args.use_pointnet) | (self.args.use_cnn & self.args.use_dvae)):
            x=torch.concat((x,ras_feat),dim=1)
        elif(self.args.use_cnn):
            x=ras_feat
        # del dvae_embedding
        x = F.relu(self.bn1(self.fc1(x)))
        # print("shape of x after fc1",x.shape)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # print("shape of x after fc2",x.shape)
        x = self.fc3(x)
        # print("shape of x after fc3 and softmax",F.log_softmax(x, dim=1).shape)
        return F.log_softmax(x, dim=1), trans, trans_feat
    
