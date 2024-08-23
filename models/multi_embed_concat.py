from __future__ import print_function
import yaml
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.dVAE.dvae_TC import DiscreteVAE
from collections import namedtuple
from pointnet import PointNetfeat

class MultiEmbed(nn.Module):
    
    def __init__(self, args, k=0, num_classes=1600, feature_transform=False, first_feature_transform=False):
        super(MultiEmbed, self).__init__()
        self.feature_transform = feature_transform
        self.first_feature_transform = first_feature_transform
        self.k = k
        self.args=args

        if(args.use_pointnet):
            self.feat = PointNetfeat( args = args, k=k, global_feat=True, feature_transform=feature_transform, first_feature_transform=first_feature_transform)
            
        if(args.use_dvae):
            config_path=args.dvae_config_path
            path=args.dvae_weight_path
            with open(config_path,'r') as file:
                config=yaml.safe_load(file)
            
            Config = namedtuple('Config', config["model"].keys())
            config_obj = Config(**config["model"])

            self.dvae = DiscreteVAE(config_obj).cuda()
        
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
            self.dvae_fc1=nn.Linear(16384, 4096)
            self.dvae_fc2=nn.Linear(4096, 1024)
            self.dvae_bn1 = nn.BatchNorm1d(4096) 
        
        if(args.use_dvae & args.use_cnn & args.use_pointnet):
            self.fc1 = nn.Linear(2304, 1024) 
            self.fc2 = nn.Linear(1024, 512) 
            self.fc3 = nn.Linear(512, num_classes) 
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(1024) 
            self.bn2 = nn.BatchNorm1d(512) 
            self.relu = nn.ReLU()
        elif(args.use_dvae & args.use_pointnet):
            self.fc1 = nn.Linear(2048, 1024) 
            self.fc2 = nn.Linear(1024, 512) 
            self.fc3 = nn.Linear(512, num_classes) 
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(1024) 
            self.bn2 = nn.BatchNorm1d(512) 
            self.relu = nn.ReLU()
        elif((args.use_cnn & args.use_pointnet) | (args.use_cnn & args.use_dvae)):
            self.fc1 = nn.Linear(1280, 1024) 
            self.fc2 = nn.Linear(1024, 512) 
            self.fc3 = nn.Linear(512, num_classes) 
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(1024) 
            self.bn2 = nn.BatchNorm1d(512) 
            self.relu = nn.ReLU()
        elif(args.use_dvae):
            self.fc1 = nn.Linear(1024, 512) 
            self.fc2 = nn.Linear(512, 256) 
            self.fc3 = nn.Linear(256, num_classes) 
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(512) 
            self.bn2 = nn.BatchNorm1d(256) 
            self.relu = nn.ReLU()
        elif(args.use_cnn ):
            self.fc1 = nn.Linear(256, 512) 
            self.fc2 = nn.Linear(512, 1024) 
            self.fc3 = nn.Linear(1024, num_classes) 
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(512) 
            self.bn2 = nn.BatchNorm1d(1024) 
            self.relu = nn.ReLU()
        
        else:
            self.fc1 = nn.Linear(1024, 512) 
            self.fc2 = nn.Linear(512, 256) 
            self.fc3 = nn.Linear(256, num_classes) 
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(512) 
            self.bn2 = nn.BatchNorm1d(256) 
            self.relu = nn.ReLU()
        


    def forward(self, x, cluster_data,ras_feat):
        """
        x : (batch_size , 3, num_points)
        cluster_data : (batch_size, 3, sampling_points)
        ras_feat : (batch_size, 256)
        """

        if(self.args.use_dvae):
            # Concatinate streamline with the cluster data
            y=torch.cat((x,cluster_data),dim=2)   
            y=cluster_data
            y = y.transpose(2, 1).cuda().to(torch.float32)
            y=y.contiguous()
            y = self.dvae.latent_shape(y) # for not training dvae
            # y = self.dvae(y)   # forward pass of dvae

            y=y.reshape(y.shape[0],y.shape[1]*y.shape[2]) # flatten (bs,16384)
            y=F.relu(self.dvae_bn1(self.dvae_fc1(y)))
            y=self.dvae_fc2(y)

        if(self.args.use_pointnet):
            x, trans, trans_feat = self.feat(cluster_data)

        else:
            trans, trans_feat=None,None

        if(self.args.use_dvae & self.args.use_pointnet):
            x = torch.concat((x,y),dim=1)

        elif(self.args.use_dvae):
            x=y

        if((self.args.use_cnn & self.args.use_pointnet) | (self.args.use_cnn & self.args.use_dvae)):
            x=torch.concat((x,ras_feat),dim=1)
        elif(self.args.use_cnn):
            x=ras_feat

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans, trans_feat
    
