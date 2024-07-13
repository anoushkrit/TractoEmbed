"""
Borrow from https://github.com/fxia22/pointnet.pytorch

Modified by 
@Author: Tengfei Xue
@Contact: txue4133@uni.sydney.edu.au
"""
from __future__ import print_function
import sys
import yaml
sys.path.append("/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/models")
sys.path.append("/scratch/jankita.scee.iitmandi/PointBERT/PointBERT")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time 
from dvae_TC import DiscreteVAE
from collections import namedtuple

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, args, k=0, k_global=0, global_feat = True, feature_transform = False, first_feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.num_features = args.num_features
        self.batch_size = args.batch_size
        self.npps = args.num_point_per_fiber  # number of points per streamline

        self.conv1 = torch.nn.Conv1d(self.num_features, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = torch.nn.Conv1d(64,128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128) # changed from 128 
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        
        self.first_feature_transform = first_feature_transform
        self.feature_transform = feature_transform
        self.k = k
        self.k_global = k_global
        if self.first_feature_transform:
            self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, x, info_point_set ):
        # x=x[:,:1,:]
        if not x.is_cuda: 
            x.cuda()

        n_pts = x.size()[2]
        
        if self.first_feature_transform:
            trans = self.stn(x) 
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None
        
        x=info_point_set
        # x = torch.cat((x,info_point_set),dim=2)   #  (num_fiber, 3, num_points)-> (num_fiber, 3, num_points+1024)
        x = F.relu(self.bn1(self.conv1(x)))      # (num_fiber, 3*2, num_points, fiber_k) -> (num_fiber, 64, num_points, fiber_k)
            
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        # print("Shape of x: after local global features [4]",x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("x shape before last conv [5]",x.shape)

        x = self.bn3(self.conv3(x)) # (256, 1024, 15)
        x = self.bn4(self.conv4(x)) # (256, 1024, 15)
        # print("x shape before max",x.shape)

        # TODO: what does this do? 
        x = torch.max(x, 2, keepdim=True)[0]
        # print("x shape after max",x.shape)
        x = x.view(-1,1024)

        # here 1024 is linked to the input layer dimensions of the MLP layers later
        if self.global_feat:
            # print("x shape after view",x.shape)

            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # print("x shape after view",x.shape)
            return torch.cat([x, pointfeat], 1).cuda(), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, args, k=0, k_global=0, num_classes=2, feature_transform=False, first_feature_transform=False):
        super(PointNetCls, self).__init__()
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
        # print("shape of x",x.shape)   
        # print("shape of info_point_set",info_point_set.shape)
        # print("shape of ras_feat",ras_feat.shape)
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
        # print("shape of x before classification",x.shape)
        # del dvae_embedding
        x = F.relu(self.bn1(self.fc1(x)))
        # print("shape of x after fc1",x.shape)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # print("shape of x after fc2",x.shape)
        x = self.fc3(x)
        # print("shape of x after fc3 and softmax",F.log_softmax(x, dim=1).shape)
        return F.log_softmax(x, dim=1), trans, trans_feat
    
