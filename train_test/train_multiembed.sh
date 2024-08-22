#!/bin/bash
#no bicubic uncomment it
method="pointnet_only_wucd_focal"
# Training params
# model_name="pointnet"            # model
epoch=30                       # epoch
batch_size=128        # batch size
lr=1e-3     
dvae_config_path="/scratch/jankita.scee.iitmandi/TractCloud/dvae_local/gold/config.yaml"  
dvae_weight_path="/scratch/jankita.scee.iitmandi/TractCloud/dvae_local/gold/ckpt-best.pth"
# Data
input_data="Embedding" # training data, 800 clusters + 800 outliers
num_f_brain=10000               # the number of streamlines in a brain
num_p_fiber=40                # the number of points on a streamline
# rot_ang_lst="45_10_10"          # data rotating
# scale_ratio_range="0.45_0.05"    # data scaling
# trans_dis=50        # data translation
sample_pts=190 
loss="focal" 
# Local-global representation
k="5"   # local, neighbor streamlines
k_ds_rate=1  # downsample the tractography when calculating neighbor streamlines
k_point_level="5"  # point-level neighbors on one streamline
# Paths
schedular="step"
folder_name=k${k}_ds${k_ds_rate}_kp${k_point_level}_bs${batch_size}_nf${num_f_brain}_np${num_p_fiber}_epoch${epoch}_lr${lr}
out_path=../ModelWeights/Method_${method}_Sample_pts${sample_pts}Data${input_data}_Unrelated100HCP_TractoEmbed/${folder_name}
# out_path="/scratch/jankita.scee.iitmandi/TractCloud/ModelWeights/golddVAE_k20_300localpc_focalloss_wucdDataEmbedding_Rot45_10_10Scale-0.45_0.05Trans50AugTimes0_Unrelated100HCP_pointnet/k20_kg0_ds1_kp5_bs128_nf10000_np15_epoch40_lr1e-3"
input_path=../${input_data}
export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python train_localpc.py --include_org_data \
                --recenter \
                --k_ds_rate ${k_ds_rate} \
                --k ${k} \
                --k_point_level ${k_point_level} \
                --num_fiber_per_brain ${num_f_brain} \
                --num_point_per_fiber ${num_p_fiber} \
                --input_path ${input_path} \
                --epoch ${epoch} \
                --out_path_base ${out_path} \
                --train_batch_size $batch_size \
                --val_batch_size $batch_size \
                --test_batch_size $batch_size  \
                --lr ${lr} \
                --sample_pts ${sample_pts}\
                --dvae_config_path ${dvae_config_path}\
                --dvae_weight_path ${dvae_weight_path}\
                --use_pointnet \
                --use_dvae \
                --use_cnn \
                --loss ${loss}\
                --scheduler ${schedular}
python test.py --out_path_base ${out_path} \
