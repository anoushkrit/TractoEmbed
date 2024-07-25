#!/bin/bash
#no bicubic uncomment it
method="pointnet_only_wucd_focal"
# Training params
model_name="pointnet"            # model
epoch=30                       # epoch
batch_size=128        # batch size
lr=1e-3     
dvae="gold"                   # learning rate
# Data
input_data="Embedding" # training data, 800 clusters + 800 outliers
num_f_brain=10000               # the number of streamlines in a brain
num_p_fiber=40                # the number of points on a streamline
rot_ang_lst="45_10_10"          # data rotating
scale_ratio_range="0.45_0.05"    # data scaling
trans_dis=50        # data translation
aug_times=0      # determine how many augmented data you want in training
test_aug_times=0   # you may train on data with heavier augmentation and test on data with lighter or no augmentation.
sample_pts=190 
loss="focal" 
# Local-global representation
k="5"   # local, neighbor streamlines
k_global="0"   # global, randomly selected streamlines in the whole-brain
k_ds_rate=1  # downsample the tractography when calculating neighbor streamlines
k_point_level="5"  # point-level neighbors on one streamline
# Paths
schedular="step"
local_global_rep_folder=k${k}_kg${k_global}_ds${k_ds_rate}_kp${k_point_level}_bs${batch_size}_nf${num_f_brain}_np${num_p_fiber}_epoch${epoch}_lr${lr}
out_path=../ModelWeights/${method}Sample_pts${sample_pts}Data${input_data}_Rot${rot_ang_lst}Scale-${scale_ratio_range}Trans${trans_dis}AugTimes${aug_times}_Unrelated100HCP_${model_name}/${local_global_rep_folder}
# out_path="/scratch/jankita.scee.iitmandi/TractCloud/ModelWeights/golddVAE_k20_300localpc_focalloss_wucdDataEmbedding_Rot45_10_10Scale-0.45_0.05Trans50AugTimes0_Unrelated100HCP_pointnet/k20_kg0_ds1_kp5_bs128_nf10000_np15_epoch40_lr1e-3"
input_path=../${input_data}
export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python train_localpc.py --include_org_data \
                --recenter \
                --k_ds_rate ${k_ds_rate} \
                --rot_ang_lst ${rot_ang_lst} \
                --scale_ratio_range ${scale_ratio_range} \
                --trans_dis ${trans_dis} \
                --aug_times ${aug_times} \
                --k ${k} \
                --k_point_level ${k_point_level} \
                --k_global ${k_global} \
                --num_fiber_per_brain ${num_f_brain} \
                --num_point_per_fiber ${num_p_fiber} \
                --input_path ${input_path} \
                --epoch ${epoch} \
                --out_path_base ${out_path} \
                --model_name $model_name \
                --train_batch_size $batch_size \
                --val_batch_size $batch_size \
                --test_batch_size $batch_size  \
                --lr ${lr} \
                --sample_pts ${sample_pts}\
                --dVAE ${dvae}\
                --use_pointnet \
                --use_dvae \
                --use_cnn \
                --loss ${loss}\
                --scheduler ${schedular}
# python test.py --out_path_base ${out_path} \
#                --aug_times ${test_aug_times} \
