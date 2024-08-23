#!/bin/bash
method="testing_working"
epoch=2             # Number of epochs
batch_size=8        # Batch Size
lr=1e-3             # Learning rate
dvae_config_path="E:\work\ICPR\TractoEmbed\dataset\config.yaml"         # dVAE configuration file
dvae_weight_path="E:\work\ICPR\TractoEmbed\dataset\ckpt-best.pth"       # dVAE model path
# Data
input_data="dataset"         # Folder name having training, validation and test data
num_f_brain=10               # Number of streamlines in a brain
num_p_fiber=40               # Number of points to take in a streamline after bicubuc interpolation
sample_pts=190               # Number of points to sample from cluster data
loss="focal" 
k="5"                        # Number of hyperlocal streamlines
k_ds_rate=1                  # Downsample the tractography when calculating neighbor streamlines
k_point_level="5"            # Point-level neighbors on one streamline
# Paths
schedular="step"            # Learning Rate schedular : step | wucd | reduceonplateau
folder_name=k${k}_ds${k_ds_rate}_kp${k_point_level}_bs${batch_size}_nf${num_f_brain}_np${num_p_fiber}_epoch${epoch}_lr${lr}
out_path=../ModelWeights/Method_${method}_Sample_pts${sample_pts}Data${input_data}_TractoEmbed/${folder_name}
input_path=../${input_data}
export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python train.py --k_ds_rate ${k_ds_rate} \
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
                --use_cnn \
                --loss ${loss}\
                --scheduler ${schedular}
# CUDA_VISIBLE_DEVICES=0  python test.py --out_path_base ${out_path} \
