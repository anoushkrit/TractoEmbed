model_weight_path=""  # path to folder container model weights and args
test_realdata_batch_size=512
input_data_path="" # path to input trk file
HCP_center_path="../dataloader/HCP_mass_center.npy"
output_trk_path="" # path to save output vtk of each class
CUDA_VISIBLE_DEVICES=0 python test_realdata.py --out_path_base ${out_path} \
               --test_realdata_batch_size ${test_realdata_batch_size} \
               --tractography_path ${input_data_path}\
               --HCP_center ${HCP_center_path}\
               --saved_trk_path{output_trk_path}