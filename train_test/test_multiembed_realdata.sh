out_path=""
test_realdata_batch_size=512
input_data_path=""
HCP_center_path="HCP_mass_center.npy"
CUDA_VISIBLE_DEVICES=0 python test_realdata.py --out_path_base ${out_path} \
               --test_realdata_batch_size ${test_realdata_batch_size} \
               --tractography_path ${input_data_path}\
               --HCP_center ${HCP_center_path}