# TractoEmbed

TractoEmbed: Modular Multi-level embeddings for Tract Segmentation

Keywords: **Tract Segmentation** · **PointCloud** · **3D Computer Vision** ·
**Tractography** · **Diffusion MRI**

![multi-embedding-model-p2 drawio](https://github.com/user-attachments/assets/666cd77b-857d-4594-ad88-34a956113291)

White matter tract segmentation is a crucial task for studying brain structural connectivity and neurosurgical planning. However, segmentation remains challenging due to issues like class imbalance between major and minor tracts, structural similarity, subject variability, and symmetric streamlines between hemispheres etc. To address these challenges
We propose **TractoEmbed**, a modular multi-level embedding framework that encodes localized representations through learning task and representation specific encoders. TractoEmbed introduces a novel hierarchical streamline data representation that captures maximum spatial information at each level, including individual streamlines, clusters and patches. Experiments show that TractoEmbed clearly outperforms state-of-the-art methods in white matter tract segmentation across different datasets, spanning various age groups. The modular framework directly allows for the integration of additional embeddings in the future works.

## **init**

## Usage

### Requirements

To run TractoEmbed, the following requirements must be met:

- **PyTorch**: Version 1.7.0 or higher
- **Python**: Version 3.7
- **CUDA**: Version 10.2 or higher

Installation of dependencies can be accomplished with:

```bash
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN (For training dVAE)

_NOTE:_ PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Dataset

The processed data utilized by Tractcloud can be downloaded from the following link:
https://github.com/SlicerDMRI/TractCloud/releases. 
The dataset includes 1 million streamlines, 800 clusters, and 800 outliers.

The directory structure for the dataset is as follows:

```bash
./  TractoEmbed
├── datasets
│   ├── train.pickle
│   ├── val.pickle
│   ├── test.pickle
```

## Model Training
 
### Patch Encoder Pretraining

To train the patch encoder(dVAE), simply run:
```
bash_scripts/train.sh <GPU_IDS>\
        --config cfgs/dvae.yaml\
        --exp_name <name>
```
Replace <GPU_IDS> with the desired GPU IDs and <name> with the experiment name.

After training the patch encoder, update the dvae config path and model weight path in the ./train_test/train_multiembed.sh file.

### Streamline Encoder Pretraining

To extract streamline embeddings, use the pretrained DeepWMA model. The embeddings should be saved in the training, validation, and test pickle files under the key "cnn_embed".

For training the streamline encoder from scratch, refer to the DeepWMA repository: 
https://github.com/zhangfanmark/DeepWMA

### Cluster Encoder Training
The cluster encoder is trained in conjunction with the multiembed classification layer, eliminating the need for pretraining.

### MECL
To train the multiembed layers, run the following commands.

```
$ cd train_test
$ sh train_multiembed.sh
```
Adjust the arguments in the train_multiembed.sh file as necessary.

### Testing

### Dataset

If you find this work useful, please cite

```bibtex

```
