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

- PyTorch >= 1.7.0
- python == 3.7
- CUDA >= 10.2

```
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

We have used the processed data used by Tractcloud. You can directly download our processed data at https://github.com/SlicerDMRI/TractCloud/releases (1 million streamlines, 800 clusters & 800 outliers). This is the streamline dataset.

```bash
./  TractoEmbed
├── datasets
│   ├── train.pickle
│   ├── val.pickle
│   ├── test.pickle
```

## Model

### Patch Encoder Pretraining


### Streamline Encoder Pretraining

Use the pretrained DeepWMA to extract the streamline encoding of each streamline from the last layer after the max pool layer and save it in the training, validation and test pickle files respectively with key "cnn_embed".

To train the streamline encoder from the scratch, you can check out: https://github.com/zhangfanmark/DeepWMA

#### Cluster Encoder Training
The cluster encoder is trained along with multiembed classification layer. There is no need to pretrain the cluster encoder.

### MECL
To train the multiembed layers, run the following commands.

```
$ cd train_test
$ sh train_multiembed.sh
```

You can also change the arguments in the train_multiembed.sh file.

### Testing

### Dataset

If you find this work useful, please cite

```bibtex

```
