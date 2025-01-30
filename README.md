# TractoEmbed

TractoEmbed: Modular Multi-level embeddings for Tract Segmentation

Keywords: **Tract Segmentation** · **PointCloud** · **3D Computer Vision** ·
**Tractography** · **Diffusion MRI**

Overview of TractoEmbed, and how it fuses multiple learnt embeddings to give a tract segmentation output. 
![multi-embedding-model-overview drawio](https://github.com/user-attachments/assets/eb81eee6-60b0-402f-b65d-ca48eb03f773)
![multi-embedding-model-p2 drawio](https://github.com/user-attachments/assets/666cd77b-857d-4594-ad88-34a956113291)

White matter tract segmentation is a crucial task for studying brain structural connectivity and neurosurgical planning. However, segmentation remains challenging due to issues like class imbalance between major and minor tracts, structural similarity, subject variability, and symmetric streamlines between hemispheres etc. To address these challenges
We propose **TractoEmbed**, a modular multi-level embedding framework that encodes localized representations through learning task and representation specific encoders. TractoEmbed introduces a novel hierarchical streamline data representation that captures maximum spatial information at each level, including individual streamlines, clusters and patches. Experiments show that TractoEmbed clearly outperforms state-of-the-art methods in white matter tract segmentation across different datasets, spanning various age groups. The modular framework directly allows for the integration of additional embeddings in the future works.


## Usage

### Clone the repository
```
git clone https://github.com/anoushkrit/TractoEmbed
cd TractoEmbed/
```

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


## Dataset

The processed data utilized by Tractcloud can be downloaded from the following link:
https://github.com/SlicerDMRI/TractCloud/releases. 
The dataset includes 1 million streamlines, 800 clusters, and 800 outliers.

The directory structure for the dataset is as follows:

```bash
./  TractoEmbed
├── dataset
│   ├── train.pickle
│   ├── val.pickle
│   ├── test.pickle
```

## Model Training
 
### Patch Encoder Pretraining

To train the patch encoder(dVAE), simply run:
```
bash models/dVAE/train.sh <GPU_IDS>\
        --config models/dVAE/cfgs/dvae.yaml\
        --exp_name <name>
```
Replace <GPU_IDS> with the desired GPU IDs and <name> with the experiment name.

After training the patch encoder, update the dvae config path and model weight path in the ./train_test/train_multiembed.sh file.

### Streamline Encoder Pretraining

To extract streamline embeddings, use the pretrained DeepWMA model. The embeddings should be saved in the training, validation, and test pickle files under the key "cnn_embed".

For training the streamline encoder from scratch, refer to the DeepWMA repository: 
https://github.com/zhangfanmark/DeepWMA

### Cluster Encoder Pretraining
The cluster encoder is trained in conjunction with the multiembed classification layer, eliminating the need for pretraining.

### MultiEmbed Classification Training 
To train the multiembed layers, run the following commands.

```
$ cd train_test
$ sh train_multiembed.sh
```
Adjust the arguments in the train_multiembed.sh file as necessary.

### Testing

## Results

| **Data**                              | **Model: Type**           | **Acc (%)** | **F1 (%)** |
|---------------------------------------|---------------------------|-------------|------------|
| **Single Streamline**                 | DeepWMA (CNN)             | 90.29       | 88.12      |
|                                       | DCNN++ (CNN)              | 91.26       | 89.14      |
|                                       | PointNet (PCD)            | 91.36       | 89.12      |
|                                       | DGCNN (Graph)             | **91.85**   | **89.78**  |
| **Local PCD** (k = 20)                | TractCloud: PointNet      | 91.51       | 89.25      |
|                                       | TractCloud: DGCNN (Graph) | 91.91       | 90.03      |
|                                       | **TractoEmbed (ours)**    | **92.09**   | **90.07**  |
| **Hyperlocal PCD** (k = 5)            | TractCloud (PointNet)     | 91.12       | 88.66      |
|                                       | **TractoEmbed (ours)**    | **93.04**   | **91.38**  |
| **Local + Global Representation**     | TractCloud: PointNet      | **92.28**   | **90.36**  |
|                                       | TractCloud: DGCNN (Graph) | 91.99       | 90.10      |

### Citation
If you find this work useful, please cite

```bibtex
@inproceedings{goel2024tractoembed,
  title={TractoEmbed: Modular Multi-level Embedding framework for white matter tract segmentation},
  author={Goel, Anoushkrit and Singh, Bipanjit and Joshi, Ankita and Jha, Ranjeet Ranjan and Ahuja, Chirag and Nigam, Aditya and Bhavsar, Arnav},
  booktitle={International Conference on Pattern Recognition},
  pages={240--255},
  year={2024},
  organization={Springer}
}
```


## Contributors
This project is mainly developed and maintained by [Anoushkrit Goel](https://github.com/anoushkrit), [Bipanjit Singh](https://github.com/BipanjitGill). Issues and contributions are very welcome at any time. 
