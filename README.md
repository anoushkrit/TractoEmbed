# TractoEmbed
TractoEmbed: Multi-level embeddings for Tract Segmentation

Keywords: **Tract Segmentation** · **PointCloud** · **3D Computer Vision** ·
**Tractography** · **Diffusion MRI**

White matter tract segmentation is a crucial task for studying brain structural connectivity and neurosurgical planning. However, segmentation remains challenging due to issues like class imbalance between major and minor tracts, structural similarity, subject variability, and symmetric streamlines between hemispheres etc. To address these challenges

We propose **TractoEmbed**, a modular multi-level embedding framework that encodes localized representations through learning task and representation specific encoders. TractoEmbed introduces a novel hierarchical streamline data representation that captures maximum spatial information at each level, including individual streamlines, clusters and patches. Experiments show that TractoEmbed clearly outperforms state-of-the-art methods in white matter tract segmentation across different datasets, spanning various age groups. The modular framework directly allows for the integration of additional embeddings in the future works.
## __init__


## Usage

## Model

### Patch Encoder Pretraining

### Streamline Encoder Pretraining

### Training 

#### Cluster Encoder Training 

### MECL

### Testing 


### Dataset
If you find this work useful, please cite
```bibtex

```
