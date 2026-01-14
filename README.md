# Spatial GNN-based AMP Classifier (SGAC)

This project implements a novel **Spatial GNN-based AMP Classifier** (SGAC) designed to accurately distinguish between Antimicrobial Peptides (AMPs) and non-Antimicrobial Peptides (non-AMPs). The method leverages graph neural networks (GNNs) to incorporate the three-dimensional spatial structure of peptides, addressing limitations in traditional sequence-based classifiers and mitigating class imbalance through weight-enhanced contrastive learning and pseudo-label distillation.

The paper has been accepted by **Briefings in Bioinformatics**, 2025.

## Setup Instructions

Follow these steps to set up the environment, download the data, and run the code:

### 1. Create a Conda Environment

Create the required environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate py38
```

### 2. Download Data

Download the dataset from the provided [Google Drive link](https://drive.google.com/file/d/1znoNWTuX1rwRN2_OOz6KTYxcnSRTJc43/view?usp=sharing) and place it in the `/data` directory.

### 3. Run the Code

Execute the `main.py` script to train and evaluate the model:

```
python main.py --gpu 0 --gnn gin --emb_dim 256   
```

#### Citation

```bibtex
@article{wang2024sgac,
  title={SGAC: A Graph Neural Network Framework for Imbalanced and Structure-Aware AMP Classification},
  author={Wang, Yingxu and Liang, Victor and Yin, Nan and Liu, Siwei and Segal, Eran},
  journal={arXiv preprint arXiv:2412.16276},
  year={2024}
}
```

