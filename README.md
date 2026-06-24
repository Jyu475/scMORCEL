# scMORCEL

## Introduction

**scMORCEL** is a deep learning framework for **novel rare cell detection in single-cell multi-omics data**. It is designed to identify potential Novel Rare Cells from a query set by learning discriminative cellular representations from labeled reference cells.

The framework integrates three major components:

1. **Multi-head gene attention** for adaptive feature enhancement;
2. **Supervised contrastive learning** for constructing a compact and separable latent representation space;
3. **Mahalanobis--Energy integrated scoring** for robust rare cell detection by jointly modeling distributional deviation and predictive uncertainty.

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- Pandas
- scikit-learn
- SciPy
- Scanpy, optional, for preprocessing single-cell data

##Installation

1. Clone this repository

```bash
git clone https://github.com/Jyu475/scMORCEL
cd scMORCEL
```

### 2. Create a conda environment

```bash
conda create -n scmorcel python=3.9
conda activate scmorcel
```
## Run core

1.read data

DATA_FORMAT = 'mtx'   # ← 在此切换: 'mtx' 或 'h5ad'
DATASET_ID  = 9       # ← 在此切换数据集编号

MTX_DATASETS = {
    # ID: (数据目录, rna_mtx, rna_features, rna_barcodes, adt_csv, label_csv)
    7: (
        "Experiments/7_GSE194122_s3d6/",
        "GSE194122_s3d6_matrix.mtx",
        "GSE194122_s3d6_features.tsv",
        "GSE194122_s3d6_barcodes.tsv",
        "GSE194122_s3d6_ADT.csv",
        "GSE194122_s3d6_label.csv",
        None   # label_index_replace: None 表示不替换
    ),
}

H5AD_DATASETS = {
    # ID: (数据目录, rna_h5ad, adt_h5ad, label_csv)
    1: (
        "Experiments/1_10X1kpbmc/",
        "10X1kpbmc_rna.h5ad",
        "10X1kpbmc_adt.h5ad",
        "10X1kpbmc_label.csv"
    ),
}

2.Run with scMORCEL

score=scMORCEL(
    test=None,
    reference=None,
    label=None,
    processing_unit="cuda",
    max_epochs=100,
    patience=10,
    model_type="attention",
    attention_heads=4,
    use_validation=True,
    validation_split=0.1,
    learning_rate=1e-3,
    score_function="mahalanobis_energy",
    mahal_energy_alpha=0.5,
    energy_temperature=1.0,
    use_contrastive=True,
    contrastive_weight=0.1,
    contrastive_temperature=0.5,
    contrastive_type="contrastive",
    verbose=True
)
