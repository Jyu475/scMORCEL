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

## Datasets

Example datasets used in the article can be downloaded from https://doi.org/10.6084/m9.figshare.32780442.

## Usage

Required objects in h5ad or mtx file for running scMORCEL

1.ADT/ATAC count matrix

2.mRNA count matrix

3.True labels

4.Run scMORCEL by following the scMORCEL_run.ipynb notebook in the “Experiments” folder.

test_score, history = scMORCEL(
    test=test,
    reference=reference,
    label=label,
    processing_unit="cuda",
    max_epochs=100,
    patience=10,
    learning_rate=1e-3,
    verbose=True
)
