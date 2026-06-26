# scMORCEL

## Introduction

scMORCEL, an **open-set representation learning framework** for novel rare cell detection in single-cell multi-omics data. Rather than treating representation learning and novelty detection as independent procedures, scMORCEL explicitly learns a latent space optimized for open-set recognition, where known cell populations form compact and well-separated manifolds while previously unseen cellular states remain distinguishable from them. Specifically, a multi-head gene attention module first enhances informative molecular signals across heterogeneous modalities, after which supervised contrastive learning jointly optimizes representation learning and cell-type discrimination to construct an open-set-aware latent space. Building upon this representation, scMORCEL jointly models distributional deviation from known cellular manifolds and predictive uncertainty of the classifier as complementary indicators of novelty. These two signals are robustly calibrated onto a common statistical scale and fused into a unified open-set confidence score for identifying novel rare cells.

The framework integrates three major components:

1. We formulate novel rare cell detection in single-cell multi-omics as an open-set recognition problem and propose scMORCEL, a unified framework that integrates open-set representation learning with novelty detection for discovering previously unseen cellular populations;
2. We develop an open-set representation learning strategy that combines multi-head gene attention with supervised contrastive optimization to construct compact manifolds for known cell populations while preserving the separability of unseen cellular states in the latent space;
3. We introduce a robust novelty estimation framework that jointly models distributional deviation and predictive uncertainty through statistically calibrated score fusion, effectively addressing the complementary failure modes of geometry-based and confidence-based open-set detection across diverse single-cell multi-omics datasets.

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
