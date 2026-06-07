# Multimodal Transformers for Land-Cover Distribution Prediction

**DI725 — Transformers and Attention-Based Deep Networks**  
Middle East Technical University, Graduate School of Informatics  
Ilayda Ergür

## Overview

This project investigates multimodal transformer architectures for predicting land-cover class composition from satellite imagery and machine-generated text captions. Given a true-color satellite image and its textual description, the task is to predict the percentage composition of seven land-cover classes (tree cover, shrubland, grassland, cropland, built-up area, barren land, water) as a multi-output regression problem.

**Research Question:** To what extent can multimodal transformer architectures leverage semantic textual descriptions (without quantitative information) to improve prediction of land-cover class distributions from segmentation masks?

## Results Summary

| Model | Caption | Test MAE (%) | Test RMSE (%) |
|-------|---------|-------------|--------------|
| ViT only (baseline) | — | 2.73 | 6.05 |
| Text only | vision_qwen3 | 6.34 | 13.17 |
| MM concat + shuffled | shuffled | 2.91 | 6.23 |
| MM concat | vision_qwen3 | 2.87 | 6.19 |
| MM concat | hybrid_gemma | 1.93 | 3.87 |
| MM concat | hybrid_qwen | 1.83 | 3.60 |
| MM concat | text_qwen3 | 1.79 | 3.49 |
| **MM cross-attention** | **text_qwen3** | **1.75** | **3.25** |

## Repository Structure

```
├── notebooks/
│   ├── DI725_phase1.ipynb          # Phase 1: proof of concept
│   ├── DI725_phase2_final.ipynb    # Phase 2: benchmarking (7 experiments)
│   └── DI725_phase3_final.ipynb    # Phase 3: ablation + cross-attention fusion
├── reports/
│   ├── DI725_phase1_report_updated.pdf
│   ├── DI725_phase2_report.pdf
│   └── DI725_phase3_report.pdf
└── requirements.txt
```

## Models

- **ViT Baseline** — ViT-Base/16 (ImageNet-21k) CLS token → MLP head
- **Text-Only** — DistilBERT-base-uncased CLS token → MLP head
- **Concatenation Fusion** — ViT + DistilBERT CLS tokens concatenated → MLP head
- **Cross-Attention Fusion** — ViT patch tokens attend to DistilBERT token sequence via MultiheadAttention → MLP head

## Setup

```bash
pip install -r requirements.txt
```

The dataset is not included in this repository. Place the dataset folder at the path configured in the notebook (`DATA_ROOT`). All notebooks are designed to run on Google Colab with Google Drive mounted.

## Reproducibility

- Random seed: 10 (applied to Python, NumPy, PyTorch)
- Differential learning rates: pretrained encoders at `1e-5`, heads at `1e-4`
- All experiments tracked with [Weights & Biases](https://wandb.ai/ilaydaergur/DI725-land-cover-phase2)
- Phase 1: 7 epochs — Phase 2 & 3: 10 epochs, batch size 8
