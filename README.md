# Single-Image Dehazing with Shifted-Window Self-Attention U-Net
## Overview

This project implements a **single-image dehazing** pipeline in PyTorch, leveraging a **shifted-window self-attention U-Net** (inspired by Swin Transformers). It processes paired hazy/clear images from the RESIDE SOTS dataset to remove atmospheric haze and restore clear, high-fidelity outputs.

## Features

- **Data Loading & Augmentation**  
  Paired hazy/ground-truth reader with Albumentations transforms (flips, crops, color jitter, blur).
- **Swin U-Net Architecture**  
  Encoder–decoder network with window-based multi-head self-attention, shifted windows, and depthwise convolutions.
- **Loss & Optimization**  
  L1 reconstruction loss, Adam optimizer, and CosineAnnealingWarmRestarts scheduler.
- **Quantitative Evaluation**  
  PSNR, SSIM, and MS-SSIM metrics for objective image quality assessment.
- **Visualization**  
  Automated side-by-side comparison of hazy input, ground truth, and dehazed output.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- albumentations
- timm
- scikit-image
- pytorch-msssim
- tqdm

---
