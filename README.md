# DepthWizard

## SIH 2026 — SIH26175

DepthWizard is an end-to-end software pipeline for single-view height estimation from optical remote-sensing imagery and interactive 3D terrain visualization.

## Current Development Status

Project initialization and environment validation completed.

### Validated

- Python 3.13.3
- PyTorch with CUDA
- NVIDIA RTX 3050 4GB Laptop GPU
- Depth Anything V2 Base
- Hugging Face Transformers inference
- Core image and geospatial processing dependencies

## Planned Pipeline

RGB Remote-Sensing Image
        ↓
Preprocessing
        ↓
Depth Anything V2 Base
        ↓
Relative Depth
        ↓
Metric Scale Calibration
        ↓
DSM Generation
        ↓
DSM Validation
        ↓
3D Terrain Reconstruction
        ↓
Interactive 3D Flythrough

## Dataset

Development datasets are stored locally and are not committed to this repository.

Primary development dataset:
- ISPRS Potsdam

## Problem Statement

SIH26175 — DepthWizard: Single-View Height Estimation and 3D Flythrough

Organization:
Indian Space Research Organisation (ISRO)