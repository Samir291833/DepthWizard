# DepthWizard

## SIH 2026 -- SIH26175

DepthWizard is an end-to-end software pipeline for **single-view height/elevation estimation from optical remote-sensing imagery and interactive 3D visualization**.

The system is being developed as a software solution for **SIH 2026 Problem Statement SIH26175**, with the long-term goal of transforming a single optical image into a spatially aligned elevation representation and an interactive 3D scene.

> **Project status: Core processing pipeline completed and validated. Web application development is the next stage.**

---

## Current Development Status

The core research and processing pipeline has been implemented and validated on the **ISPRS Potsdam** dataset.

### Completed

* Project environment and dependency setup
* GPU-enabled PyTorch inference
* ISPRS Potsdam dataset preparation and spatial organization
* RGB imagery, DSM and label alignment
* Train/validation/test spatial split
* Geospatial metadata verification and correction
* Depth Anything V2 Base integration
* RGB image preprocessing
* Monocular relative-depth/disparity inference
* CPU and CUDA inference paths
* Deterministic inference validation
* Relative DSM (rDSM) generation
* Reference DSM evaluation
* Calibration experiments
* Geospatial pixel-to-world coordinate transformations
* WGS84 coordinate conversion where applicable
* Spatial alignment and round-trip validation
* Regular-grid 3D mesh reconstruction
* glTF/GLB generation
* Texture generation and embedding
* Mesh-to-source-pixel mapping
* UV coordinate validation
* End-to-end pipeline validation
* CUDA regression validation against the previous CPU baseline

### Current Hardware Validation

* Python 3.13.3
* PyTorch 2.13.0 + CUDA 13.0
* NVIDIA RTX 3050 Laptop GPU -- 4 GB VRAM
* Hugging Face Transformers
* Depth Anything V2 Base
* Rasterio
* NumPy / SciPy / scikit-learn
* FastAPI dependencies prepared for the application layer

The current Depth Anything V2 Base inference pipeline has been validated on the RTX 3050 with GPU memory usage suitable for the available 4 GB VRAM.

---

## Core Processing Pipeline

```text
Optical RGB Remote-Sensing Image
              |
              v
        Input Validation
              |
              v
        Image Preprocessing
              |
              v
    Depth Anything V2 Base
              |
              v
       Relative Disparity
              |
              v
       Relative DSM (rDSM)
              |
              v
     Reference / Calibration
        (when applicable)
              |
              v
     Geospatial Alignment
              |
              v
      3D Mesh Reconstruction
              |
              v
       Textured GLB / glTF
```

The core pipeline currently produces a **relative surface representation** from monocular depth.

Raw monocular depth is **not assumed to be metric elevation in metres**. Metric calibration is treated separately and requires a suitable reference or calibration source.

---

## Geospatial Support

DepthWizard is being designed to support both:

### Non-georeferenced imagery

PNG/JPG imagery can be processed to produce:

* relative depth
* relative DSM/rDSM
* 3D reconstruction
* textured 3D visualization

### Georeferenced imagery

GeoTIFF imagery can preserve and use:

* CRS
* affine transform
* spatial resolution
* bounds
* projected coordinates
* geographic coordinate conversion where supported

The current geospatial pipeline has been tested using the **EPSG:32633 UTM** imagery in the Potsdam dataset.

---

## 3D Reconstruction

The reconstruction stage converts the generated relative surface into a regular-grid 3D mesh.

Current implementation supports:

* controlled mesh decimation
* source-pixel-to-mesh mapping
* UV generation
* RGB texture mapping
* GLB/glTF 2.0 output
* reconstruction metadata
* preservation of source spatial metadata

The browser-based interactive visualization layer is **not yet implemented** and is planned for the next development stage.

---

## Validation

The completed core pipeline has undergone validation covering:

* Input/data integrity
* Depth inference
* Determinism
* DSM/rDSM generation
* Calibration behavior
* Reference DSM accuracy
* Spatial/geospatial transformations
* Pixel-to-world round trips
* Mesh/source-pixel correspondence
* GLB structure
* Texture integrity
* End-to-end execution
* CUDA regression

The validation artifacts are retained locally under `outputs/` and are intentionally not committed to Git at this development stage.

---

## Dataset

Development datasets are stored locally and are **not committed to this repository**.

Primary development dataset:

* **ISPRS Potsdam**

The project uses the Potsdam RGB imagery, DSM, labels, nDSM-derived data and SRTM reference data during development and validation.

Large datasets and generated artifacts are excluded through `.gitignore`.

---

## Project Architecture

The current core processing modules are organized under:

```text
depthwizard/
|-- calibrate.py
|-- config.py
|-- diagnostics.py
|-- dsm.py
|-- estimator.py
|-- evaluate.py
|-- geo.py
|-- loader.py
|-- preprocess.py
|-- reconstruct.py
`-- __init__.py
```

Execution and diagnostic scripts:

```text
scripts/
|-- inspect_depth.py
|-- run_depth.py
`-- run_eval.py
```

---

## Next Development Stage

The core processing pipeline is now complete. Development will continue with the application layer.

Planned next stages include:

```text
Completed Core Pipeline
          |
          v
       FastAPI
       Backend
          |
          v
   API Integration
          |
          v
   Three.js Web Viewer
          |
          v
 Interactive 3D Scene
          |
          v
 Point / Pixel Query
          |
          v
Geographic Coordinate Display
          |
          v
 DSM / rDSM Value Inspection
          |
          v
 End-to-End Web Application
          |
          v
 Final Integration & Validation
```

The final application is intended to allow users to upload imagery, run the elevation-estimation pipeline, visualize the resulting 3D surface interactively, and inspect spatial information directly from the generated scene.

---

## Problem Statement

**SIH26175 -- DepthWizard: Single-View Height Estimation and 3D Flythrough**

Organization:

**Indian Space Research Organisation (ISRO)**

---

## Development Note

This repository is under active development.

The current milestone establishes a validated foundation for the DepthWizard application. The backend, interactive web interface, point-query functionality, and final application integration are still under development.
