# 3D Gaussian Splatting

A minimal implementation of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) built from the ground up with custom CUDA kernels.

## Overview

This project implements the core 3DGS rasterization algorithm entirely from scratch, including:
- Custom CUDA kernels for tile-based gaussian projection and rendering
- Differentiable rasterizer with PyTorch bindings
- Training pipeline with COLMAP dataset integration
- Comprehensive unit tests for mathematical correctness

## Technical Implementation

### Custom CUDA Rasterizer
The rasterizer (`src/gaussian_splatting/rasterizer/forward.cu`) implements:

1. **Gaussian Projection**: Projects 3D gaussians to 2D screen space
   - Pinhole camera projection with covariance matrix transformation
   - Quaternion-based rotation handling
   - Frustum and depth culling

2. **Tile-Based Rendering**: Optimizes rendering through spatial partitioning
   - Computes tile coverage per gaussian
   - Duplicates gaussians across tiles using prefix sum
   - Radix sorts by (tile_id, depth) for front-to-back rendering

3. **Alpha Blending**: Renders gaussians with proper compositing
   - Evaluates 2D gaussian weights per pixel
   - Alpha compositing with early ray termination

### Architecture
```
src/
├── gaussian_splatting/
│   ├── rasterizer/
│   │   ├── forward.cu          # CUDA kernels
│   │   ├── bindings.cpp        # PyTorch bindings
│   │   └── helpers.h           # GPU math utilities
│   ├── utils/
│   │   ├── dataset.py          # COLMAP data loading
│   │   └── gaussian.py         # Gaussian model & parameters
│   ├── config.py               # Training hyperparameters
│   └── main.py                 # Training loop
```

## Usage

```bash
poetry install
python src/gaussian_splatting/main.py path/to/colmap/scene
```

**Dataset Structure:**
```
scene/
├── sparse/0/  # COLMAP reconstruction
└── images/    # Training images
```

You can download pre-processed COLMAP datasets from [here](https://demuc.de/colmap/datasets/).
