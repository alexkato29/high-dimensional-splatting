# Gaussian Splatting in High Dimensions

A unique, efficient, and minimal implementation of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), built from scratch with custom CUDA kernels. We exactly handle high dimensional feature vectors *without* compression or downsampling of dimensionality, motivated by [Feature3DGS](https://arxiv.org/abs/2312.03203).

![Inria 3DGS, gsplat and ours on three held out Truck views after 5,000 iterations](docs/truck_5k_grid.png)

## Overview

We implement the core 3DGS rasterization algorithm entirely from scratch, including:
- Tile-based gaussian projection, rendering, and gradient computation (in CUDA).
- PyTorch bindings to support python-hosted training and inference integration.
- A training pipeline with handled COLMAP dataset ingestion and integration.

Of emphasis is the performance on high-dimensional features, where we implement a novel algorithm to handle `{32, 64, 128, 256, 512}` dimensional vectors, with a natural extension to higher dimensions available. Unlike compression (proposed by Feature3DGS) or downsampling (proposed by [Quantile Rendering](https://arxiv.org/abs/2512.20927)) techniques, this algorithm is exact in its computation. Performance is meaningfully improved over the official nerfstudio `gsplat` implementation with output parity achieved.

A detailed explanation of the algorithm and motivation are available in my blog post.

## Performance

Against `gsplat` 1.6.0 at its fastest configuration, we compare our approach on an NVIDIA L4 GPU, with all times reported in milliseconds.

| Channels | Forward (ours) | Forward (gsplat) | Speedup | Backward (ours) | Backward (gsplat) | Speedup | Forward + backward |
| -------- | -------------- | ---------------- | ------- | --------------- | ----------------- | ------- | ------------------ |
| 32       | 4.35           | 7.57             | 1.74x   | 11.27           | 29.35             | 2.60x   | 2.36x              |
| 64       | 5.39           | 10.05            | 1.86x   | 13.51           | 49.00             | 3.63x   | 3.12x              |
| 128      | 7.39           | 23.09            | 3.12x   | 19.20           | 111.42            | 5.80x   | 5.06x              |
| 256      | 15.04          | 42.34            | 2.82x   | 33.61           | 236.98            | 7.05x   | 5.74x              |
| 512      | 32.64          | 85.18            | 2.61x   | 66.88           | 522.44            | 7.81x   | 6.11x              |

All use the Tanks and Temples Truck scene (~1.18M gaussians after 5,000 iterations, a single 979x546 view). Each time is the median of 50 runs and covers projection, binning and blending of the RGB image plus the feature image.

Reproduce these results with:
```bash
uv run --group bench python -m bench.harness path/to/scene path/to/model.pt --channels 32 64 128 256 512
```

## Usage

```bash
uv sync
uv run python -m gaussian_splatting.main path/to/colmap/scene
```

**Dataset Structure:**
```
scene/
├── sparse/0/  # COLMAP reconstruction
└── images/    # Training images
```

You can download pre-processed COLMAP datasets from [here](https://demuc.de/colmap/datasets/).
