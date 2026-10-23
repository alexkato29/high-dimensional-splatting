# Efficient Gaussian Splatting in High Dimensions

A unique, efficient, and minimal implementation of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), built from scratch with custom CUDA kernels. We handle high dimensional feature vectors *without* compression or downsampling of dimensionality, motivated by [Feature3DGS](https://arxiv.org/abs/2312.03203). Feature images match an FP32 reference to within `1e-3`, the rounding error of the TF32 tensor cores used in the blend. Feature gradients match to FP32 rounding error, and are computed in full FP32.

![Inria 3DGS, gsplat and ours on three held out Truck views after 5,000 iterations](docs/truck_5k_grid.png)

## Overview

We implement the core 3DGS rasterization algorithm entirely from scratch, including:
- Tile-based gaussian projection, rendering, and gradient computation (in CUDA).
- PyTorch bindings to support python-hosted training and inference integration.
- A training pipeline with COLMAP dataset ingestion.

Of emphasis is the performance on high-dimensional features, handling `{32, 64, 128, 256, 512}` dimensional vectors with a natural extension to higher dimensions available. Unlike compression (proposed by Feature3DGS) or blending only a subset of gaussians per ray (proposed by [Quantile Rendering](https://arxiv.org/abs/2512.20927)), every contributing gaussian is blended at full dimensionality. The forward pass blends on TF32 tensor cores. Concurrent work by [Zhang et al. (NeurIPS 2026)](https://openreview.net/forum?id=PHnQZyfoZ1) describes a related tensor-core approach in its abstract, though their full paper is not yet available as of October 23rd, 2026. The backward pass uses a novel layout: each lane owns a slice of channels across its warp's pixels, so feature gradients accumulate within a lane with no cross-lane reduction. Performance is meaningfully improved over the official nerfstudio `gsplat` implementation, with outputs matching within TF32 rounding tolerance.

*Algorithmic writeup coming soon.*

## Performance

Against `gsplat` 1.6.0 at its fastest configuration, we compare our approach on an NVIDIA L4 GPU, with all times reported in milliseconds.

| Channels | Forward (ours) | Forward (gsplat) | Speedup | Backward (ours) | Backward (gsplat) | Speedup | Forward + backward |
| -------- | -------------- | ---------------- | ------- | --------------- | ----------------- | ------- | ------------------ |
| 32       | 4.13           | 7.44             | 1.80x   | 10.59           | 28.93             | 2.73x   | 2.47x              |
| 64       | 4.73           | 9.88             | 2.09x   | 12.53           | 47.59             | 3.80x   | 3.33x              |
| 128      | 6.67           | 22.90            | 3.43x   | 18.31           | 108.14            | 5.91x   | 5.25x              |
| 256      | 10.46          | 41.84            | 4.00x   | 31.87           | 229.95            | 7.22x   | 6.42x              |
| 512      | 18.55          | 82.99            | 4.47x   | 64.28           | 509.72            | 7.93x   | 7.16x              |

All use the Tanks and Temples Truck scene (~1.18M gaussians after 5,000 iterations, a single 979x546 view). Each time is the median of 50 runs and covers projection, binning and blending of the RGB image plus the feature image.

**Reproduce these results on the same hardware with:**
```bash
uv run --group bench python -m bench.harness path/to/scene path/to/model.pt --channels 32 64 128 256 512
```

*Note: PyTorch 2.10 (cu128), nvcc 12.9, last measured October 23rd, 2026.*

## Reading the Docs
Line by line notes of the novel kernels live in `.sidenotes/`. They're kept there to keep the source code readable and use [sidenotes](https://github.com/alexkato29/zed-sidenotes), a light format I wrote that anchors notes in a separate file to lines in the source.

To read them as inline comments, a side by side markdown, or any other format, you can point an AI agent at them:
```text
Look at .sidenotes/ to understand the format. Convert it to <inline comments | side by side markdown | whatever your preferred format is>.
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

## License

[Apache 2.0](LICENSE), applying to all versions of this repository, including commits made before the change from MIT. See [NOTICE](NOTICE).
