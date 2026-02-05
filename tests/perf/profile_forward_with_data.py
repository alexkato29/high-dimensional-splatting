import argparse
import torch
import time
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gaussian_splatting.rasterizer import rasterize
from gaussian_splatting.utils.dataset import ColmapDataset
from gaussian_splatting.utils.gaussian import GaussianModel


def profile_with_real_data(
    data_path: str,
    num_views: int,
    verbose: bool,
    seed: int | None
) -> List[torch.Tensor]:
    if verbose:
        print(f"=== Profiling with Real COLMAP Data ===")
        print(f"Data path: {data_path}")
        print(f"Number of views to render: {num_views}\n")

    if verbose:
        print("Loading COLMAP dataset...")
    dataset: ColmapDataset = ColmapDataset(data_path)

    if verbose:
        print(f"Loaded {len(dataset.images)} images")
        print(f"Point cloud: {dataset.point_cloud.points.shape[0]:,} points")
        print(f"Image resolution: {dataset.images[0].width}x{dataset.images[0].height}\n")

    if verbose:
        print("Initializing Gaussian model...")
    model: GaussianModel = GaussianModel(dataset.point_cloud)

    if verbose:
        print(f"Initialized {model.xyz.shape[0]:,} Gaussians\n")

    if seed: 
        random.seed(seed)
    selected_views: list = random.sample(dataset.images, min(num_views, len(dataset.images)))

    if verbose:
        print(f"Selected views: {[view.uid for view in selected_views]}\n")

    if verbose:
        print("Warming up...")

    for view in selected_views[:1]:
        if verbose:
            print(f"Warming up with view {view.uid}...")
        world_to_cam: torch.Tensor = view.world_to_camera_matrix
        output: torch.Tensor = rasterize(
            model.xyz,
            model.scales,
            model.quaternions,
            model.opacities,
            model.rgb,
            world_to_cam,
            view.focal_x,
            view.focal_y,
            view.c_x,
            view.c_y,
            view.width,
            view.height
        )

    torch.cuda.synchronize()

    if verbose:
        print("Profiling...")

    start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
    end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

    outputs: List[torch.Tensor] = []
    start_event.record()

    for i, view in enumerate(selected_views):
        if verbose:
            print(f"Rendering view {i+1}/{len(selected_views)}: image {view.uid} ({view.width}x{view.height})...")
        world_to_cam: torch.Tensor = view.world_to_camera_matrix
        output: torch.Tensor = rasterize(
            model.xyz,
            model.scales,
            model.quaternions,
            model.opacities,
            model.rgb,
            world_to_cam,
            view.focal_x,
            view.focal_y,
            view.c_x,
            view.c_y,
            view.width,
            view.height
        )
        outputs.append(output)

    end_event.record()
    torch.cuda.synchronize()

    if verbose:
        gpu_time: float = start_event.elapsed_time(end_event) / 1000.0
        avg_time: float = gpu_time / len(selected_views) * 1000

        print("\n=== Results ===")
        print(f"Total GPU time: {gpu_time:.3f}s")
        print(f"Average time per view: {avg_time:.2f}ms")
        print(f"Throughput: {len(selected_views) / gpu_time:.1f} FPS")
        print(f"\nOutput shapes: {[o.shape for o in outputs]}")
        print(f"Output ranges: {[(o.min().item(), o.max().item()) for o in outputs]}")

    return outputs


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Profile rasterizer with real COLMAP data')
    parser.add_argument(
        '--data-path',
        type=str,
        default='../../data_1080p',
        help='Path to COLMAP data directory (contains sparse/ and images/)'
    )
    parser.add_argument(
        '--num-views',
        type=int,
        default=3,
        help='Number of random views to render'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )

    args: argparse.Namespace = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!", file=sys.stderr)
        sys.exit(1)

    data_path: Path = Path(args.data_path).resolve()
    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print()

    profile_with_real_data(
        str(data_path), 
        args.num_views, 
        not args.quiet, 
        args.seed
    )


if __name__ == '__main__':
    main()
