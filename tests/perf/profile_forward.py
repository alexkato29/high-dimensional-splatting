import argparse
import torch
import time
from pathlib import Path
import sys
from typing import Any

from gaussian_splatting.rasterizer import rasterize

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


SCENARIOS: dict[str, dict[str, Any]] = {
    "small": {
        "num_gaussians": 10_000,
        "image_width": 640,
        "image_height": 480,
        "warmup_iters": 5,
        "profile_iters": 1,
        "description": "Quick test for development"
    },
    "medium": {
        "num_gaussians": 100_000,
        "image_width": 1920,
        "image_height": 1080,
        "warmup_iters": 10,
        "profile_iters": 1,
        "description": "Realistic workload (1080p, 100k gaussians)"
    },
    "large": {
        "num_gaussians": 500_000,
        "image_width": 1920,
        "image_height": 1080,
        "warmup_iters": 10,
        "profile_iters": 1,
        "description": "Large scene (1080p, 500k gaussians)"
    },
    "huge": {
        "num_gaussians": 1_000_000,
        "image_width": 3840,
        "image_height": 2160,
        "warmup_iters": 5,
        "profile_iters": 1,
        "description": "Stress test (4K, 1M gaussians)"
    },
}


def create_test_data(num_gaussians: int, image_width: int, image_height: int, device: str = 'cuda') -> dict[str, Any]:
    means3D: torch.Tensor = torch.randn(num_gaussians, 3, device=device) * 2.0
    scales: torch.Tensor = torch.randn(num_gaussians, 3, device=device) * 0.5 - 1.0
    quaternions: torch.Tensor = torch.randn(num_gaussians, 4, device=device)
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    opacities: torch.Tensor = torch.rand(num_gaussians, 1, device=device) * 0.8 + 0.2
    colors: torch.Tensor = torch.rand(num_gaussians, 3, device=device)
    world_to_cam: torch.Tensor = torch.eye(4, device=device)

    focal_x: float
    focal_y: float
    focal_x = focal_y = float(image_width)
    c_x: float = image_width / 2.0
    c_y: float = image_height / 2.0

    return {
        'means3D': means3D,
        'scales': scales,
        'quaternions': quaternions,
        'opacities': opacities,
        'colors': colors,
        'world_to_cam_matrix': world_to_cam,
        'focal_x': focal_x,
        'focal_y': focal_y,
        'c_x': c_x,
        'c_y': c_y,
        'image_width': image_width,
        'image_height': image_height,
    }


def profile_rasterizer(scenario_name: str = 'medium', verbose: bool = True) -> torch.Tensor:
    scenario: dict[str, Any] = SCENARIOS[scenario_name]

    if verbose:
        print(f"=== Profiling Scenario: {scenario_name} ===")
        print(f"Description: {scenario['description']}")
        print(f"Gaussians: {scenario['num_gaussians']:,}")
        print(f"Resolution: {scenario['image_width']}x{scenario['image_height']}")
        print(f"Warmup iterations: {scenario['warmup_iters']}")
        print(f"Profile iterations: {scenario['profile_iters']}\n")

    if verbose:
        print("Creating test data...")
    data: dict[str, Any] = create_test_data(
        scenario['num_gaussians'],
        scenario['image_width'],
        scenario['image_height']
    )

    if verbose:
        print(f"Warming up ({scenario['warmup_iters']} iterations)...")

    output: torch.Tensor = torch.empty()
    for _ in range(scenario['warmup_iters']):
        output = rasterize(**data)

    torch.cuda.synchronize()

    start: float = 0.0
    if verbose:
        print(f"Profiling ({scenario['profile_iters']} iterations)...")
        start: float = time.perf_counter()

    torch.cuda.synchronize()
    start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
    end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(scenario['profile_iters']):
        output = rasterize(**data)
    end_event.record()

    torch.cuda.synchronize()

    if verbose:
        wall_time: float = time.perf_counter() - start
        gpu_time: float = start_event.elapsed_time(end_event) / 1000.0
        avg_gpu_time: float = gpu_time / scenario['profile_iters'] * 1000

        print("\n=== Results ===")
        print(f"Total wall time: {wall_time:.3f}s")
        print(f"Total GPU time: {gpu_time:.3f}s")
        print(f"Average GPU time per frame: {avg_gpu_time:.2f}ms")
        print(f"Throughput: {scenario['profile_iters'] / gpu_time:.1f} FPS\n")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile Gaussian Splatting rasterizer')
    parser.add_argument(
        '--scenario',
        choices=list(SCENARIOS.keys()),
        default='medium',
        help='Test scenario to run'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output (useful when running under profilers)'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print()

    profile_rasterizer(args.scenario, verbose=not args.quiet)


if __name__ == '__main__':
    main()
