import os
import torch
from typing import Any, Optional
from torch.utils.cpp_extension import load
from pathlib import Path

_rasterizer_module: Optional[Any] = None


def _get_rasterizer() -> Any:
	global _rasterizer_module

	if _rasterizer_module is not None:
		return _rasterizer_module

	current_dir: Path = Path(__file__).parent

	sources: list[str] = [
		str(current_dir / "forward.cu"),
		str(current_dir / "bindings.cpp")
	]

	print("Compiling Gaussian Rasterizer CUDA extension (this may take a minute)...")

	_rasterizer_module = load(
		name="gaussian_rasterizer",
		sources=sources,
		extra_cuda_cflags=[
			"-O3",
			"--use_fast_math",
			"-std=c++17"
		],
		extra_cflags=["-O3", "-std=c++17"],
		verbose=True
	)

	print("CUDA extension compiled successfully")
	return _rasterizer_module


class RasterizeGaussians(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx,
		means3D: torch.Tensor,
		scales: torch.Tensor,
		quaternions: torch.Tensor,
		opacities: torch.Tensor,
		colors: torch.Tensor,
		world_to_cam_matrix: torch.Tensor,
		focal_x: float,
		focal_y: float,
		c_x: float,
		c_y: float,
		image_width: int,
		image_height: int
	) -> torch.Tensor:
		_C = _get_rasterizer()

		output = _C.rasterize(
			means3D.contiguous(),
			scales.contiguous(),
			quaternions.contiguous(),
			opacities.contiguous(),
			colors.contiguous(),
			world_to_cam_matrix.contiguous(),
			float(focal_x),
			float(focal_y),
			float(c_x),
			float(c_y),
			int(image_width),
			int(image_height)
		)

		return output

	@staticmethod
	def backward(ctx, grad_output):
		return None, None, None, None, None, None, None, None, None, None, None, None


def rasterize(
	means3D: torch.Tensor,
	scales: torch.Tensor,
	quaternions: torch.Tensor,
	opacities: torch.Tensor,
	colors: torch.Tensor,
	world_to_cam_matrix: torch.Tensor,
	focal_x: float,
	focal_y: float,
	c_x: float,
	c_y: float,
	image_width: int,
	image_height: int
) -> torch.Tensor:
	"""
	Rasterize 3D Gaussians to a 2D image.

	Args:
		means3D: [N, 3] Gaussian centers in world space
		scales: [N, 3] Gaussian scales (log space)
		quaternions: [N, 4] Gaussian rotations as quaternions [w, x, y, z]
		opacities: [N, 1] Gaussian opacities (0-1)
		colors: [N, 3] Gaussian RGB colors (0-1)
		world_to_cam_matrix: [4, 4] World-to-camera transformation matrix
		focal_x: Focal length in pixels (x-axis)
		focal_y: Focal length in pixels (y-axis)
		c_x: Principal point x-coordinate
		c_y: Principal point y-coordinate
		image_width: Output image width
		image_height: Output image height

	Returns:
		[H, W, 3] Rendered image (RGB, 0-1 range)
	"""
	assert means3D.is_cuda, "means3D must be on CUDA"
	assert means3D.ndim == 2 and means3D.shape[1] == 3
	assert scales.shape == means3D.shape
	assert quaternions.shape == (means3D.shape[0], 4)
	assert opacities.shape == (means3D.shape[0], 1)
	assert colors.shape == means3D.shape
	assert world_to_cam_matrix.shape == (4, 4)

	return RasterizeGaussians.apply(
		means3D,
		scales,
		quaternions,
		opacities,
		colors,
		world_to_cam_matrix,
		focal_x,
		focal_y,
		c_x,
		c_y,
		image_width,
		image_height
	)
