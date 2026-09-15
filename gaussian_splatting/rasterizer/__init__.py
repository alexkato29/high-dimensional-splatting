from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.cpp_extension import load

from gaussian_splatting.rasterizer.projection import project_gaussians  # noqa: F401

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
			"-std=c++17",
			"-lineinfo"
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
		means2D: torch.Tensor,
		depths: torch.Tensor,
		radii: torch.Tensor,
		conics: torch.Tensor,
		colors: torch.Tensor,
		opacities: torch.Tensor,
		image_width: int,
		image_height: int
	) -> torch.Tensor:
		return _get_rasterizer().rasterize(
			means2D.contiguous(),
			depths.contiguous(),
			radii.contiguous(),
			conics.contiguous(),
			colors.contiguous(),
			opacities.contiguous(),
			int(image_width),
			int(image_height)
		)

	@staticmethod
	def backward(ctx, grad_output):
		return None, None, None, None, None, None, None, None


def rasterize(
	means2D: torch.Tensor,
	depths: torch.Tensor,
	radii: torch.Tensor,
	conics: torch.Tensor,
	colors: torch.Tensor,
	opacities: torch.Tensor,
	image_width: int,
	image_height: int
) -> torch.Tensor:
	"""
	Alpha-blend projected gaussians into an image, front to back, on 16x16 pixel tiles.

	Args:
		means2D: [N, 2] centers in pixels
		depths: [N] camera-space depths, used to sort
		radii: [N] footprint radii in pixels, 0 for culled gaussians
		conics: [N, 3] inverse 2D covariances as (xx, xy, yy)
		colors: [N, 3] RGB
		opacities: [N, 1] in (0, 1)
		image_width, image_height: output size in pixels

	The first four come from project_gaussians.

	Returns:
		[H, W, 3] rendered image
	"""
	n = means2D.shape[0]
	assert means2D.shape == (n, 2)
	assert depths.shape == (n,) and radii.shape == (n,)
	assert conics.shape == (n, 3) and colors.shape == (n, 3)
	assert opacities.shape == (n, 1)

	return RasterizeGaussians.apply(means2D, depths, radii, conics, colors, opacities, image_width, image_height)
