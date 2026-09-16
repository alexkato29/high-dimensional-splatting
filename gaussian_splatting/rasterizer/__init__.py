from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from gaussian_splatting.rasterizer.projection import project_gaussians  # noqa: F401

_rasterizer_module: Any | None = None


def _get_rasterizer() -> Any:
	global _rasterizer_module

	if _rasterizer_module is not None:
		return _rasterizer_module

	current_dir: Path = Path(__file__).parent

	sources: list[str] = [
		str(current_dir / "forward.cu"),
		str(current_dir / "backward.cu"),
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
		background: torch.Tensor,
		image_width: int,
		image_height: int
	) -> torch.Tensor:
		image, final_transmittance, n_contrib, values_sorted, tile_ranges, conic, color_opacity = (
			_get_rasterizer().rasterize(
				means2D.contiguous(),
				depths.contiguous(),
				radii.contiguous(),
				conics.contiguous(),
				colors.contiguous(),
				opacities.contiguous(),
				background.contiguous(),
				int(image_width),
				int(image_height)
			)
		)
		ctx.save_for_backward(means2D, conic, color_opacity, values_sorted, tile_ranges,
							  final_transmittance, n_contrib, background)
		ctx.image_size = (int(image_width), int(image_height))
		return image

	@staticmethod
	def backward(ctx, grad_image):
		(means2D, conic, color_opacity, values_sorted, tile_ranges,
		 final_transmittance, n_contrib, background) = ctx.saved_tensors
		width, height = ctx.image_size
		grad_means2D, grad_conics, grad_colors, grad_opacities = _get_rasterizer().rasterize_backward(
			grad_image.contiguous(),
			means2D,
			conic,
			color_opacity,
			values_sorted,
			tile_ranges,
			final_transmittance,
			n_contrib,
			background,
			width,
			height
		)
		return grad_means2D, None, None, grad_conics, grad_colors, grad_opacities, None, None, None


def rasterize(
	means2D: torch.Tensor,
	depths: torch.Tensor,
	radii: torch.Tensor,
	conics: torch.Tensor,
	colors: torch.Tensor,
	opacities: torch.Tensor,
	image_width: int,
	image_height: int,
	background: torch.Tensor | None = None
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
		background: [3] RGB shown wherever the gaussians do not cover, black by default

	The first four come from project_gaussians.

	Returns:
		[H, W, 3] rendered image
	"""
	n = means2D.shape[0]
	assert means2D.shape == (n, 2)
	assert depths.shape == (n,) and radii.shape == (n,)
	assert conics.shape == (n, 3) and colors.shape == (n, 3)
	assert opacities.shape == (n, 1)
	if background is None:
		background = torch.zeros(3, device=means2D.device, dtype=means2D.dtype)
	assert background.shape == (3,)

	return RasterizeGaussians.apply(means2D, depths, radii, conics, colors, opacities, background,
									image_width, image_height)
