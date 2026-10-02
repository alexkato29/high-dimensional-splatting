from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from gaussian_splatting.rasterizer.projection import project_gaussians  # noqa: F401

_rasterizer_module: Any | None = None


def _get_rasterizer() -> Any:
	"""Compiles the CUDA extension on first use and caches the loaded module.

	Returns:
		The imported extension module exposing rasterize and rasterize_backward.
	"""
	global _rasterizer_module

	if _rasterizer_module is not None:
		return _rasterizer_module

	current_dir = Path(__file__).parent

	sources = [
		str(current_dir / "forward.cu"),
		str(current_dir / "backward.cu"),
		str(current_dir / "features.cu"),
		str(current_dir / "sh.cu"),
		str(current_dir / "projection.cu"),
		str(current_dir / "ssim.cu"),
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
	"""Autograd node wrapping the CUDA tile binning and alpha blend."""

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
		features: torch.Tensor | None,
		image_width: int,
		image_height: int
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		"""Renders the image and saves what the backward pass needs to replay the blend.

		Args:
			ctx: Autograd context the saved tensors are stashed on.
			means2D: [N, 2] centers in pixels.
			depths: [N] camera space depths, used to sort.
			radii: [N] footprint radii in pixels, 0 for culled gaussians.
			conics: [N, 3] inverse 2D covariances as (xx, xy, yy).
			colors: [N, 3] RGB.
			opacities: [N, 1] in (0, 1).
			background: [3] RGB shown wherever the gaussians do not cover.
			features: [N, D] channels blended over a zero background, or None.
			image_width: Output width in pixels.
			image_height: Output height in pixels.

		Returns:
			[H, W, 3] rendered image, followed by the [H, W, D] feature image when features are given.
		"""
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
		ctx.image_size = (int(image_width), int(image_height))
		if features is None:
			ctx.save_for_backward(means2D, conic, color_opacity, values_sorted, tile_ranges,
								  final_transmittance, n_contrib, background, None, None, None)
			return image
		features = features.contiguous()
		feature_planes, feature_transmittance, feature_contrib = _get_rasterizer().rasterize_features(
			features, means2D.contiguous(), conic, color_opacity, values_sorted, tile_ranges,
			int(image_width), int(image_height))
		ctx.save_for_backward(means2D, conic, color_opacity, values_sorted, tile_ranges,
							  final_transmittance, n_contrib, background, features, feature_transmittance,
							  feature_contrib)
		return image, feature_planes.permute(1, 2, 0)

	@staticmethod
	def backward(ctx, grad_image: torch.Tensor, grad_features: torch.Tensor | None = None) -> tuple[torch.Tensor | None, ...]:
		"""Walks each pixel's gaussian list back to front to recover the per gaussian gradients.

		Args:
			ctx: Autograd context holding the tensors saved during the forward pass.
			grad_image: [H, W, 3] gradient of the loss with respect to the rendered image.
			grad_features: [H, W, D] gradient with respect to the feature image, when features were given.

		Returns:
			One gradient per forward argument, and None for the arguments that are not
			differentiable, namely depths, radii, background and the image size.
		"""
		(means2D, conic, color_opacity, values_sorted, tile_ranges, final_transmittance, n_contrib, background,
		 features, feature_transmittance, feature_contrib) = ctx.saved_tensors
		width, height = ctx.image_size
		with torch.cuda.nvtx.range("rasterize_backward"):
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
		grad_feats = None
		# Frozen features still shape the geometry gradients, so any of means2D, conics, opacities or
		# features needing a gradient runs the feature backward.
		if features is not None and any(ctx.needs_input_grad[i] for i in (0, 3, 5, 7)):
			with torch.cuda.nvtx.range("rasterize_features_backward"):
				grad_feats, feat_means2D, feat_conics, feat_opacities = (
					_get_rasterizer().rasterize_features_backward(
						grad_features.permute(2, 0, 1).contiguous(), features, means2D, conic, color_opacity,
						values_sorted, tile_ranges, feature_transmittance, feature_contrib, width, height))
			# Both images depend on the same geometry, so their gradients add up.
			grad_means2D += feat_means2D
			grad_conics += feat_conics
			grad_opacities += feat_opacities
			if not ctx.needs_input_grad[7]:
				grad_feats = None
		return grad_means2D, None, None, grad_conics, grad_colors, grad_opacities, None, grad_feats, None, None


def rasterize(
	means2D: torch.Tensor,
	depths: torch.Tensor,
	radii: torch.Tensor,
	conics: torch.Tensor,
	colors: torch.Tensor,
	opacities: torch.Tensor,
	image_width: int,
	image_height: int,
	background: torch.Tensor | None = None,
	features: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
	"""Alpha blends projected gaussians into an image, front to back, on 16x16 pixel tiles.

	The first four arguments come from project_gaussians.

	Args:
		means2D: [N, 2] centers in pixels.
		depths: [N] camera space depths, used to sort.
		radii: [N] footprint radii in pixels, 0 for culled gaussians.
		conics: [N, 3] inverse 2D covariances as (xx, xy, yy).
		colors: [N, 3] RGB.
		opacities: [N, 1] in (0, 1).
		image_width: Output width in pixels.
		image_height: Output height in pixels.
		background: [3] RGB shown wherever the gaussians do not cover, black by default.
		features: [N, D] channels blended with the same weights over a zero background, with D one of
			32, 64 or 128. None renders RGB only.

	Returns:
		[H, W, 3] rendered image, or (image, [H, W, D] feature image) when features are given.
	"""
	n = means2D.shape[0]
	assert means2D.shape == (n, 2)
	assert depths.shape == (n,) and radii.shape == (n,)
	assert conics.shape == (n, 3) and colors.shape == (n, 3)
	assert opacities.shape == (n, 1)
	if background is None:
		background = torch.zeros(3, device=means2D.device, dtype=means2D.dtype)
	assert background.shape == (3,)
	assert features is None or (features.ndim == 2 and features.shape[0] == n)

	return RasterizeGaussians.apply(means2D, depths, radii, conics, colors, opacities, background, features,
									image_width, image_height)
