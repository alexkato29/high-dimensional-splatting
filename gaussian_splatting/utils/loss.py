from functools import lru_cache

import torch
import torch.nn.functional as F

from gaussian_splatting.rasterizer import _get_rasterizer

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def l1_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	"""Mean absolute error between a render and its ground truth.

	Args:
		rendered: [H, W, 3] rendered image in [0, 1].
		gt: [H, W, 3] ground truth image in [0, 1].

	Returns:
		Scalar tensor holding the mean absolute difference.
	"""
	return (rendered - gt).abs().mean()


def psnr(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	"""Peak signal to noise ratio, the metric the 3DGS paper reports.

	Args:
		rendered: [H, W, 3] rendered image in [0, 1].
		gt: [H, W, 3] ground truth image in [0, 1].

	Returns:
		Scalar tensor holding the PSNR in decibels, where higher is better.
	"""
	return -10.0 * torch.log10(((rendered - gt) ** 2).mean())


@lru_cache(maxsize=4)
def _window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	"""Builds the separable gaussian window SSIM averages over, cached per device and dtype.

	Args:
		size: Side length of the square window in pixels.
		sigma: Standard deviation of the gaussian in pixels.
		device: Device the window is allocated on.
		dtype: Floating point type of the window.

	Returns:
		[3, 1, size, size] window, one identical copy per color channel so conv2d can run grouped.
	"""
	coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
	g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
	g = g / g.sum()
	return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


class FusedSSIM(torch.autograd.Function):
	"""Autograd node for the CUDA SSIM, which fuses the blurs and the per pixel math."""

	@staticmethod
	def forward(ctx, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
		"""Computes mean SSIM and keeps the per pixel partials the backward pass blurs.

		Args:
			ctx: Autograd context the saved tensors are stashed on.
			rendered: [H, W, 3] contiguous CUDA image in [0, 1].
			gt: [H, W, 3] contiguous CUDA image in [0, 1].

		Returns:
			Scalar tensor holding the mean SSIM.
		"""
		ssim_map, d_mu, d_aa, d_ab = _get_rasterizer().ssim_forward(rendered, gt)
		ctx.save_for_backward(rendered, gt, d_mu, d_aa, d_ab)
		return ssim_map.mean()

	@staticmethod
	def backward(ctx, grad_mean: torch.Tensor) -> tuple[torch.Tensor, None]:
		"""Blurs the saved partials back onto the rendered image.

		Args:
			ctx: Autograd context holding the tensors saved during the forward pass.
			grad_mean: Scalar gradient of the loss with respect to the mean SSIM.

		Returns:
			The gradient for the rendered image, and None for the ground truth.
		"""
		rendered, gt, d_mu, d_aa, d_ab = ctx.saved_tensors
		grad = _get_rasterizer().ssim_backward(grad_mean.reshape(1).contiguous(), rendered, gt, d_mu, d_aa, d_ab)
		return grad, None


def ssim(rendered: torch.Tensor, gt: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
	"""Structural similarity, comparing local means, variances and covariance per channel.

	Unlike L1 this rewards getting local structure right rather than every pixel value, which is
	why the paper mixes the two.

	Args:
		rendered: [H, W, 3] rendered image in [0, 1].
		gt: [H, W, 3] ground truth image in [0, 1].
		size: Side length of the gaussian window, 11 to match the reference.
		sigma: Standard deviation of the window, 1.5 to match the reference.

	Returns:
		Scalar tensor holding the mean SSIM over every pixel and channel, where 1 is identical.
	"""
	if rendered.is_cuda and rendered.dtype == torch.float32 and size == 11 and sigma == 1.5:
		return FusedSSIM.apply(rendered.contiguous(), gt.detach().contiguous())

	a = rendered.permute(2, 0, 1).unsqueeze(0)
	b = gt.permute(2, 0, 1).unsqueeze(0)
	window = _window(size, sigma, a.device, a.dtype)

	def blur(t: torch.Tensor) -> torch.Tensor:
		"""Applies the gaussian window to one image.

		Args:
			t: [1, 3, H, W] image.

		Returns:
			[1, 3, H, W] locally weighted average, same padding so the size is unchanged.
		"""
		return F.conv2d(t, window, padding=size // 2, groups=3)

	mu_a, mu_b = blur(a), blur(b)
	mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
	var_a = blur(a * a) - mu_a2
	var_b = blur(b * b) - mu_b2
	cov = blur(a * b) - mu_ab

	numerator = (2 * mu_ab + C1) * (2 * cov + C2)
	denominator = (mu_a2 + mu_b2 + C1) * (var_a + var_b + C2)
	return (numerator / denominator).mean()
