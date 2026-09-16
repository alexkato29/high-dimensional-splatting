from functools import lru_cache

import torch
import torch.nn.functional as F

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
