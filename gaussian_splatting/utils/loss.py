from functools import lru_cache

import torch
import torch.nn.functional as F

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def l1_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	return (rendered - gt).abs().mean()


def psnr(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	return -10.0 * torch.log10(((rendered - gt) ** 2).mean())


@lru_cache(maxsize=4)
def _window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
	g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
	g = g / g.sum()
	return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(rendered: torch.Tensor, gt: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
	"""Mean SSIM between two [H, W, 3] images, gaussian-windowed per channel."""
	a = rendered.permute(2, 0, 1).unsqueeze(0)
	b = gt.permute(2, 0, 1).unsqueeze(0)
	window = _window(size, sigma, a.device, a.dtype)

	def blur(t: torch.Tensor) -> torch.Tensor:
		return F.conv2d(t, window, padding=size // 2, groups=3)

	mu_a, mu_b = blur(a), blur(b)
	mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
	var_a = blur(a * a) - mu_a2
	var_b = blur(b * b) - mu_b2
	cov = blur(a * b) - mu_ab

	numerator = (2 * mu_ab + C1) * (2 * cov + C2)
	denominator = (mu_a2 + mu_b2 + C1) * (var_a + var_b + C2)
	return (numerator / denominator).mean()
