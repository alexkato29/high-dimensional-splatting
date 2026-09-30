import pytest
import torch
import torch.nn.functional as F

from gaussian_splatting.rasterizer import project_gaussians, rasterize
from tests.oracle import rasterize_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rasterizer needs CUDA")


def scene(seed: int, n: int, width: int, height: int, channels: int = 3) -> tuple:
	"""Random gaussians in front of an identity camera, projected by the real projection kernel.

	Args:
		seed: Seed for every random draw.
		n: Number of gaussians.
		width: Image width in pixels.
		height: Image height in pixels.
		channels: Columns of the returned colors.

	Returns:
		(means2D, depths, radii, conics, colors, opacities), the inputs rasterize() takes.
	"""
	g = torch.Generator(device="cuda").manual_seed(seed)
	rand = lambda *shape: torch.rand(*shape, generator=g, device="cuda")
	means = rand(n, 3) * torch.tensor([4.0, 3.0, 4.0], device="cuda") + torch.tensor([-2.0, -1.5, 1.0], device="cuda")
	scales = torch.exp(rand(n, 3) * 3 - 4.5)
	quats = F.normalize(torch.randn(n, 4, generator=g, device="cuda"), dim=-1)
	opacities = rand(n, 1) * 0.95 + 0.04
	colors = rand(n, channels)
	focal = 0.8 * width
	with torch.no_grad():
		means2D, depths, radii, conics = project_gaussians(
			means, scales, quats, opacities, torch.eye(4, device="cuda"),
			focal, focal, width / 2, height / 2, width, height)
	return means2D, depths, radii, conics, colors, opacities


def render_and_grads(rasterizer, inputs: tuple, width: int, height: int, background: torch.Tensor,
					 grad_image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
	"""Renders once and backpropagates grad_image.

	Returns:
		The image and the gradients of means2D, conics, colors and opacities, in that order.
	"""
	means2D, depths, radii, conics, colors, opacities = inputs
	leaves = [t.detach().clone().requires_grad_(True) for t in (means2D, conics, colors, opacities)]
	image = rasterizer(leaves[0], depths, radii, leaves[1], leaves[2], leaves[3], width, height, background)
	image.backward(grad_image)
	return image.detach(), [t.grad for t in leaves]


def test_rgb_matches_oracle():
	width, height = 128, 96
	inputs = scene(0, 3000, width, height)
	g = torch.Generator(device="cuda").manual_seed(1)
	grad_image = torch.rand(height, width, 3, generator=g, device="cuda")
	background = torch.tensor([0.3, 0.5, 0.7], device="cuda")

	image, grads = render_and_grads(rasterize, inputs, width, height, background, grad_image)
	target_image, target_grads = render_and_grads(rasterize_reference, inputs, width, height, background, grad_image)

	diff = torch.abs(image - target_image)
	perc_99 = torch.quantile(diff.flatten(), 0.99)
	assert diff.max() < 2e-3 and perc_99 < 1e-4

	for name, got, want in zip(("means2D", "conics", "colors", "opacities"), grads, target_grads):
		scale = want.abs().max().item()
		bad = ~torch.isclose(got, want, rtol=1e-3, atol=1e-4 * scale)

		# A gaussian sitting on a cutoff blends in with one precision and not the other so a few can miss.
		assert bad.float().mean() < 1e-3, f"{name}: {bad.sum().item()} of {bad.numel()} outside tolerance"
		err = (got - want).abs().max().item()
		assert err < 2e-3 * scale, f"{name}: max error {err:.2e} of max {scale:.2e}"
