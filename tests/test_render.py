from collections.abc import Sequence

import pytest
import torch

from gaussian_splatting.rasterizer import rasterize
from tests.oracle import rasterize_reference
from tests.scenes import scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rasterizer needs CUDA")


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


def assert_image_close(got: torch.Tensor, want: torch.Tensor, max_error: float = 2e-3) -> None:
	diff = (got - want).abs()
	perc_99 = torch.quantile(diff.flatten(), 0.99)
	assert diff.max() < max_error and perc_99 < 1e-4, f"max {diff.max().item():.2e}, 99th {perc_99.item():.2e}"


def assert_grads_close(names: tuple[str, ...], grads: Sequence[torch.Tensor | None],
					   target_grads: Sequence[torch.Tensor | None]) -> None:
	for name, got, want in zip(names, grads, target_grads):
		assert got is not None and want is not None, f"{name}: no gradient"
		assert got.shape == want.shape, f"{name}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
		if want.numel() == 0:
			continue
		scale = want.abs().max().item()
		bad = ~torch.isclose(got, want, rtol=1e-3, atol=1e-4 * scale)

		# A gaussian sitting on a cutoff blends in with one precision and not the other so a few can miss.
		assert bad.float().mean() < 1e-3, f"{name}: {bad.sum().item()} of {bad.numel()} outside tolerance"
		err = (got - want).abs().max().item()
		assert err <= 2e-3 * scale, f"{name}: max error {err:.2e} of max {scale:.2e}"


@pytest.mark.parametrize("channels", [32, 64, 128])
@torch.no_grad()
def test_features_forward_matches_oracle(channels):
	width, height = 128, 96
	inputs = scene(0, 3000, width, height)
	g = torch.Generator(device="cuda").manual_seed(2)
	features = torch.rand(3000, channels, generator=g, device="cuda")

	image, feature_image = rasterize(*inputs, width, height, features=features)
	assert torch.equal(image, rasterize(*inputs, width, height))
	means2D, depths, radii, conics, _, opacities = inputs
	assert_image_close(feature_image, rasterize_reference(means2D, depths, radii, conics, features, opacities, width, height))


def test_unsupported_feature_width_is_rejected():
	width, height = 32, 32
	inputs = scene(0, 100, width, height)
	with pytest.raises(RuntimeError, match="32, 64 or 128"):
		rasterize(*inputs, width, height, features=torch.rand(100, 24, device="cuda"))


def test_rgb_matches_oracle():
	width, height = 128, 96
	inputs = scene(0, 3000, width, height)
	g = torch.Generator(device="cuda").manual_seed(1)
	grad_image = torch.rand(height, width, 3, generator=g, device="cuda")
	background = torch.tensor([0.3, 0.5, 0.7], device="cuda")

	image, grads = render_and_grads(rasterize, inputs, width, height, background, grad_image)
	target_image, target_grads = render_and_grads(rasterize_reference, inputs, width, height, background, grad_image)

	assert_image_close(image, target_image)

	assert_grads_close(("means2D", "conics", "colors", "opacities"), grads, target_grads)


@pytest.mark.parametrize("channels", [32, 64, 128])
def test_features_backward_matches_oracle(channels):
	width, height = 128, 96
	means2D, depths, radii, conics, colors, opacities = scene(0, 3000, width, height)
	g = torch.Generator(device="cuda").manual_seed(3)
	features = torch.rand(3000, channels, generator=g, device="cuda")
	grad_features = torch.rand(height, width, channels, generator=g, device="cuda")

	def feature_grads(render):
		leaves = [t.detach().clone().requires_grad_(True) for t in (means2D, conics, features, opacities)]
		render(*leaves).backward(grad_features)
		return [t.grad for t in leaves]

	got = feature_grads(lambda m, c, f, o: rasterize(m, depths, radii, c, colors, o, width, height, features=f)[1])
	want = feature_grads(lambda m, c, f, o: rasterize_reference(m, depths, radii, c, f, o, width, height))
	assert_grads_close(("means2D", "conics", "features", "opacities"), got, want)
