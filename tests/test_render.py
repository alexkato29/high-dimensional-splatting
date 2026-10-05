import math
from collections.abc import Sequence

import pytest
import torch

from gaussian_splatting.rasterizer import _get_rasterizer, rasterize
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
	# kthvalue because quantile refuses more than 16M elements and a D = 32 feature image of Truck is 17M.
	perc_99 = diff.flatten().kthvalue(max(1, math.ceil(0.99 * diff.numel()))).values
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


@pytest.mark.parametrize("channels, slice_width", [(32, 32), (64, 64), (64, 32), (128, 128), (128, 64), (128, 32)])
def test_features_backward_matches_oracle(channels, slice_width, monkeypatch):
	monkeypatch.setattr("gaussian_splatting.rasterizer.FEATURE_SLICE_WIDTH", slice_width)
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


@pytest.mark.parametrize("channels", [32, 64, 128])
def test_frozen_features_still_backpropagate_into_geometry(channels):
	width, height = 128, 96
	means2D, depths, radii, conics, colors, opacities = scene(0, 3000, width, height)
	g = torch.Generator(device="cuda").manual_seed(3)
	features = torch.rand(3000, channels, generator=g, device="cuda")
	grad_features = torch.rand(height, width, channels, generator=g, device="cuda")

	def geometry_grads(render):
		leaves = [t.detach().clone().requires_grad_(True) for t in (means2D, conics, opacities)]
		render(*leaves).backward(grad_features)
		return [t.grad for t in leaves]

	# Features never require grad here, so only the geometry asks for the feature backward.
	got = geometry_grads(lambda m, c, o: rasterize(m, depths, radii, c, colors, o, width, height, features=features)[1])
	want = geometry_grads(lambda m, c, o: rasterize_reference(m, depths, radii, c, features, o, width, height))
	assert_grads_close(("means2D", "conics", "opacities"), got, want)


def flat_scene(means2D: list, depths: list, sigmas: list, opacities: list) -> tuple:
	"""Round gaussians placed directly in pixels, skipping projection so each edge case lands exactly.

	Radii only gate culling in the kernel, binning comes from opacity and conic. So, they are all 1.
	"""
	n = len(depths)
	sigma = torch.tensor(sigmas, device="cuda")
	g = torch.Generator(device="cuda").manual_seed(n)
	return (
		torch.tensor(means2D, device="cuda"),
		torch.tensor(depths, device="cuda"),
		torch.ones(n, device="cuda"),
		torch.stack([sigma**-2, torch.zeros_like(sigma), sigma**-2], dim=-1),
		torch.rand(n, 3, generator=g, device="cuda"),
		torch.tensor(opacities, device="cuda")[:, None],
	)


def assert_matches_oracle(inputs: tuple, width: int, height: int) -> None:
	"""RGB and D = 128 features, forward and backward, against the float64 oracle."""
	means2D, depths, radii, conics, colors, opacities = inputs
	g = torch.Generator(device="cuda").manual_seed(7)
	background = torch.tensor([0.3, 0.5, 0.7], device="cuda")
	grad_image = torch.rand(height, width, 3, generator=g, device="cuda")
	image, grads = render_and_grads(rasterize, inputs, width, height, background, grad_image)
	target_image, target_grads = render_and_grads(rasterize_reference, inputs, width, height, background, grad_image)
	assert_image_close(image, target_image)
	assert_grads_close(("means2D", "conics", "colors", "opacities"), grads, target_grads)

	# D = 128 runs batches of 64 and two lanes per pixel, the layout furthest from RGB.
	features = torch.rand(len(depths), 128, generator=g, device="cuda")
	grad_features = torch.rand(height, width, 128, generator=g, device="cuda")

	def feature_render(render):
		leaves = [t.detach().clone().requires_grad_(True) for t in (means2D, conics, features, opacities)]
		out = render(*leaves)
		out.backward(grad_features)
		return out.detach(), [t.grad for t in leaves]

	got_image, got = feature_render(
		lambda m, c, f, o: rasterize(m, depths, radii, c, colors, o, width, height, features=f)[1])
	want_image, want = feature_render(lambda m, c, f, o: rasterize_reference(m, depths, radii, c, f, o, width, height))
	assert_image_close(got_image, want_image)
	assert_grads_close(("means2D", "conics", "features", "opacities"), got, want)


def test_depth_ties_blend_in_index_order():
	inputs = flat_scene([[7.5, 8.5], [9.5, 8.5]], [2.0, 2.0], [3.0, 3.0], [0.7, 0.7])
	assert_matches_oracle(inputs, 16, 16)

	swapped = tuple(t.flip(0) for t in inputs)
	assert not torch.equal(rasterize(*inputs, 16, 16), rasterize(*swapped, 16, 16))


def test_min_alpha_threshold():
	inputs = flat_scene([[4.5, 4.5], [12.5, 4.5]], [1.0, 1.0], [0.5, 0.5], [1.01 / 255, 0.99 / 255])
	assert_matches_oracle(inputs, 16, 8)

	image = rasterize(*inputs, 16, 8)
	assert image[4, 4].abs().sum() > 0, "the gaussian above MIN_ALPHA must blend"
	assert torch.equal(image[4, 12], torch.zeros(3, device="cuda")), "the one below must not"


def test_max_alpha_clamp():
	inputs = flat_scene([[8.5, 8.5]], [1.0], [2.0], [1.0])
	assert_matches_oracle(inputs, 16, 16)

	colors = inputs[4]
	torch.testing.assert_close(rasterize(*inputs, 16, 16)[8, 8], 0.99 * colors[0])


def test_transmittance_floor_stops_the_blend():
	n = 50
	inputs = flat_scene([[8.5, 8.5]] * n, [1.0 + i for i in range(n)], [4.0] * n, [0.9] * n)
	assert_matches_oracle(inputs, 16, 16)


def test_dense_tile_walks_past_every_batch():
	g = torch.Generator(device="cuda").manual_seed(11)
	n = 2000
	means = (torch.rand(n, 2, generator=g, device="cuda") * 16).tolist()
	depths = (1 + torch.rand(n, generator=g, device="cuda")).tolist()
	inputs = flat_scene(means, depths, [12.0] * n, [0.02] * n)
	assert_matches_oracle(inputs, 16, 16)

	means2D, depths_t, radii, conics, colors, opacities = inputs
	n_contrib = _get_rasterizer().rasterize(means2D, depths_t, radii, conics, colors, opacities,
											torch.zeros(3, device="cuda"), 16, 16)[2]
	assert n_contrib.min() > 256, f"some pixel stopped at {n_contrib.min().item()}, before the 256 batch edge"
