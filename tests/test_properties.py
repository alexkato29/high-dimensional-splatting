import pytest
import torch
from hypothesis import given

from gaussian_splatting.rasterizer import rasterize
from tests.scenes import gaussians, project, scene, scene_sizes
from tests.test_render import assert_grads_close, assert_image_close

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rasterizer needs CUDA")

CHANNELS = [32, 64, 128]
GRAD_NAMES = ("means2D", "conics", "colors", "opacities", "features")


def rand(g: torch.Generator, *shape: int) -> torch.Tensor:
	return torch.rand(*shape, generator=g, device="cuda")


def render_with_grads(inputs: tuple, features: torch.Tensor, width: int, height: int, grad_image: torch.Tensor,
					  grad_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list]:
	"""Renders RGB and features once and backpropagates both images.

	Returns:
		The image, the feature image and the gradients in GRAD_NAMES order.
	"""
	means2D, depths, radii, conics, colors, opacities = inputs
	leaves = [t.detach().clone().requires_grad_(True) for t in (means2D, conics, colors, opacities, features)]
	image, feature_image = rasterize(leaves[0], depths, radii, leaves[1], leaves[2], leaves[3], width, height,
									 features=leaves[4])
	torch.autograd.backward((image, feature_image), (grad_image, grad_features))
	return image.detach(), feature_image.detach(), [t.grad for t in leaves]


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
def test_render_is_deterministic(channels, sizes):
	seed, n, width, height = sizes
	inputs = scene(seed, n, width, height)
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = rand(g, n, channels)
	grad_image, grad_features = rand(g, height, width, 3), rand(g, height, width, channels)

	image, feature_image, grads = render_with_grads(inputs, features, width, height, grad_image, grad_features)
	image_again, feature_image_again, grads_again = render_with_grads(
		inputs, features, width, height, grad_image, grad_features)

	assert torch.equal(image, image_again)
	assert torch.equal(feature_image, feature_image_again)
	# atomicAdd order changes between runs, so the backward is only reproducible up to rounding.
	assert_grads_close(GRAD_NAMES, grads_again, grads)


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
def test_gaussian_order_does_not_matter(channels, sizes):
	seed, n, width, height = sizes
	means2D, depths, radii, conics, colors, opacities = scene(seed, n, width, height)
	# Depths only feed the sort, so their ranks render the same scene with every tie broken.
	depths = depths.argsort().argsort().float()
	inputs = (means2D, depths, radii, conics, colors, opacities)
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = rand(g, n, channels)
	grad_image, grad_features = rand(g, height, width, 3), rand(g, height, width, channels)
	perm = torch.randperm(n, generator=g, device="cuda")
	image, feature_image, grads = render_with_grads(inputs, features, width, height, grad_image, grad_features)
	perm_image, perm_feature_image, perm_grads = render_with_grads(
		tuple(t[perm] for t in inputs), features[perm], width, height, grad_image, grad_features)

	assert torch.equal(image, perm_image)
	assert torch.equal(feature_image, perm_feature_image)
	unperm = perm.argsort()
	assert_grads_close(GRAD_NAMES, [grad[unperm] for grad in perm_grads], grads)


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@torch.no_grad()
def test_invisible_gaussians_change_nothing(channels, sizes):
	seed, n, width, height = sizes
	visible = gaussians(seed, n)
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = rand(g, n, channels)

	wall, behind, outside = gaussians(seed + 1, 300), gaussians(seed + 2, 300), gaussians(seed + 3, 300)
	wall["means"][:, 2] *= 0.2
	wall["opacities"].zero_()
	behind["means"][:, 2] *= -1
	outside["means"][:, 0] += 20
	hidden = [wall, behind, outside]
	everything = {key: torch.cat([visible[key]] + [h[key] for h in hidden]) for key in visible}
	all_features = torch.cat([features, rand(g, 900, channels)])

	image, feature_image = rasterize(*project(visible, width, height), width, height, features=features)
	image_all, feature_image_all = rasterize(*project(everything, width, height), width, height,
											 features=all_features)

	assert torch.equal(image, image_all)
	assert torch.equal(feature_image, feature_image_all)


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@torch.no_grad()
def test_blend_weights_are_convex(channels, sizes):
	seed, n, width, height = sizes
	means2D, depths, radii, conics, _, opacities = scene(seed, n, width, height)
	geometry = (means2D, depths, radii, conics, torch.ones(n, 3, device="cuda"), opacities)

	# Ones over a black background blend to 1 - T, the share of light the gaussians took.
	rgb_coverage, coverage = rasterize(*geometry, width, height, features=torch.ones(n, channels, device="cuda"))
	assert torch.equal(coverage, coverage[..., :1].expand_as(coverage))
	assert (coverage >= 0).all() and (coverage <= 1 + 1e-6).all()
	# Both passes compute alpha and T on their own so they must agree on the coverage.
	assert_image_close(rgb_coverage, coverage[..., :3])

	constant = rasterize(*geometry, width, height, features=torch.full((n, channels), 0.3, device="cuda"))[1]
	torch.testing.assert_close(constant, 0.3 * coverage, rtol=1e-5, atol=1e-6)

	low, high = 0.25, 0.75
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = low + (high - low) * rand(g, n, channels)
	feature_image = rasterize(*geometry, width, height, features=features)[1]
	assert (feature_image >= low * coverage - 1e-5).all()
	assert (feature_image <= high * coverage + 1e-5).all()


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
def test_feature_channels_are_independent(channels, sizes):
	seed, n, width, height = sizes
	inputs = scene(seed, n, width, height)
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = rand(g, n, channels)
	channel = seed % channels
	others = torch.arange(channels, device="cuda") != channel

	with torch.no_grad():
		bumped = features.clone()
		bumped[:, channel] += rand(g, n)
		image = rasterize(*inputs, width, height, features=features)[1]
		bumped_image = rasterize(*inputs, width, height, features=bumped)[1]

	# Weights come from geometry alone, so the other channels run the exact same arithmetic.
	assert torch.equal(image[..., others], bumped_image[..., others])

	# A loss on one channel can only send gradient back to that channel.
	leaf = features.clone().requires_grad_(True)
	grad_features = torch.zeros(height, width, channels, device="cuda")
	grad_features[..., channel] = rand(g, height, width)
	rasterize(*inputs, width, height, features=leaf)[1].backward(grad_features)
	assert leaf.grad is not None
	assert (leaf.grad[:, others] == 0).all()


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@torch.no_grad()
def test_features_blend_linearly(channels, sizes):
	seed, n, width, height = sizes
	inputs = scene(seed, n, width, height)
	g = torch.Generator(device="cuda").manual_seed(seed)
	first, second = rand(g, n, channels), rand(g, n, channels)
	a, b = 2.0, -0.5

	mixed = rasterize(*inputs, width, height, features=a * first + b * second)[1]
	want = (a * rasterize(*inputs, width, height, features=first)[1]
			+ b * rasterize(*inputs, width, height, features=second)[1])
	torch.testing.assert_close(mixed, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("half", [32, 64])
@given(sizes=scene_sizes())
def test_concatenated_features_render_like_halves(half, sizes):
	seed, n, width, height = sizes
	inputs = scene(seed, n, width, height)
	g = torch.Generator(device="cuda").manual_seed(seed)
	left, right = rand(g, n, half), rand(g, n, half)
	grad_features = rand(g, height, width, 2 * half)

	def render_with_grad(features, grad):
		leaf = features.clone().requires_grad_(True)
		image = rasterize(*inputs, width, height, features=leaf)[1]
		image.backward(grad)
		return image.detach(), leaf.grad

	whole_image, whole_grad = render_with_grad(torch.cat([left, right], dim=1), grad_features)
	left_image, left_grad = render_with_grad(left, grad_features[..., :half].contiguous())
	right_image, right_grad = render_with_grad(right, grad_features[..., half:].contiguous())

	# The whole width runs a different kernel instance than each half due to templating.
	assert torch.equal(whole_image, torch.cat([left_image, right_image], dim=-1))
	assert_grads_close(("features",), [whole_grad], [torch.cat([left_grad, right_grad], dim=1)])
