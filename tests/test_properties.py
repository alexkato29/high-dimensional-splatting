import pytest
import torch
from hypothesis import given

from gaussian_splatting.rasterizer import rasterize
from tests.scenes import gaussians, project, scene, scene_sizes
from tests.test_render import assert_grads_close, assert_image_close

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rasterizer needs CUDA")

CHANNELS = [32, 64, 128]
GRAD_NAMES = ("means2D", "conics", "colors", "opacities", "features")


def rand(seed: int, *shape: int) -> torch.Tensor:
	g = torch.Generator(device="cuda").manual_seed(seed)
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
	features = rand(seed, n, channels)
	grad_image, grad_features = rand(seed + 1, height, width, 3), rand(seed + 2, height, width, channels)

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
	features = rand(seed, n, channels)
	grad_image, grad_features = rand(seed + 1, height, width, 3), rand(seed + 2, height, width, channels)

	g = torch.Generator(device="cuda").manual_seed(seed)
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
	features = rand(seed, n, channels)

	wall, behind, outside = gaussians(seed + 1, 300), gaussians(seed + 2, 300), gaussians(seed + 3, 300)
	wall["means"][:, 2] *= 0.2
	wall["opacities"].zero_()
	behind["means"][:, 2] *= -1
	outside["means"][:, 0] += 20
	hidden = [wall, behind, outside]
	everything = {key: torch.cat([visible[key]] + [h[key] for h in hidden]) for key in visible}
	all_features = torch.cat([features, rand(seed + 4, 900, channels)])

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
	features = low + (high - low) * rand(seed, n, channels)
	feature_image = rasterize(*geometry, width, height, features=features)[1]
	assert (feature_image >= low * coverage - 1e-5).all()
	assert (feature_image <= high * coverage + 1e-5).all()
