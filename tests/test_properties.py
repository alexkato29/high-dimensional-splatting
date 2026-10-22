import pytest
import torch
import torch.nn.functional as F
from hypothesis import given

from gaussian_splatting.rasterizer import rasterize
from gaussian_splatting.rasterizer.projection import quat_to_rotmat
from tests.scenes import gaussians, project, scene, scene_sizes, with_size_examples
from tests.test_render import (
	FEATURE_MAX_ERROR,
	FEATURE_PERC_99,
	assert_features_close,
	assert_grads_close,
	assert_image_close,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rasterizer needs CUDA")

CHANNELS = [32, 64, 128]
GRAD_NAMES = ("means2D", "conics", "colors", "opacities", "features")
# Room for one gaussian flipping across MIN_ALPHA between two renders of the same scene.
FLIP_MAX_ERROR = 1e-2


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
@with_size_examples
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
@with_size_examples
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
@with_size_examples
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
	assert_features_close(feature_image_all, feature_image, renders=2)


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@with_size_examples
@torch.no_grad()
def test_blend_weights_are_convex(channels, sizes):
	seed, n, width, height = sizes
	means2D, depths, radii, conics, _, opacities = scene(seed, n, width, height)
	geometry = (means2D, depths, radii, conics, torch.ones(n, 3, device="cuda"), opacities)

	# Ones over a black background blend to 1 - T, the share of light the gaussians took.
	rgb_coverage, coverage = rasterize(*geometry, width, height, features=torch.ones(n, channels, device="cuda"))
	assert torch.equal(coverage, coverage[..., :1].expand_as(coverage))
	# Rounded TF32 weights can each land half a step high, so a fully covered pixel may read just over 1.
	assert (coverage >= 0).all() and (coverage <= 1 + FEATURE_MAX_ERROR).all()
	# Both passes compute alpha and T on their own so they must agree on the coverage.
	assert_features_close(coverage[..., :3], rgb_coverage)

	constant = rasterize(*geometry, width, height, features=torch.full((n, channels), 0.3, device="cuda"))[1]
	assert_features_close(constant, 0.3 * coverage, renders=2)

	low, high = 0.25, 0.75
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = low + (high - low) * rand(g, n, channels)
	feature_image = rasterize(*geometry, width, height, features=features)[1]
	assert (feature_image >= low * coverage - 1e-5).all()
	assert (feature_image <= high * coverage + 1e-5).all()


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@with_size_examples
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
@with_size_examples
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
	assert_features_close(mixed, want, renders=1 + abs(a) + abs(b))


@pytest.mark.parametrize("half", [32, 64])
@given(sizes=scene_sizes())
@with_size_examples
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


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
	"""Hamilton product of (w, x, y, z) quaternions, broadcasting over leading dims.

	quat_to_rotmat(quat_multiply(q1, q2)) = quat_to_rotmat(q1) @ quat_to_rotmat(q2), so q2 rotates first.
	"""
	a1, b1, c1, d1 = q1.unbind(-1)
	a2, b2, c2, d2 = q2.unbind(-1)
	return torch.stack([
		a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
		a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
		a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
		a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
	], dim=-1)


def mirror(g: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	"""Reflects every gaussian across the camera's vertical plane, x -> -x in camera space.

	Args:
		g: gaussians() output, seen by the identity camera.

	Returns:
		A new dict in the same format, input untouched.
	"""
	mirrored = dict(g)
	flip_x = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=g["means"].device, dtype=g["means"].dtype))
	mirrored["means"] = g["means"] @ flip_x

	# Mirroring an ellipsoid looks the same as turning it 180 degrees about the mirror's normal.
	half_turn_x = torch.tensor([0.0, 1.0, 0.0, 0.0], device=g["quats"].device)
	mirrored["quats"] = quat_multiply(half_turn_x, g["quats"])
	return mirrored


def move(g: dict[str, torch.Tensor], rotation: torch.Tensor,
		 translation: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
	"""Moves every gaussian by one rigid transform and moves the camera with them.

	Args:
		g: gaussians() output, seen by the identity camera.
		rotation: [4] unit quaternion (w, x, y, z).
		translation: [3].

	Returns:
		The moved gaussians as a new dict, and the [4, 4] world_to_camera that still sees them
		exactly as the identity camera saw the originals.
	"""
	rot = quat_to_rotmat(rotation[None])[0]
	inv_rot = rot.T
	inv_t = -inv_rot @ translation

	moved = dict(g)
	# Points are rows, so we actually apply the transposed rotation.
	moved["means"] = g["means"] @ inv_rot + translation
	moved["quats"] = quat_multiply(rotation, g["quats"])

	W = torch.eye(4, device=rotation.device)
	W[:3, :3] = inv_rot
	W[:3, 3] = inv_t

	return moved, W


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@with_size_examples
@torch.no_grad()
def test_mirrored_scene_renders_mirrored_image(channels, sizes):
	seed, n, width, height = sizes
	original = gaussians(seed, n)
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = rand(g, n, channels)

	image, feature_image = rasterize(*project(original, width, height), width, height, features=features)
	mirrored_image, mirrored_feature_image = rasterize(
		*project(mirror(original), width, height), width, height, features=features)

	# Same math in a different float order, so only close.
	assert_image_close(mirrored_image, image.flip(1), max_error=FLIP_MAX_ERROR)
	assert_image_close(mirrored_feature_image, feature_image.flip(1), max_error=FLIP_MAX_ERROR,
					   max_perc_99=2 * FEATURE_PERC_99)


@pytest.mark.parametrize("channels", CHANNELS)
@given(sizes=scene_sizes())
@with_size_examples
@torch.no_grad()
def test_moving_scene_and_camera_together_changes_nothing(channels, sizes):
	seed, n, width, height = sizes
	original = gaussians(seed, n)
	g = torch.Generator(device="cuda").manual_seed(seed)
	features = rand(g, n, channels)
	rotation = F.normalize(torch.randn(4, generator=g, device="cuda"), dim=0)
	translation = torch.randn(3, generator=g, device="cuda")

	image, feature_image = rasterize(*project(original, width, height), width, height, features=features)
	moved, world_to_camera = move(original, rotation, translation)
	moved_image, moved_feature_image = rasterize(
		*project(moved, width, height, world_to_camera), width, height, features=features)

	assert_image_close(moved_image, image, max_error=FLIP_MAX_ERROR)
	assert_image_close(moved_feature_image, feature_image, max_error=FLIP_MAX_ERROR, max_perc_99=2 * FEATURE_PERC_99)


def test_quat_multiply_composes_rotations():
	g = torch.Generator().manual_seed(0)
	q1, q2 = F.normalize(torch.randn(2, 64, 4, generator=g, dtype=torch.float64), dim=-1)
	torch.testing.assert_close(quat_to_rotmat(quat_multiply(q1, q2)), quat_to_rotmat(q1) @ quat_to_rotmat(q2))
