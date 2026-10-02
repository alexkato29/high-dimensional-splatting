import torch
import torch.nn.functional as F
from hypothesis import strategies as st

from gaussian_splatting.rasterizer import project_gaussians

# (means2D, depths, radii, conics, colors, opacities)
RasterizeInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def gaussians(seed: int, n: int, channels: int = 3) -> dict[str, torch.Tensor]:
	"""Random 3D gaussians in a box in front of an identity camera.

	Args:
		seed: Seed for every random draw.
		n: Number of gaussians.
		channels: Columns of the returned colors.

	Returns:
		means [N, 3], scales [N, 3], quats [N, 4], opacities [N, 1] and colors [N, channels], by name.
	"""
	g = torch.Generator(device="cuda").manual_seed(seed)
	rand = lambda *shape: torch.rand(*shape, generator=g, device="cuda")
	means = rand(n, 3) * torch.tensor([4.0, 3.0, 4.0], device="cuda") + torch.tensor([-2.0, -1.5, 1.0], device="cuda")
	scales = torch.exp(rand(n, 3) * 3 - 4.5)
	quats = F.normalize(torch.randn(n, 4, generator=g, device="cuda"), dim=-1)
	opacities = rand(n, 1) * 0.95 + 0.04
	colors = rand(n, channels)
	return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "colors": colors}


@torch.no_grad()
def project(g: dict[str, torch.Tensor], width: int, height: int,
			world_to_camera: torch.Tensor | None = None) -> RasterizeInputs:
	"""Projects gaussians() output with the real projection kernel.

	Returns:
		(means2D, depths, radii, conics, colors, opacities), the inputs rasterize() takes.
	"""
	if world_to_camera is None:
		world_to_camera = torch.eye(4, device="cuda")
	focal = 0.8 * width
	means2D, depths, radii, conics = project_gaussians(
		g["means"], g["scales"], g["quats"], g["opacities"], world_to_camera,
		focal, focal, width / 2, height / 2, width, height)
	return means2D, depths, radii, conics, g["colors"], g["opacities"]


def scene(seed: int, n: int, width: int, height: int, channels: int = 3) -> RasterizeInputs:
	"""Random gaussians projected through an identity camera, see gaussians() and project()."""
	return project(gaussians(seed, n, channels), width, height)


@st.composite
def scene_sizes(draw) -> tuple[int, int, int, int]:
	"""Draws (seed, n, width, height) for scene().

	Hypothesis draws only these knobs and torch draws the gaussians, so shrinking works on four
	integers instead of thousands of floats.
	"""
	seed = draw(st.integers(min_value=0, max_value=(2**32 - 1)))
	width = draw(st.integers(min_value=1, max_value=160))
	height = draw(st.integers(min_value=1, max_value=160))

	# The fp64 oracle keeps every pixel and gaussian for autograd. For large feature vectors like D=128,
	# memory requirements blow up quickly. This forces it to stay reasonably small.
	max_n = min(10_000, (128 * 96 * 3000) // (width * height))
	n = draw(st.integers(min_value=0, max_value=max_n))

	return (seed, n, width, height)
