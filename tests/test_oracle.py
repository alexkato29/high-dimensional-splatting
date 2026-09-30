import pytest
import torch

from tests.oracle import rasterize_reference

BACKGROUND = (0.3, 0.5, 0.7)


def render_one_pixel(opacities: list[float]) -> torch.Tensor:
	"""Stacks gaussians on the center of a 1x1 image, front to back, gaussian i colored with channel i.

	Centered on the pixel, each gaussian's alpha is exactly its opacity (clamped to 0.99).
	"""
	n = len(opacities)
	return rasterize_reference(
		means2D=torch.full((n, 2), 0.5),
		depths=torch.arange(n, dtype=torch.float32),
		radii=torch.ones(n),
		conics=torch.tensor([[1.0, 0.0, 1.0]]).repeat(n, 1),
		colors=torch.eye(3)[:n],
		opacities=torch.tensor(opacities).reshape(n, 1),
		width=1,
		height=1,
		background=torch.tensor(BACKGROUND),
	)[0, 0]


@pytest.mark.parametrize("opacities, expected", [
	# alpha * color + (1 - alpha) * background.
	([0.5], [0.5 + 0.5 * 0.3, 0.5 * 0.5, 0.5 * 0.7]),
	# Clamped to 0.99. The third would leave 1e-6 < 1e-4, so it is skipped and 1e-4 shows background.
	([1.0, 1.0, 1.0], [0.99 + 1e-4 * 0.3, 0.0099 + 1e-4 * 0.5, 1e-4 * 0.7]),
	# The third would leave 5e-5, skipped even though 0.005 of the light still reaches it.
	([0.99, 0.5, 0.99], [0.99 + 0.005 * 0.3, 0.005 + 0.005 * 0.5, 0.005 * 0.7]),
	# No gaussians, only background.
	([], list(BACKGROUND)),
])
def test_hand_solved_pixel(opacities, expected):
	assert torch.allclose(render_one_pixel(opacities), torch.tensor(expected), atol=1e-6)
