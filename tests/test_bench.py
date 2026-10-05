import pytest
import torch

from gaussian_splatting.utils.dataset import Camera
from tests.scenes import gaussians

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="both rasterizers need CUDA")
# Imports gsplat, which only the bench dependency group installs.
harness = pytest.importorskip("bench.harness")


def test_gate_accepts_gsplat_and_refuses_a_corrupted_output():
	width, height, channels = 128, 96, 32
	g = gaussians(0, 2000)
	camera = Camera(0, "synthetic", torch.eye(4, device="cuda"), torch.zeros(3, device="cuda"), 0.8 * width,
					0.8 * width, width / 2, height / 2, width, height, torch.zeros(height, width, 3, dtype=torch.uint8))
	gen = torch.Generator(device="cuda").manual_seed(1)
	inputs = {
		"means": g["means"], "scales": g["scales"], "quats": g["quats"], "opacities": g["opacities"][:, 0],
		"colors": g["colors"],
		"features": torch.rand(2000, channels, generator=gen, device="cuda"),
		"grad_image": torch.rand(height, width, 3, generator=gen, device="cuda"),
		"grad_feature_image": torch.rand(height, width, channels, generator=gen, device="cuda"),
	}
	ours = harness.render_with_grads(harness.render_ours, inputs, camera=camera)
	theirs = harness.render_with_grads(harness.render_gsplat, inputs, camera=camera)
	assert harness.agrees(theirs, ours, "clean")

	# One percent off on a single output must stop the timing run.
	for name in ("feature_image", "grad_means"):
		assert not harness.agrees({**theirs, name: theirs[name] * 1.01}, ours, f"{name} corrupted")
