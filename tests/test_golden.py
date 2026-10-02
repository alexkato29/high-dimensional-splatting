import hashlib
import os
from pathlib import Path

import pytest
import torch

from gaussian_splatting.rasterizer import project_gaussians, rasterize
from gaussian_splatting.utils.dataset import ColmapDataset
from gaussian_splatting.utils.gaussian import GaussianModel
from tests.test_render import assert_grads_close

GOLDEN = os.environ.get("GS_GOLDEN")
UPDATE = os.environ.get("GS_UPDATE_GOLDEN") == "1"
FEATURES = 128

pytestmark = [
	pytest.mark.skipif(not torch.cuda.is_available(), reason="the rasterizer needs CUDA"),
	pytest.mark.skipif(GOLDEN is None, reason="set GS_GOLDEN to the reference file to run the golden test"),
]


def source_digest() -> str:
	"""Hash of every rasterizer and model source file so that the reference names the code that wrote it."""
	root = Path(__file__).parents[1] / "gaussian_splatting"
	files = sorted(p for p in root.rglob("*") if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"})
	h = hashlib.sha256()
	for p in files:
		h.update(str(p.relative_to(root)).encode())
		h.update(p.read_bytes())
	return h.hexdigest()[:16]


def environment() -> dict[str, str]:
	return {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "source": source_digest()}


def render_scene() -> dict[str, torch.Tensor]:
	"""Renders the first training view of GS_SCENE with GS_CHECKPOINT and backpropagates a seeded loss."""
	dataset = ColmapDataset(os.environ["GS_SCENE"])
	state = torch.load(os.environ["GS_CHECKPOINT"], map_location="cpu")
	model = GaussianModel(dataset.point_cloud)
	model.params = {name: torch.nn.Parameter(t.cuda().contiguous()) for name, t in state["params"].items()}
	model.active_sh_degree = state["active_sh_degree"]
	camera = dataset.train_cameras[0]
	n = model.means.shape[0]

	g = torch.Generator(device="cuda").manual_seed(0)
	features = torch.rand(n, FEATURES, generator=g, device="cuda", requires_grad=True)
	grad_image = torch.rand(camera.height, camera.width, 3, generator=g, device="cuda")
	grad_features = torch.rand(camera.height, camera.width, FEATURES, generator=g, device="cuda")

	means2D, depths, radii, conics = project_gaussians(
		model.means, model.scales, model.quats, model.opacities, camera.world_to_camera,
		camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
	# The SH backward steps Adam itself and never writes a gradient, so the colors are the leaf here.
	with torch.no_grad():
		colors = model.colors(camera.center, radii)
	colors.requires_grad_(True)
	image, feature_image = rasterize(
		means2D, depths, radii, conics, colors, model.opacities, camera.width, camera.height, features=features)
	torch.autograd.backward((image, feature_image), (grad_image, grad_features))

	out = {"image": image.detach(), "feature_image": feature_image.detach()}
	leaves = {name: model.params[name] for name in ("means", "scales", "quats", "opacities")}
	for name, param in [*leaves.items(), ("colors", colors), ("features", features)]:
		assert param.grad is not None, f"{name}: no gradient"
		out[f"grad_{name}"] = param.grad
	return {key: value.cpu() for key, value in out.items()}


def test_render_matches_golden_reference():
	path = Path(GOLDEN)
	got = render_scene()
	if UPDATE:
		torch.save({"environment": environment(), "outputs": got}, path)
		pytest.fail(f"wrote a new reference to {path}, rerun without GS_UPDATE_GOLDEN to test against it")
	assert path.exists(), f"GS_GOLDEN={path} does not exist, write one with GS_UPDATE_GOLDEN=1"

	reference = torch.load(path)
	written_by, now = reference["environment"], environment()
	context = f"reference written by {written_by}, running {now}"
	want = reference["outputs"]
	assert got.keys() == want.keys(), f"outputs changed from {sorted(want)} to {sorted(got)}. {context}"

	# Same GPU and build should reproduce the forward exactly, though different GPUs can get tolderance.
	exact = written_by["gpu"] == now["gpu"] and written_by["torch"] == now["torch"]
	for key in ("image", "feature_image"):
		if exact:
			assert torch.equal(got[key], want[key]), f"{key} changed. {context}"
		else:
			torch.testing.assert_close(got[key], want[key], rtol=0, atol=1e-4, msg=f"{key} drifted. {context}")

	grads = sorted(key for key in got if key.startswith("grad_"))
	try:
		assert_grads_close(tuple(grads), [got[k] for k in grads], [want[k] for k in grads])
	except AssertionError as error:
		raise AssertionError(f"{error}. {context}") from error
