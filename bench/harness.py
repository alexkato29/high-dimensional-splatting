import argparse

import gsplat
import torch

from gaussian_splatting.rasterizer import project_gaussians, rasterize
from gaussian_splatting.rasterizer.projection import LOW_PASS, NEAR_PLANE
from gaussian_splatting.utils.dataset import Camera, ColmapDataset
from gaussian_splatting.utils.gaussian import GaussianModel
from tests.test_render import assert_grads_close, assert_image_close

OUTPUTS = ("image", "feature_image")
LEAVES = ("means", "scales", "quats", "opacities", "colors", "features")


def load_model(dataset: ColmapDataset, checkpoint: str) -> GaussianModel:
	"""Builds a model from a model.pt written by save_checkpoint."""
	state = torch.load(checkpoint, map_location="cpu")
	model = GaussianModel(dataset.point_cloud)
	model.params = {name: torch.nn.Parameter(t.cuda().contiguous()) for name, t in state["params"].items()}
	model.active_sh_degree = state["active_sh_degree"]
	return model


def shared_inputs(model: GaussianModel, camera: Camera, channels: int, seed: int = 0) -> dict[str, torch.Tensor]:
	"""Everything both libraries render from, activated and detached so each run gets fresh leaves.

	Colors come from our SH once and are passed in as plain values, so SH is not part of the comparison.
	"""
	g = torch.Generator(device="cuda").manual_seed(seed)
	with torch.no_grad():
		_, _, radii, _ = project_gaussians(
			model.means, model.scales, model.quats, model.opacities, camera.world_to_camera,
			camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
		colors = model.colors(camera.center, radii)
	n = model.means.shape[0]
	# TODO: Replace this with actual feature vectors
	return {
		"means": model.means.detach(),
		"scales": model.scales.detach(),
		"quats": model.quats.detach(),
		"opacities": model.opacities.detach()[:, 0],
		"colors": colors,
		"features": torch.rand(n, channels, generator=g, device="cuda"),
		"grad_image": torch.rand(camera.height, camera.width, 3, generator=g, device="cuda"),
		"grad_feature_image": torch.rand(camera.height, camera.width, channels, generator=g, device="cuda"),
	}


def render_ours(leaves: dict[str, torch.Tensor], camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
	means2D, depths, radii, conics = project_gaussians(
		leaves["means"], leaves["scales"], leaves["quats"], leaves["opacities"][:, None], camera.world_to_camera,
		camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
	return rasterize(means2D, depths, radii, conics, leaves["colors"], leaves["opacities"][:, None],
					 camera.width, camera.height, features=leaves["features"])


def render_gsplat(leaves: dict[str, torch.Tensor], camera: Camera, channel_chunk: int = 32,
				  one_call: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
	"""gsplat with our near plane and low pass filter, either one call over RGB + features or one call each."""
	K = torch.tensor([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], device="cuda")

	def call(values: torch.Tensor) -> torch.Tensor:
		out, _, _ = gsplat.rasterization(
			leaves["means"], leaves["quats"], leaves["scales"], leaves["opacities"], values,
			camera.world_to_camera[None], K[None], camera.width, camera.height,
			near_plane=NEAR_PLANE, eps2d=LOW_PASS, channel_chunk=channel_chunk)
		return out[0]

	if one_call:
		out = call(torch.cat([leaves["colors"], leaves["features"]], dim=1))
		return out[..., :3], out[..., 3:]
	return call(leaves["colors"]), call(leaves["features"])


def render_with_grads(render, inputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
	"""One render and backward through `render`, returning both images and the gradient of every leaf."""
	leaves = {name: inputs[name].clone().requires_grad_(True) for name in LEAVES}
	image, feature_image = render(leaves, **kwargs)
	torch.autograd.backward((image, feature_image), (inputs["grad_image"], inputs["grad_feature_image"]))
	out = {"image": image.detach(), "feature_image": feature_image.detach()}
	for name, leaf in leaves.items():
		assert leaf.grad is not None, f"{name}: no gradient"
		out[f"grad_{name}"] = leaf.grad
	return out


def stats(got: torch.Tensor, want: torch.Tensor) -> str:
	diff = (got - want).abs().flatten().float()
	p99 = torch.quantile(diff[torch.randperm(len(diff), device=diff.device)[:2**24]], 0.99)
	rel_l2 = (got - want).norm() / want.norm().clamp_min(1e-30)
	return f"max {diff.max().item():.2e}  p99 {p99.item():.2e}  rel L2 {rel_l2.item():.2e}"


def assert_agree(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
	"""Refuses to let a timing run start when gsplat and ours disagree on one output or gradient."""
	if name.endswith(OUTPUTS):
		assert_image_close(got, want)
	else:
		assert_grads_close((name,), [got], [want])


def main() -> None:
	parser = argparse.ArgumentParser(description="Check ours and gsplat agree on one view, then time them")
	parser.add_argument("scene", help="COLMAP scene folder")
	parser.add_argument("checkpoint", help="model.pt written by save_checkpoint")
	parser.add_argument("--view", type=int, default=0, help="index into the training cameras")
	parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
	args = parser.parse_args()

	dataset = ColmapDataset(args.scene)
	model = load_model(dataset, args.checkpoint)
	camera = dataset.train_cameras[args.view]
	print(f"gsplat {gsplat.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
	print(f"{model.means.shape[0]} gaussians, view {camera.name} {camera.width}x{camera.height}")

	for channels in args.channels:
		inputs = shared_inputs(model, camera, channels)
		ours = render_with_grads(render_ours, inputs, camera=camera)
		theirs = render_with_grads(render_gsplat, inputs, camera=camera)
		for name in ours:
			print(f"D={channels} {name:24s} {stats(theirs[name], ours[name])}")
		for name in ours:
			assert_agree(f"D={channels} {name}", theirs[name], ours[name])
		print(f"D={channels} agrees")


if __name__ == "__main__":
	main()
