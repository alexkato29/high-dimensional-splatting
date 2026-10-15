import argparse
import itertools
import json
import statistics
import subprocess
import time
from pathlib import Path

import gsplat
import torch
import torch.nn.functional as F

from gaussian_splatting.rasterizer import _get_rasterizer, project_gaussians, rasterize
from gaussian_splatting.rasterizer.projection import LOW_PASS, NEAR_PLANE
from gaussian_splatting.utils.dataset import Camera, ColmapDataset
from gaussian_splatting.utils.gaussian import GaussianModel
from tests.test_golden import environment
from tests.test_properties import FLIP_MAX_ERROR
from tests.test_render import assert_grads_close, assert_image_close

OUTPUTS = ("image", "feature_image")
LEAVES = ("means", "scales", "quats", "opacities", "colors", "features")


def load_model(dataset: ColmapDataset, checkpoint: str) -> GaussianModel:
	state = torch.load(checkpoint, map_location="cpu")
	model = GaussianModel(dataset.point_cloud)
	model.params = {name: torch.nn.Parameter(t.cuda().contiguous()) for name, t in state["params"].items()}
	model.active_sh_degree = state["active_sh_degree"]
	return model


def shared_inputs(model: GaussianModel, camera: Camera, channels: int, seed: int = 0) -> dict[str, torch.Tensor]:
	g = torch.Generator(device="cuda").manual_seed(seed)
	with torch.no_grad():
		_, _, radii, _ = project_gaussians(
			model.means, model.scales, model.quats, model.opacities, camera.world_to_camera,
			camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
		colors = model.colors(camera.center, radii)
	n = model.means.shape[0]
	# TODO: Replace this with actual feature vectors
	inputs = {
		"means": model.means.detach(),
		"scales": model.scales.detach(),
		"quats": model.params["quats"].detach(),
		"opacities": model.opacities.detach()[:, 0],
		"colors": colors,
		"features": torch.rand(n, channels, generator=g, device="cuda"),
		"grad_image": torch.rand(camera.height, camera.width, 3, generator=g, device="cuda"),
		"grad_feature_image": torch.rand(camera.height, camera.width, channels, generator=g, device="cuda"),
	}

	# 0 channels renders RGB alone.
	if channels == 0:
		del inputs["features"], inputs["grad_feature_image"]
	return inputs


def render_ours(leaves: dict[str, torch.Tensor], camera: Camera) -> tuple[torch.Tensor, ...]:
	# gsplat normalizes quats inside its projection, though we do not...
	means2D, depths, radii, conics = project_gaussians(
		leaves["means"], leaves["scales"], F.normalize(leaves["quats"], dim=-1), leaves["opacities"][:, None],
		camera.world_to_camera, camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
	out = rasterize(means2D, depths, radii, conics, leaves["colors"], leaves["opacities"][:, None],
					camera.width, camera.height, features=leaves.get("features"))
	return out if isinstance(out, tuple) else (out,)


def render_gsplat(leaves: dict[str, torch.Tensor], camera: Camera, channel_chunk: int = 32,
				  one_call: bool = False, packed: bool = True) -> tuple[torch.Tensor, ...]:
	K = torch.tensor([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], device="cuda")

	def call(values: torch.Tensor) -> torch.Tensor:
		out, _, _ = gsplat.rasterization(
			leaves["means"], leaves["quats"], leaves["scales"], leaves["opacities"], values,
			camera.world_to_camera[None], K[None], camera.width, camera.height,
			near_plane=NEAR_PLANE, eps2d=LOW_PASS, channel_chunk=channel_chunk, packed=packed)
		return out[0]

	# `one_call` is supported just to ensure that our two individual rasters (of features and rgb separately)
	# does not lose to a single unified render case in gsplat.
	if "features" not in leaves:
		return (call(leaves["colors"]),)
	if one_call:
		out = call(torch.cat([leaves["colors"], leaves["features"]], dim=1))
		return out[..., :3], out[..., 3:]
	return call(leaves["colors"]), call(leaves["features"])


def render_with_grads(render, inputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
	leaves = {name: inputs[name].clone().requires_grad_(True) for name in LEAVES if name in inputs}
	images = render(leaves, **kwargs)
	torch.autograd.backward(images, [inputs[f"grad_{name}"] for name in OUTPUTS[:len(images)]])
	out = {name: image.detach() for name, image in zip(OUTPUTS, images)}
	for name, leaf in leaves.items():
		assert leaf.grad is not None, f"{name}: no gradient"
		out[f"grad_{name}"] = leaf.grad
	return out


def stats(got: torch.Tensor, want: torch.Tensor) -> str:
	diff = (got - want).abs().flatten().float()
	# quantile caps at 16M elements, and a permutation of a wide feature tensor alone is over 4 GB.
	sample = diff if len(diff) <= 2**24 else diff[torch.randint(len(diff), (2**24,), device=diff.device)]
	p99 = torch.quantile(sample, 0.99)
	rel_l2 = diff.norm() / want.norm().clamp_min(1e-30)
	return f"max {diff.max().item():.2e}  p99 {p99.item():.2e}  rel L2 {rel_l2.item():.2e}"


def assert_agree(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
	if name.endswith(OUTPUTS):
		assert_image_close(got, want, max_error=FLIP_MAX_ERROR)
	else:
		assert_grads_close((name,), [got], [want])


def power_cap_us() -> int:
	out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True, check=True).stdout
	for line in out.splitlines():
		if "SW Power Capping" in line and line.strip().endswith("us"):
			return int(line.split()[-2])
	return -1


def sm_clock_mhz() -> int:
	out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
						 capture_output=True, text=True, check=True).stdout
	return int(out.split()[0])


def time_ms(fn, before=None, iters: int = 50, warmup: int = 10) -> dict[str, float]:
	for _ in range(warmup):
		if before:
			before()
		fn()
	torch.cuda.synchronize()
	time.sleep(0.5)
	cap, clock = power_cap_us(), sm_clock_mhz()
	times = []
	for _ in range(iters):
		if before:
			before()
		torch.cuda.synchronize()
		start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
		start.record()
		fn()
		end.record()
		end.synchronize()
		times.append(start.elapsed_time(end))
		time.sleep(max(0.01, 1.5 * times[-1] / 1000))
	return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times),
			"power_capped_us": power_cap_us() - cap, "sm_clock_mhz_start": clock, "sm_clock_mhz_end": sm_clock_mhz()}


def time_render(render, inputs: dict[str, torch.Tensor], label: str, **kwargs) -> dict[str, dict[str, float]]:
	leaves = {name: inputs[name].clone().requires_grad_(True) for name in LEAVES if name in inputs}
	grads = [inputs[f"grad_{name}"] for name in OUTPUTS if f"grad_{name}" in inputs]
	state = {}

	def forward():
		state["out"] = render(leaves, **kwargs)

	def backward():
		torch.autograd.backward(state["out"], grads)

	def reset():
		for leaf in leaves.values():
			leaf.grad = None
		forward()

	with torch.cuda.nvtx.range(f"{label} fwd"):
		fwd = time_ms(forward)
	with torch.cuda.nvtx.range(f"{label} bwd"):
		bwd = time_ms(backward, before=reset)
	return {"fwd": fwd, "bwd": bwd}


def ours_pairs(inputs: dict[str, torch.Tensor], camera: Camera) -> int:
	with torch.no_grad():
		means2D, depths, radii, conics = project_gaussians(
			inputs["means"], inputs["scales"], F.normalize(inputs["quats"], dim=-1), inputs["opacities"][:, None],
			camera.world_to_camera, camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
		out = _get_rasterizer().rasterize(means2D, depths, radii, conics, inputs["colors"],
										  inputs["opacities"][:, None], torch.zeros(3, device="cuda"),
										  camera.width, camera.height)
	return out[3].numel()


def gsplat_pairs(inputs: dict[str, torch.Tensor], camera: Camera, packed: bool) -> int:
	K = torch.tensor([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], device="cuda")
	with torch.no_grad():
		_, _, meta = gsplat.rasterization(
			inputs["means"], inputs["quats"], inputs["scales"], inputs["opacities"], inputs["colors"],
			camera.world_to_camera[None], K[None], camera.width, camera.height,
			near_plane=NEAR_PLANE, eps2d=LOW_PASS, packed=packed)
	return meta["flatten_ids"].numel()


def agrees(theirs: dict[str, torch.Tensor], ours: dict[str, torch.Tensor], label: str) -> bool:
	for name, want in ours.items():
		print(f"{label} {name:24s} {stats(theirs[name], want.cuda())}")
	try:
		for name, want in ours.items():
			assert_agree(name, theirs[name], want.cuda())
	except AssertionError as error:
		print(f"{label} DISAGREES, not timed: {error}")
		return False
	return True


def main() -> None:
	parser = argparse.ArgumentParser(description="Check ours and gsplat agree on one view, then time them")
	parser.add_argument("scene", help="COLMAP scene folder")
	parser.add_argument("checkpoint", help="model.pt written by save_checkpoint")
	parser.add_argument("--view", type=int, default=0, help="index into the training cameras")
	parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128], help="feature widths, 0 for RGB alone")
	parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
	args = parser.parse_args()

	dataset = ColmapDataset(args.scene)
	model = load_model(dataset, args.checkpoint)
	camera = dataset.train_cameras[args.view]
	env = {**environment(), "gsplat": gsplat.__version__, "scene": args.scene, "checkpoint": args.checkpoint,
		   "view": camera.name, "width": camera.width, "height": camera.height, "gaussians": model.means.shape[0]}
	print(env)

	rows = []
	for channels in args.channels:
		inputs = shared_inputs(model, camera, channels)
		# Held on the CPU so the reference does not take memory from either timed run.
		ours = {name: t.cpu() for name, t in render_with_grads(render_ours, inputs, camera=camera).items()}
		rows.append({"impl": "ours", "channels": channels, "config": {}, "pairs": ours_pairs(inputs, camera),
					 **time_render(render_ours, inputs, f"ours D={channels}", camera=camera)})
		print(f"D={channels} ours {rows[-1]['fwd']['median_ms']:.2f} / {rows[-1]['bwd']['median_ms']:.2f} ms")

		chunks, calls = ((32, 64, 128), (False, True)) if channels else ((32,), (False,))
		for packed, chunk, one_call in itertools.product((True, False), chunks, calls):
			config = {"packed": packed, "channel_chunk": chunk, "one_call": one_call}
			label = f"gsplat D={channels} packed={packed} chunk={chunk} one_call={one_call}"
			try:
				theirs = render_with_grads(render_gsplat, inputs, camera=camera, **config)
				agreed = agrees(theirs, ours, label)
				# The gate's copy of gsplat's outputs does not fit beside the timed run at D = 512.
				del theirs
				if not agreed:
					continue
				timing = time_render(render_gsplat, inputs, label, camera=camera, **config)
			except (RuntimeError, ValueError) as error:
				print(f"{label} cannot run: {str(error).splitlines()[0]}")
				continue
			rows.append({"impl": "gsplat", "channels": channels, "config": config,
						 "pairs": gsplat_pairs(inputs, camera, packed), **timing})
			print(f"{label} {rows[-1]['fwd']['median_ms']:.2f} / {rows[-1]['bwd']['median_ms']:.2f} ms")

	args.out.write_text(json.dumps({"environment": env, "rows": rows}, indent=1))
	for channels in args.channels:
		mine = next(r for r in rows if r["impl"] == "ours" and r["channels"] == channels)
		theirs = [r for r in rows if r["impl"] == "gsplat" and r["channels"] == channels]
		if not theirs:
			print(f"D={channels}: no gsplat configuration agreed")
			continue
		for direction in ("fwd", "bwd"):
			best = min(theirs, key=lambda r: r[direction]["median_ms"])
			ratio = best[direction]["median_ms"] / mine[direction]["median_ms"]
			print(f"D={channels} {direction}: ours {mine[direction]['median_ms']:.2f} ms, best gsplat "
				  f"{best[direction]['median_ms']:.2f} ms {best['config']}, {ratio:.2f}x")
	print(f"wrote {args.out}")


if __name__ == "__main__":
	main()
