import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.rasterizer import project_gaussians, rasterize
from gaussian_splatting.utils.dataset import Camera, ColmapDataset
from gaussian_splatting.utils.gaussian import GaussianModel
from gaussian_splatting.utils.loss import l1_loss, psnr, ssim


def render(camera: Camera, gaussians: GaussianModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	means2D, depths, radii, conics = project_gaussians(
		gaussians.means, gaussians.scales, gaussians.quats, camera.world_to_camera,
		camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height
	)
	if means2D.requires_grad:
		means2D.retain_grad()
	image = rasterize(
		means2D, depths, radii, conics, gaussians.colors(camera.center), gaussians.opacities,
		camera.width, camera.height
	)
	return image, means2D, radii


def position_lr(params: TrainingParams, extent: float, iteration: int) -> float:
	"""Exponential decay from init to final over training, both scaled by the scene size."""
	t = min(iteration / params.iterations, 1.0)
	return (params.position_lr_init * extent) ** (1 - t) * (params.position_lr_final * extent) ** t


@torch.no_grad()
def evaluate(dataset: ColmapDataset, model: GaussianModel) -> float:
	"""Mean PSNR over the held out cameras."""
	scores = [psnr(render(camera, model)[0], camera.image.float() / 255.0)
			  for camera in dataset.test_cameras]
	return torch.stack(scores).mean().item()


def save_checkpoint(output_dir: Path, model: GaussianModel, iteration: int, test_psnr: float) -> None:
	torch.save(
		{
			"params": {name: p.detach().cpu() for name, p in model.params.items()},
			"active_sh_degree": model.active_sh_degree,
			"iteration": iteration,
			"test_psnr": test_psnr,
		},
		output_dir / "model.pt",
	)


def save_comparison(output_dir: Path, iteration: int, rendered: torch.Tensor, gt: torch.Tensor) -> None:
	images = [(t.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for t in (rendered, gt)]
	Image.fromarray(np.concatenate(images, axis=1)).save(output_dir / f"iter_{iteration:06d}.png")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train 3D gaussian splats on a COLMAP scene")
	defaults = TrainingParams()
	parser.add_argument("data_path", help="COLMAP scene laid out as <path>/images and <path>/sparse/0")
	parser.add_argument("--iterations", type=int, default=defaults.iterations)
	parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
	parser.add_argument("--max-width", type=int, default=1600, help="downscale images wider than this")
	parser.add_argument("--test-every", type=int, default=8, help="hold out every Nth image")
	parser.add_argument("--output", type=Path, default=None, help="defaults to outputs/training_<timestamp>")
	parser.add_argument("--seed", type=int, default=None)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)

	print(f"Loading dataset from {args.data_path}")
	dataset = ColmapDataset(args.data_path, max_width=args.max_width, test_every=args.test_every)
	print(f"Loaded {len(dataset.train_cameras)} train / {len(dataset.test_cameras)} test images and {len(dataset.point_cloud.points)} points")

	model = GaussianModel(dataset.point_cloud)
	print(f"Initialized {model.means.shape[0]} Gaussians, scene extent {dataset.extent:.2f}")

	params = TrainingParams()
	params.iterations = args.iterations
	params.eval_interval = args.eval_interval
	optimizer = torch.optim.Adam(model.get_optimizer_params(params, dataset.extent), eps=1e-15)

	output_dir = args.output or Path("outputs") / f"training_{int(time.time())}"
	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"Saving to {output_dir}")
	print(f"Starting training for {params.iterations} iterations...")

	epoch: list[Camera] = []
	times: list[float] = []
	best_psnr: float = float("-inf")
	for iteration in range(1, params.iterations + 1):
		start: float = time.time()

		if not epoch:
			epoch = random.sample(dataset.train_cameras, len(dataset.train_cameras))
		camera: Camera = epoch.pop()

		for group in optimizer.param_groups:
			if group["name"] == "means":
				group["lr"] = position_lr(params, dataset.extent, iteration)
		if iteration % params.sh_degree_interval == 0:
			model.active_sh_degree = min(model.active_sh_degree + 1, model.max_sh_degree)

		gt_image = camera.image.float() / 255.0
		rendered_image, means2D, radii = render(camera, model)

		loss = (
			(1 - params.lambda_dssim) * l1_loss(rendered_image, gt_image)
			+ params.lambda_dssim * (1 - ssim(rendered_image, gt_image))
		)
		loss.backward()

		with torch.no_grad():
			if iteration < params.densify_until_iter and means2D.grad is not None:
				model.add_densification_stats(means2D.grad, radii, camera.width, camera.height)
				if iteration > params.densify_from_iter and iteration % params.densify_interval == 0:
					max_screen = params.max_screen_size if iteration > params.opacity_reset_interval else None
					model.densify_and_prune(optimizer, params, dataset.extent, max_screen)
				if iteration % params.opacity_reset_interval == 0:
					model.reset_opacities(optimizer)

			optimizer.step()
			optimizer.zero_grad(set_to_none=True)

		times.append(time.time() - start)
		if iteration % 100 == 0:
			print(f"Iteration {iteration}/{params.iterations}, Loss: {loss.item():.6f}, "
				  f"Gaussians: {model.means.shape[0]}, Avg. Time: {np.mean(times) * 1000:.1f}ms")
			times = []
		if iteration % params.eval_interval == 0 or iteration == params.iterations:
			test_psnr = evaluate(dataset, model)
			print(f"Iteration {iteration}: test PSNR {test_psnr:.2f} dB")
			save_comparison(output_dir, iteration, rendered_image, gt_image)
			if test_psnr > best_psnr:
				best_psnr = test_psnr
				save_checkpoint(output_dir, model, iteration, test_psnr)

	print(f"Training completed, best test PSNR {best_psnr:.2f} dB. Output in {output_dir}")


if __name__ == "__main__":
	train(parse_args())
