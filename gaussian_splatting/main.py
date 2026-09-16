import random
import sys
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


def save_comparison(output_dir: Path, iteration: int, rendered: torch.Tensor, gt: torch.Tensor) -> None:
	images = [(t.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for t in (rendered, gt)]
	Image.fromarray(np.concatenate(images, axis=1)).save(output_dir / f"iter_{iteration:06d}.png")


def train(data_path: str) -> None:
	print(f"Loading dataset from {data_path}")
	dataset: ColmapDataset = ColmapDataset(data_path)
	print(f"Loaded {len(dataset.train_cameras)} train / {len(dataset.test_cameras)} test images and {len(dataset.point_cloud.points)} points")

	model: GaussianModel = GaussianModel(dataset.point_cloud)
	print(f"Initialized {model.means.shape[0]} Gaussians, scene extent {dataset.extent:.2f}")

	params: TrainingParams = TrainingParams()
	optimizer = torch.optim.Adam(model.get_optimizer_params(params, dataset.extent), eps=1e-15)

	output_dir: Path = Path("outputs") / f"training_{int(time.time())}"
	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"Saving to {output_dir}")
	print(f"Starting training for {params.iterations} iterations...")

	epoch: list[Camera] = []
	times: list[float] = []
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

		gt_image: torch.Tensor = camera.image.float() / 255.0
		rendered_image, means2D, radii = render(camera, model)

		loss: torch.Tensor = (
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
			print(f"Iteration {iteration}: test PSNR {evaluate(dataset, model):.2f} dB")
			save_comparison(output_dir, iteration, rendered_image, gt_image)

	torch.save({name: p.detach().cpu() for name, p in model.params.items()}, output_dir / "model.pt")
	print(f"Training completed. Output in {output_dir}")


if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise ValueError("Missing arguments. Usage: python main.py <path to colmap scene>")
	train(sys.argv[1])
