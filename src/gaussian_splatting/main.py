import sys
import time
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.utils.dataset import ColmapDataset, RawImage
from gaussian_splatting.utils.gaussian import GaussianModel
from gaussian_splatting.rasterizer import rasterize


def render(camera: RawImage, gaussians: GaussianModel) -> torch.Tensor:
	rendered = rasterize(
		means3D=gaussians.xyz,
		scales=gaussians.scales,
		quaternions=gaussians.quaternions,
		opacities=gaussians.opacities,
		colors=gaussians.rgb,
		world_to_cam_matrix=camera.world_to_camera_matrix,
		focal_x=float(camera.focal_x),
		focal_y=float(camera.focal_y),
		c_x=float(camera.c_x),
		c_y=float(camera.c_y),
		image_width=camera.width,
		image_height=camera.height
	)
	return rendered


def save_images(output_dir: Path, iteration: int, rendered: torch.Tensor, gt: torch.Tensor):
	rendered_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
	gt_np = (gt.detach().cpu().numpy() * 255).astype(np.uint8)

	rendered_img = Image.fromarray(rendered_np)
	gt_img = Image.fromarray(gt_np)

	rendered_img.save(output_dir / f"iter_{iteration:06d}_rendered.png")
	gt_img.save(output_dir / f"iter_{iteration:06d}_gt.png")

	comparison = np.concatenate([rendered_np, gt_np], axis=1)
	comparison_img = Image.fromarray(comparison)
	comparison_img.save(output_dir / f"iter_{iteration:06d}_comparison.png")


def train(data_path: str) -> None:
	print(f"Loading dataset from {data_path}")
	dataset: ColmapDataset = ColmapDataset(data_path)
	print(f"Loaded {len(dataset.images)} images and {len(dataset.point_cloud.points)} points")

	model: GaussianModel = GaussianModel(dataset.point_cloud)
	print(f"Initialized {model.xyz.shape[0]} Gaussians")

	params: TrainingParams = TrainingParams()
	optimizer = torch.optim.Adam(params=model.get_optimizer_params(), eps=1e-15)

	output_dir: Path = Path("outputs") / f"training_{int(time.time())}"
	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"Saving images to {output_dir}")

	print(f"Starting training for {params.iterations} iterations...")

	times: list[float] = []
	for iteration in range(params.iterations):
		start: float = time.time()
		image: RawImage = random.choice(dataset.images)
		gt_image: torch.Tensor = image.image

		rendered_image = render(image, model)

		loss: torch.Tensor = torch.nn.functional.mse_loss(rendered_image, gt_image)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		end: float = time.time()
		times.append(end - start)

		if iteration % 100 == 0:
			print(f"Iteration {iteration}/{params.iterations}, Loss: {loss.item():.6f}, Avg. Time: {np.mean(times):.4f}s")
			save_images(output_dir, iteration, rendered_image, gt_image)
			times = []

	print(f"Training completed. Images saved to {output_dir}")


if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise ValueError("Missing arguments. Usage: python main.py <path to colmap scene>")
	train(sys.argv[1])
