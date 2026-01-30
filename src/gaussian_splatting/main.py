import sys
import torch
import random
from pathlib import Path

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.utils.dataset import ColmapDataset, RawImage
from gaussian_splatting.utils.gaussian import GaussianModel


def render(camera, gaussians):
	# TODO: actually write the CUDA rasterizer
	raise NotImplementedError("CUDA rasterizer not yet implemented")


def train(data_path: str) -> None:
	dataset: ColmapDataset = ColmapDataset(data_path)
	model: GaussianModel = GaussianModel(dataset.point_cloud)
	params: TrainingParams = TrainingParams()

	optimizer = torch.optim.Adam(params=model.get_optimizer_params(), eps=1e-15)

	for _ in range(params.iterations):
		image: RawImage = random.choice(dataset.images)
		gt_image: torch.Tensor = image.image

		rendered_image = render(image, model)

		loss: torch.Tensor = torch.nn.functional.mse_loss(rendered_image, gt_image)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise ValueError("Missing arguments. Usage: python main.py <path to colmap scene>")
	train(sys.argv[1])
