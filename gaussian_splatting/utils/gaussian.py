from typing import Any

import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.utils.dataset import PointCloud
from gaussian_splatting.utils.sh import eval_sh, num_sh_coeffs, rgb_to_sh


class GaussianModel:
	"""
	The learnable gaussians. Each parameter is stored unconstrained, so Adam can step it freely, and
	mapped through an activation on read that enforces what it must satisfy: scales positive (exp),
	opacity in (0, 1) (sigmoid), rotation a unit quaternion (normalize).

	All parameters live in one dict so densification can add or remove gaussians by applying the same
	index or concatenation to every entry, and to the matching optimizer state, without naming each one.
	"""
	def __init__(self, point_cloud: PointCloud, max_sh_degree: int = 3, device: str = "cuda"):
		means = torch.from_numpy(point_cloud.points)
		n = len(means)

		# Each gaussian starts as an isotropic blob whose standard deviation is the RMS distance to its 3
		# nearest neighbors, so neighboring splats overlap enough to cover the surface without holes.
		# k=4 because each point's nearest neighbor is itself.
		dists, _ = KDTree(point_cloud.points).query(point_cloud.points, k=4)
		dist2 = torch.from_numpy((dists[:, 1:] ** 2).mean(axis=1)).float().clamp_min(1e-7)
		scales = torch.log(torch.sqrt(dist2)).unsqueeze(-1).repeat(1, 3)

		# Degree 0 reproduces the point cloud's colors exactly, higher degrees start at zero.
		sh = torch.zeros(n, num_sh_coeffs(max_sh_degree), 3)
		sh[:, 0] = rgb_to_sh(torch.from_numpy(point_cloud.colors))

		self.params: dict[str, torch.nn.Parameter] = {
			name: torch.nn.Parameter(value.contiguous().to(device))
			for name, value in {
				"means": means,
				"scales": scales,
				"quats": torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1),
				"opacities": torch.logit(torch.full((n, 1), 0.1)),
				# Split so the higher degrees can learn more slowly than the base color.
				"sh_dc": sh[:, :1],
				"sh_rest": sh[:, 1:],
			}.items()
		}
		self.max_sh_degree = max_sh_degree
		# Training starts at degree 0 (a plain color) and raises this over time, so the optimizer gets
		# geometry and base color right before it can explain errors away with view-dependent effects.
		self.active_sh_degree = 0

	@property
	def means(self) -> torch.Tensor:
		return self.params["means"]

	@property
	def scales(self) -> torch.Tensor:
		return torch.exp(self.params["scales"])

	@property
	def quats(self) -> torch.Tensor:
		return F.normalize(self.params["quats"], dim=-1)

	@property
	def opacities(self) -> torch.Tensor:
		return torch.sigmoid(self.params["opacities"])

	def colors(self, camera_center: torch.Tensor) -> torch.Tensor:
		"""[N, 3] RGB of every gaussian as seen from camera_center."""
		dirs = F.normalize(self.means - camera_center, dim=-1)
		sh = torch.cat([self.params["sh_dc"], self.params["sh_rest"]], dim=1)
		return eval_sh(sh, dirs, self.active_sh_degree)

	def get_optimizer_params(self) -> list[dict[str, Any]]:
		lrs = TrainingParams()
		return [
			{"params": [self.params["means"]], "lr": lrs.position_lr, "name": "means"},
			{"params": [self.params["scales"]], "lr": lrs.scaling_lr, "name": "scales"},
			{"params": [self.params["quats"]], "lr": lrs.rotation_lr, "name": "quats"},
			{"params": [self.params["opacities"]], "lr": lrs.opacity_lr, "name": "opacities"},
			{"params": [self.params["sh_dc"]], "lr": lrs.sh_dc_lr, "name": "sh_dc"},
			{"params": [self.params["sh_rest"]], "lr": lrs.sh_rest_lr, "name": "sh_rest"},
		]
