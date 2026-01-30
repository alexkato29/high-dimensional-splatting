from typing import Any
from scipy.spatial import KDTree
import numpy as np
import torch
import torch.nn as nn
from gaussian_splatting.config import TrainingParams
from gaussian_splatting.utils.dataset import PointCloud


class GaussianModel(nn.Module):
	def __init__(self, point_cloud: PointCloud):
		super().__init__()

		self._xyz: nn.Parameter
		self._scaling_vecs: nn.Parameter
		self._quaternions: nn.Parameter
		self._opacities: nn.Parameter
		# The actual paper uses Spherical Harmonics, but for simplicity I will add that last
		self._rgb: nn.Parameter

		self._initialize_from_point_cloud(point_cloud)

	def _initialize_from_point_cloud(self, point_cloud: PointCloud) -> None:
		num_points: int = len(point_cloud.points)

		# We use SciPy to compute distances between points. We, for now,
		# leave it off the GPU to avoid redundant data transfer overhead
		cpu_xyz: torch.Tensor = torch.tensor(point_cloud.points, dtype=torch.float32)

		dist: torch.Tensor = torch.clamp_min(self._compute_nearest_neighbor_dist(cpu_xyz), 1e-7)
		# This makes shape [d1, d2, ...] -> [[d1.1, d1.2, d1.3], [d2.1, d2.2, d2.3], ...]
		scales: torch.Tensor = torch.log(dist).unsqueeze(-1).repeat(1, 3)
		self._scaling_vecs = nn.Parameter(scales.cuda(), requires_grad=True)

		self._xyz = nn.Parameter(
			cpu_xyz.cuda(),
			requires_grad=True
		)

		self._quaternions = nn.Parameter(
			torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_points, 1).cuda(),
			requires_grad=True
		)

		opacities: torch.Tensor = self._inverse_sigmoid(
			0.1 * torch.ones((num_points, 1), dtype=torch.float32)
		)
		self._opacities = nn.Parameter(opacities.cuda(), requires_grad=True)

		self._rgb = nn.Parameter(
			torch.tensor(point_cloud.colors, dtype=torch.float32).cuda(),
			requires_grad=True
		)

	def _compute_nearest_neighbor_dist(self, cpu_points: torch.Tensor) -> torch.Tensor:
		points: np.ndarray = cpu_points.numpy()
		tree: KDTree = KDTree(points)
		distances, _ = tree.query(points, k=2)
		return torch.tensor(distances[:, 1], dtype=torch.float32)

	def _inverse_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
		return torch.log(x / (1 - x))

	@property
	def xyz(self) -> torch.Tensor:
		return self._xyz

	@property
	def scales(self) -> torch.Tensor:
		return torch.exp(self._scaling_vecs)

	@property
	def quaternions(self) -> torch.Tensor:
		return self._quaternions / torch.norm(self._quaternions, dim=1, keepdim=True)

	@property
	def opacities(self) -> torch.Tensor:
		return torch.sigmoid(self._opacities)

	@property
	def rgb(self) -> torch.Tensor:
		return torch.clamp(self._rgb, 0.0, 1.0)

	def get_optimizer_params(self) -> list[dict[str, Any]]:
		params: TrainingParams = TrainingParams()
		return [
			{'params': [self._xyz], 'lr': params.position_lr, "name": "xyz"},
			{'params': [self._scaling_vecs], 'lr': params.scaling_lr, "name": "scaling_vecs"},
			{'params': [self._quaternions], 'lr': params.rotation_lr, "name": "quaternions"},
			{'params': [self._opacities], 'lr': params.opacity_lr, "name": "opacities"},
			{'params': [self._rgb], 'lr': params.rgb_lr, "name": "rgb"}
		]

	# CRITICAL TODO: I need to be able to split and prune gaussians.
