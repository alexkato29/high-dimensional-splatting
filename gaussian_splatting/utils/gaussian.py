from typing import Any, Callable

import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.rasterizer.projection import quat_to_rotmat
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
		self.device = device
		self._reset_stats(n)

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

	def get_optimizer_params(self, params: TrainingParams, extent: float) -> list[dict[str, Any]]:
		return [
			{"params": [self.params["means"]], "lr": params.position_lr_init * extent, "name": "means"},
			{"params": [self.params["scales"]], "lr": params.scaling_lr, "name": "scales"},
			{"params": [self.params["quats"]], "lr": params.rotation_lr, "name": "quats"},
			{"params": [self.params["opacities"]], "lr": params.opacity_lr, "name": "opacities"},
			{"params": [self.params["sh_dc"]], "lr": params.sh_dc_lr, "name": "sh_dc"},
			{"params": [self.params["sh_rest"]], "lr": params.sh_rest_lr, "name": "sh_rest"},
		]

	def _reset_stats(self, n: int) -> None:
		self.max_radii2D = torch.zeros(n, device=self.device)
		self.means2D_grad_accum = torch.zeros(n, 1, device=self.device)
		self.denom = torch.zeros(n, 1, device=self.device)

	def add_densification_stats(self, means2D_grad: torch.Tensor, radii: torch.Tensor,
								width: int, height: int) -> None:
		"""Running average of how hard each visible gaussian is being pulled across the image."""
		visible = radii > 0
		self.max_radii2D = torch.where(visible, torch.maximum(self.max_radii2D, radii), self.max_radii2D)
		scale = torch.tensor([0.5 * width, 0.5 * height], device=means2D_grad.device)
		self.means2D_grad_accum[visible] += (means2D_grad * scale)[visible].norm(dim=-1, keepdim=True)
		self.denom[visible] += 1

	def densify_and_prune(self, optimizer: torch.optim.Optimizer, params: TrainingParams,
						  extent: float, max_screen_size: int | None) -> None:
		grads = torch.nan_to_num(self.means2D_grad_accum / self.denom.clamp_min(1)).squeeze(-1)

		self._clone(optimizer, grads, params, extent)
		self._split(optimizer, grads, params, extent)

		prune = (self.opacities < params.min_opacity).squeeze(-1)
		if max_screen_size is not None:
			prune |= self.max_radii2D > max_screen_size
			prune |= self.scales.max(dim=1).values > 0.1 * extent
		self._prune(optimizer, ~prune)

	def reset_opacities(self, optimizer: torch.optim.Optimizer) -> None:
		"""
		Periodically pushes every opacity near zero. Gaussians the render actually needs recover within
		a few hundred iterations, floaters near the cameras never do and get pruned.
		"""
		reset = torch.logit(self.opacities.clamp(1e-6, 0.01))
		self._replace_params(
			optimizer,
			lambda name, t: reset if name == "opacities" else t,
			lambda name, t: torch.zeros_like(t) if name == "opacities" else t,
		)

	def _clone(self, optimizer: torch.optim.Optimizer, grads: torch.Tensor,
			   params: TrainingParams, extent: float) -> None:
		"""Under-reconstructed regions. The gaussian is small, so copy it and let the two drift apart."""
		selected = (grads >= params.densify_grad_threshold) & (
			self.scales.max(dim=1).values <= params.percent_dense * extent
		)
		self._append(optimizer, {name: p[selected] for name, p in self.params.items()})

	def _split(self, optimizer: torch.optim.Optimizer, grads: torch.Tensor,
			   params: TrainingParams, extent: float, into: int = 2) -> None:
		"""Over-reconstructed regions. One big gaussian becomes two smaller ones sampled inside it."""
		padded = torch.zeros(self.means.shape[0], device=grads.device)
		padded[:grads.shape[0]] = grads
		selected = (padded >= params.densify_grad_threshold) & (
			self.scales.max(dim=1).values > params.percent_dense * extent
		)

		stds = self.scales[selected].repeat(into, 1)
		offsets = torch.normal(torch.zeros_like(stds), stds).unsqueeze(-1)
		rotations = quat_to_rotmat(self.quats[selected]).repeat(into, 1, 1)

		new = {name: p[selected].repeat(into, *([1] * (p.dim() - 1))) for name, p in self.params.items()}
		new["means"] = new["means"] + torch.bmm(rotations, offsets).squeeze(-1)
		new["scales"] = torch.log(stds / (0.8 * into))
		self._append(optimizer, new)

		split_count = int(selected.sum())
		keep = torch.ones(self.means.shape[0], dtype=torch.bool, device=grads.device)
		keep[:selected.shape[0]] = ~selected
		assert self.means.shape[0] == selected.shape[0] + into * split_count
		self._prune(optimizer, keep)

	def _append(self, optimizer: torch.optim.Optimizer, extension: dict[str, torch.Tensor]) -> None:
		self._replace_params(
			optimizer,
			lambda name, t: torch.cat([t, extension[name]], dim=0),
			lambda name, t: torch.cat([t, torch.zeros_like(extension[name])], dim=0),
		)
		self._reset_stats(self.means.shape[0])

	def _prune(self, optimizer: torch.optim.Optimizer, keep: torch.Tensor) -> None:
		self._replace_params(optimizer, lambda name, t: t[keep], lambda name, t: t[keep])
		self.max_radii2D = self.max_radii2D[keep]
		self.means2D_grad_accum = self.means2D_grad_accum[keep]
		self.denom = self.denom[keep]

	def _replace_params(self, optimizer: torch.optim.Optimizer,
						param_fn: Callable[[str, torch.Tensor], torch.Tensor],
						state_fn: Callable[[str, torch.Tensor], torch.Tensor]) -> None:
		"""
		Swaps every parameter for a new tensor and carries its Adam state along. Adam keeps one moment
		per element, so a parameter that gains or loses rows without the same surgery on exp_avg and
		exp_avg_sq would step new gaussians with a stale momentum belonging to a different gaussian.
		"""
		for group in optimizer.param_groups:
			name = group["name"]
			old = group["params"][0]
			state = optimizer.state.pop(old, None)
			new = torch.nn.Parameter(param_fn(name, old.detach()).contiguous())
			if state is not None:
				state["exp_avg"] = state_fn(name, state["exp_avg"])
				state["exp_avg_sq"] = state_fn(name, state["exp_avg_sq"])
				optimizer.state[new] = state
			group["params"][0] = new
			self.params[name] = new
