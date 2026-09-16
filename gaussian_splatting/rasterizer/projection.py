import torch

NEAR_PLANE = 0.2
LOW_PASS = 0.3
JACOBIAN_FOV_SCALE = 1.3


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
	"""[N, 4] unit quaternions (w, x, y, z) -> [N, 3, 3] rotation matrices."""
	w, x, y, z = quats.unbind(-1)
	return torch.stack([
		1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
		2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
		2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
	], dim=-1).reshape(-1, 3, 3)


def project_gaussians(
	means: torch.Tensor,
	scales: torch.Tensor,
	quats: torch.Tensor,
	world_to_camera: torch.Tensor,
	fx: float,
	fy: float,
	cx: float,
	cy: float,
	width: int,
	height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	means [N, 3], scales [N, 3], quats [N, 4] unit (w, x, y, z), world_to_camera [4, 4].

	Returns:
		means2D [N, 2]: centers in pixels.
		depths [N]: camera-space z, used only to sort.
		radii [N]: footprint radius in pixels, 0 for culled gaussians. Not differentiable.
		conics [N, 3]: inverse 2D covariance as (xx, xy, yy).
	"""
	# torch.compile's inductor knows that, more often than not, cuBLAS is faster than any generated
	# kernel. So, it defaults to calling it. However, cuBLAS is rather slow for our 3x3 GEMMs due
	# to overhead. This thus doesn't remotely approach the handwritten CUDA implementation in
	# terms of runtime. But, we implement it in torch to save sanity on the backward pass.
	R = world_to_camera[:3, :3]
	p = means @ R.T + world_to_camera[:3, 3]
	depths = p[:, 2]
	x, y, z = p[:, 0], p[:, 1], depths.clamp_min(NEAR_PLANE)
	means2D = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)

	M = quat_to_rotmat(quats) * scales[:, None, :]
	cov3D = M @ M.transpose(1, 2)

	z_inv = 1 / (z + 1e-6)
	limit_x = JACOBIAN_FOV_SCALE * (0.5 * width / fx)
	limit_y = JACOBIAN_FOV_SCALE * (0.5 * height / fy)
	jx = (x * z_inv).clamp(-limit_x, limit_x) * z
	jy = (y * z_inv).clamp(-limit_y, limit_y) * z
	zeros = torch.zeros_like(z)
	J = torch.stack([
		fx * z_inv, zeros, -fx * jx * z_inv * z_inv,
		zeros, fy * z_inv, -fy * jy * z_inv * z_inv,
	], dim=-1).reshape(-1, 2, 3)
	JW = J @ R
	cov2D = JW @ cov3D @ JW.transpose(1, 2)
	a = cov2D[:, 0, 0] + LOW_PASS
	b = cov2D[:, 0, 1]
	c = cov2D[:, 1, 1] + LOW_PASS

	det = a * c - b * b
	conics = torch.stack([c, -b, a], dim=-1) / torch.where(det > 0, det, 1.0)[:, None]

	with torch.no_grad():
		trace = a + c
		radii = 3 * torch.sqrt(0.5 * (trace + torch.sqrt((trace * trace - 4 * det).clamp_min(0))))
		u, v = means2D.unbind(-1)
		visible = (
			(depths > NEAR_PLANE)
			& (u - radii <= width) & (u + radii >= 0) & (v - radii <= height) & (v + radii >= 0)
			& (det > 0)
		)
		radii = torch.where(visible, radii, 0.0)

	return means2D, depths, radii, conics
