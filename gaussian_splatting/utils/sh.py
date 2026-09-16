import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (
	-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
	-0.4570457994644658, 1.445305721320277, -0.5900435899266435,
)


def num_sh_coeffs(degree: int) -> int:
	"""Counts the spherical harmonic coefficients up to and including a degree.

	Args:
		degree: Highest spherical harmonic degree, from 0 to 3.

	Returns:
		The coefficient count, which is (degree + 1) squared.
	"""
	return (degree + 1) ** 2


def sh_basis(dirs: torch.Tensor) -> torch.Tensor:
	"""Evaluates every spherical harmonic basis function for a set of directions.

	Args:
		dirs: [N, 3] unit direction vectors.

	Returns:
		[N, 16] value of each basis function of degrees 0 through 3.
	"""
	x, y, z = dirs.unbind(-1)
	xx, yy, zz = x * x, y * y, z * z
	xy, yz, xz = x * y, y * z, x * z
	return torch.stack([
		torch.full_like(x, C0),
		-C1 * y, C1 * z, -C1 * x,
		C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
		C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
		C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
		C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy),
	], dim=-1)


def eval_sh(sh: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
	"""Turns learned spherical harmonic weights into an RGB color per gaussian.

	Each channel is a dot product of the weights with the basis values for that gaussian's
	viewing direction, so the color changes with where the camera is.

	Args:
		sh: [N, K, 3] learned weights ordered by increasing degree.
		dirs: [N, 3] unit vectors from the camera to each gaussian.
		degree: Highest degree to use, so every band above it is ignored.

	Returns:
		[N, 3] RGB, offset by 0.5 so all zero weights mean mid gray, clamped to non-negative.
	"""
	k = num_sh_coeffs(degree)
	rgb = torch.einsum("nk,nkc->nc", sh_basis(dirs)[:, :k], sh[:, :k])
	return (rgb + 0.5).clamp_min(0.0)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
	"""Inverts eval_sh at degree 0, which is how the model is initialized from point cloud colors.

	Args:
		rgb: [N, 3] colors in [0, 1].

	Returns:
		[N, 3] degree 0 weights that make eval_sh return rgb from every direction.
	"""
	return (rgb - 0.5) / C0
