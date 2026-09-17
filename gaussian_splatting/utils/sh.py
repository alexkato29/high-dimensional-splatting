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


@torch.compile(dynamic=True)
def eval_sh(sh_dc: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
	"""Turns learned spherical harmonic weights into an RGB color per gaussian.

	Each channel is a weighted sum of the basis functions evaluated at that gaussian's viewing
	direction, so the color changes with where the camera is. The sum is written out term by term,
	with every operation pointwise, so torch.compile can fuse it into a single kernel.

	Args:
		sh_dc: [N, 1, 3] degree 0 weights.
		sh_rest: [N, 15, 3] weights for degrees 1 through 3, ordered by increasing degree.
		dirs: [N, 3] unit vectors from the camera to each gaussian.
		degree: Highest degree to use, so every band above it is ignored.

	Returns:
		[N, 3] RGB, offset by 0.5 so all zero weights mean mid gray, clamped to non-negative.
	"""
	rgb = C0 * sh_dc[:, 0] + 0.5
	if degree > 0:
		x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
		rgb = rgb - C1 * y * sh_rest[:, 0] + C1 * z * sh_rest[:, 1] - C1 * x * sh_rest[:, 2]
		if degree > 1:
			xx, yy, zz = x * x, y * y, z * z
			xy, yz, xz = x * y, y * z, x * z
			rgb = (
				rgb
				+ C2[0] * xy * sh_rest[:, 3]
				+ C2[1] * yz * sh_rest[:, 4]
				+ C2[2] * (2 * zz - xx - yy) * sh_rest[:, 5]
				+ C2[3] * xz * sh_rest[:, 6]
				+ C2[4] * (xx - yy) * sh_rest[:, 7]
			)
			if degree > 2:
				rgb = (
					rgb
					+ C3[0] * y * (3 * xx - yy) * sh_rest[:, 8]
					+ C3[1] * xy * z * sh_rest[:, 9]
					+ C3[2] * y * (4 * zz - xx - yy) * sh_rest[:, 10]
					+ C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_rest[:, 11]
					+ C3[4] * x * (4 * zz - xx - yy) * sh_rest[:, 12]
					+ C3[5] * z * (xx - yy) * sh_rest[:, 13]
					+ C3[6] * x * (xx - 3 * yy) * sh_rest[:, 14]
				)
	return rgb.clamp_min(0.0)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
	"""Inverts eval_sh at degree 0, which is how the model is initialized from point cloud colors.

	Args:
		rgb: [N, 3] colors in [0, 1].

	Returns:
		[N, 3] degree 0 weights that make eval_sh return rgb from every direction.
	"""
	return (rgb - 0.5) / C0
