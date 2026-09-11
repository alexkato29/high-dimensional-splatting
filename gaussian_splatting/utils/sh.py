"""
Spherical harmonics (SH): how a gaussian's color changes with the direction it is viewed from.

A view-dependent color is a function on the sphere of directions. SH are the sphere's version of a
Fourier series: fixed basis functions of increasing angular frequency, where a function is stored as
one weight per basis function. Degree l adds 2l + 1 functions, so degrees 0-3 give 1 + 3 + 5 + 7 = 16
weights per color channel. The weights are learned; the basis functions below are fixed math.

The constants scale each basis function to unit energy over the sphere. They are derived, not tuned:
the constant function c needs c^2 * 4pi = 1 (the sphere's area is 4pi), so c = 1 / (2 sqrt(pi)) = C0.
Polynomials, signs and ordering match the reference implementation, so trained models interchange.
"""
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (
	-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
	-0.4570457994644658, 1.445305721320277, -0.5900435899266435,
)


def num_sh_coeffs(degree: int) -> int:
	return (degree + 1) ** 2


def sh_basis(dirs: torch.Tensor) -> torch.Tensor:
	"""[N, 3] unit directions -> [N, 16] values of every basis function, degrees 0 through 3."""
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
	"""
	sh: [N, K, 3] learned weights. dirs: [N, 3] unit vectors from the camera to each gaussian.
	Returns [N, 3] RGB using degrees 0..degree: per channel, a dot product of weights and basis values.
	"""
	k = num_sh_coeffs(degree)
	rgb = torch.einsum("nk,nkc->nc", sh_basis(dirs)[:, :k], sh[:, :k])
	# The paper's convention is to +0.5 so all-zero weights mean mid-gray, clamped to non-negative.
	return (rgb + 0.5).clamp_min(0.0)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
	"""The degree-0 weight that makes eval_sh return rgb from every direction."""
	return (rgb - 0.5) / C0
