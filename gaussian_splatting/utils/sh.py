import torch

from gaussian_splatting.rasterizer import _get_rasterizer

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


class EvalSH(torch.autograd.Function):
	"""Autograd node wrapping the CUDA spherical harmonics kernels."""

	@staticmethod
	def forward(ctx, sh_dc: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
		"""Evaluates the colors and saves the inputs the backward needs.

		Args:
			ctx: Autograd context the saved tensors are stashed on.
			sh_dc: [N, 1, 3] degree 0 weights, contiguous.
			sh_rest: [N, 15, 3] weights for degrees 1 through 3, contiguous.
			dirs: [N, 3] unit vectors from the camera to each gaussian, contiguous.
			degree: Highest degree to evaluate.

		Returns:
			[N, 3] RGB.
		"""
		ctx.save_for_backward(sh_dc, sh_rest, dirs)
		ctx.degree = degree
		return _get_rasterizer().eval_sh(sh_dc, sh_rest, dirs, degree)

	@staticmethod
	def backward(ctx, grad_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
		"""Recomputes the basis and scatters the gradients back to the weights and the directions.

		Args:
			ctx: Autograd context holding the saved inputs.
			grad_rgb: [N, 3] gradient of the loss with respect to the colors.

		Returns:
			Gradients for sh_dc, sh_rest and dirs, and None for the degree.
		"""
		sh_dc, sh_rest, dirs = ctx.saved_tensors
		grad_dc, grad_rest, grad_dirs = _get_rasterizer().eval_sh_backward(
			grad_rgb.contiguous(), sh_dc, sh_rest, dirs, ctx.degree
		)
		return grad_dc, grad_rest, grad_dirs, None


def eval_sh(sh_dc: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
	"""Turns learned spherical harmonic weights into an RGB color per gaussian.

	One CUDA thread per gaussian evaluates the basis functions for its viewing direction and mixes
	the weights, so no basis or coefficient tensor is ever materialized.

	Args:
		sh_dc: [N, 1, 3] degree 0 weights.
		sh_rest: [N, 15, 3] weights for degrees 1 through 3, ordered by increasing degree.
		dirs: [N, 3] unit vectors from the camera to each gaussian.
		degree: Highest degree to use, so every band above it is ignored.

	Returns:
		[N, 3] RGB, offset by 0.5 so all zero weights mean mid gray, clamped to non-negative.
	"""
	return EvalSH.apply(sh_dc.contiguous(), sh_rest.contiguous(), dirs.contiguous(), degree)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
	"""Inverts eval_sh at degree 0, which is how the model is initialized from point cloud colors.

	Args:
		rgb: [N, 3] colors in [0, 1].

	Returns:
		[N, 3] degree 0 weights that make eval_sh return rgb from every direction.
	"""
	return (rgb - 0.5) / C0
