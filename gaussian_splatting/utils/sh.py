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
	"""Turns learned spherical harmonic weights into an RGB color per gaussian, in one CUDA kernel.

	One thread per gaussian evaluates the basis functions for its viewing direction and mixes the
	weights, so no basis or coefficient tensor is ever materialized. Call it through eval_sh with
	sh_dc [N, 1, 3], sh_rest [N, 15, 3], dirs [N, 3], the highest degree to evaluate and the
	projection radii, and it returns [N, 3] RGB offset by 0.5 and clamped to non-negative. Culled
	gaussians are skipped and get a color of zero. The backward never produces SH gradients, it steps
	the SH weights with the given Adam instead.
	"""

	@staticmethod
	def forward(ctx, sh_dc: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor, degree: int,
				radii: torch.Tensor, optimizer: torch.optim.Adam | None) -> torch.Tensor:
		"""Evaluates the colors and saves the inputs the backward needs.

		Args:
			ctx: Autograd context the saved tensors are stashed on.
			sh_dc: [N, 1, 3] degree 0 weights, contiguous.
			sh_rest: [N, 15, 3] weights for degrees 1 through 3, contiguous.
			dirs: [N, 3] unit vectors from the camera to each gaussian, contiguous.
			degree: Highest degree to evaluate.
			radii: [N] projection radii, 0 for culled gaussians.
			optimizer: Adam holding sh_dc and sh_rest in groups named "sh_dc" and "sh_rest", which the
				backward steps in place. None is only valid when nothing will backpropagate.

		Returns:
			[N, 3] RGB.
		"""
		sh_dc, sh_rest, dirs, radii = sh_dc.contiguous(), sh_rest.contiguous(), dirs.contiguous(), radii.contiguous()
		ctx.save_for_backward(dirs, radii)
		ctx.degree = degree
		ctx.optimizer = optimizer
		return _get_rasterizer().eval_sh(sh_dc, sh_rest, dirs, degree, radii)

	@staticmethod
	def backward(ctx, grad_rgb: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
		"""Steps the SH weights with Adam in place and returns the gradient for the directions.

		Args:
			ctx: Autograd context holding the saved inputs.
			grad_rgb: [N, 3] gradient of the loss with respect to the colors.

		Returns:
			The gradient for dirs, and None for everything else, including sh_dc and sh_rest
			because Adam already consumed their gradients.
		"""
		if ctx.optimizer is None:
			raise RuntimeError("eval_sh backward needs the Adam that holds sh_dc and sh_rest, set model.sh_optimizer")
		dirs, radii = ctx.saved_tensors
		with torch.cuda.nvtx.range("sh_backward"):
			dc, dc_group, dc_state = _adam_slot(ctx.optimizer, "sh_dc")
			rest, rest_group, rest_state = _adam_slot(ctx.optimizer, "sh_rest")
			beta1, beta2 = dc_group["betas"]
			grad_dirs = _get_rasterizer().eval_sh_backward_adam(
				grad_rgb.contiguous(), dc.data, rest.data, dirs, ctx.degree, radii,
				dc_state["exp_avg"], dc_state["exp_avg_sq"], rest_state["exp_avg"], rest_state["exp_avg_sq"],
				dc_state["step"], rest_state["step"], dc_group["lr"], rest_group["lr"], beta1, beta2,
				dc_group["eps"]
			)
		return None, None, grad_dirs, None, None, None


def _adam_slot(optimizer: torch.optim.Adam, name: str) -> tuple[torch.Tensor, dict, dict]:
	"""Finds a named parameter's Adam group and state, creating the state and counting this step.

	The state matches what torch.optim.Adam(fused=True) creates, so either side can step it next.

	Args:
		optimizer: Adam with one parameter per group and a "name" key on each group.
		name: Group name to look up.

	Returns:
		The parameter, its group and its state, with state["step"] already incremented.
	"""
	group = next(g for g in optimizer.param_groups if g["name"] == name)
	param = group["params"][0]
	state = optimizer.state[param]
	if not state:
		state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
		state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
		state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
	state["step"] += 1
	return param, group, state


eval_sh = EvalSH.apply


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
	"""Inverts eval_sh at degree 0, which is how the model is initialized from point cloud colors.

	Args:
		rgb: [N, 3] colors in [0, 1].

	Returns:
		[N, 3] degree 0 weights that make eval_sh return rgb from every direction.
	"""
	return (rgb - 0.5) / C0
