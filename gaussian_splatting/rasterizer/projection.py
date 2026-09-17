import torch

NEAR_PLANE = 0.2
LOW_PASS = 0.3
JACOBIAN_FOV_SCALE = 1.3


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
	"""Converts rotation quaternions into rotation matrices.

	Args:
		quats: [N, 4] unit quaternions ordered (w, x, y, z).

	Returns:
		[N, 3, 3] rotation matrices.
	"""
	w, x, y, z = quats.unbind(-1)
	return torch.stack([
		1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
		2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
		2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
	], dim=-1).reshape(-1, 3, 3)


class ProjectGaussians(torch.autograd.Function):
	"""Projects 3D gaussians into the image plane as 2D gaussians, in one CUDA kernel.

	Call it through project_gaussians with means [N, 3], activated scales [N, 3], unit quats [N, 4],
	the [4, 4] world to camera transform, the intrinsics and the image size. It returns means2D
	[N, 2] in pixels, depths [N] used only to sort, radii [N] which are 0 for culled gaussians and
	carry no gradient, and conics [N, 3] as the inverse 2D covariance (xx, xy, yy).
	"""

	@staticmethod
	def forward(ctx, means, scales, quats, world_to_camera, fx, fy, cx, cy, width, height):
		"""Projects the gaussians and saves the inputs the backward recomputes from.

		Args:
			ctx: Autograd context the saved tensors are stashed on.
			means: [N, 3] gaussian centers in world space.
			scales: [N, 3] per axis standard deviations, already activated.
			quats: [N, 4] unit quaternions ordered (w, x, y, z).
			world_to_camera: [4, 4] world to camera transform.
			fx: Horizontal focal length in pixels.
			fy: Vertical focal length in pixels.
			cx: Horizontal principal point in pixels.
			cy: Vertical principal point in pixels.
			width: Image width in pixels.
			height: Image height in pixels.

		Returns:
			means2D, depths, radii and conics, as project_gaussians_torch returns them.
		"""
		from gaussian_splatting.rasterizer import _get_rasterizer

		means, scales, quats = means.contiguous(), scales.contiguous(), quats.contiguous()
		outputs = _get_rasterizer().project_gaussians(
			means, scales, quats, world_to_camera, fx, fy, cx, cy, width, height,
			NEAR_PLANE, LOW_PASS, JACOBIAN_FOV_SCALE
		)
		ctx.save_for_backward(means, scales, quats, world_to_camera)
		ctx.camera = (fx, fy, cx, cy, width, height)
		ctx.mark_non_differentiable(outputs[1], outputs[2])
		return outputs

	@staticmethod
	def backward(ctx, grad_means2D, grad_depths, grad_radii, grad_conics):
		"""Recomputes the projection and walks the gradients back to the parameters.

		Depths and radii carry no gradient, matching the torch version where radii is computed
		under no_grad and depths only ever feeds the sort.

		Args:
			ctx: Autograd context holding the saved inputs.
			grad_means2D: [N, 2] gradient of the loss with respect to the pixel centers.
			grad_depths: Ignored.
			grad_radii: Ignored.
			grad_conics: [N, 3] gradient of the loss with respect to the conics.

		Returns:
			Gradients for means, scales and quats, and None for the camera arguments.
		"""
		from gaussian_splatting.rasterizer import _get_rasterizer

		means, scales, quats, world_to_camera = ctx.saved_tensors
		grads = _get_rasterizer().project_gaussians_backward(
			grad_means2D.contiguous(), grad_conics.contiguous(), means, scales, quats,
			world_to_camera, *ctx.camera, NEAR_PLANE, LOW_PASS, JACOBIAN_FOV_SCALE
		)
		return (*grads, None, None, None, None, None, None, None)


project_gaussians = ProjectGaussians.apply
