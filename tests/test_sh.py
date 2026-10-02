import pytest
import torch
import torch.nn.functional as F

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.utils.sh import eval_sh, rgb_to_sh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="eval_sh needs CUDA")

N = 5000


def eval_sh_reference(sh: torch.Tensor, dirs: torch.Tensor, degree: int, radii: torch.Tensor) -> torch.Tensor:
	"""The real spherical harmonics as the 3DGS reference code writes them out, one term per line.

	Args:
		sh: [N, 16, 3] weights, degree 0 first.
		dirs: [N, 3] unit viewing directions.
		degree: Highest degree to evaluate.
		radii: [N] culled gaussians (0) come out black.

	Returns:
		[N, 3] RGB, offset by 0.5 and clamped at 0.
	"""
	x, y, z = (dirs[:, i:i + 1] for i in range(3))
	xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
	terms = [0.28209479177387814 + 0 * x]
	if degree >= 1:
		terms += [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
	if degree >= 2:
		terms += [
			1.0925484305920792 * xy,
			-1.0925484305920792 * yz,
			0.31539156525252005 * (2 * zz - xx - yy),
			-1.0925484305920792 * xz,
			0.5462742152960396 * (xx - yy),
		]
	if degree >= 3:
		terms += [
			-0.5900435899266435 * y * (3 * xx - yy),
			2.890611442640554 * xy * z,
			-0.4570457994644658 * y * (4 * zz - xx - yy),
			0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
			-0.4570457994644658 * x * (4 * zz - xx - yy),
			1.445305721320277 * z * (xx - yy),
			-0.5900435899266435 * x * (xx - 3 * yy),
		]
	rgb = 0.5 + sum(term * sh[:, j] for j, term in enumerate(terms))
	return torch.where(radii[:, None] > 0, rgb.clamp_min(0), torch.zeros_like(rgb))


def inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Random weights big enough that some colors clamp at 0, unit directions, and ~10% culled gaussians."""
	g = torch.Generator(device="cuda").manual_seed(seed)
	sh = torch.randn(N, 16, 3, generator=g, device="cuda") * 0.5
	dirs = F.normalize(torch.randn(N, 3, generator=g, device="cuda"), dim=-1)
	radii = (torch.rand(N, generator=g, device="cuda") > 0.1).float()
	return sh, dirs, radii


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_eval_sh_matches_reference(degree):
	sh, dirs, radii = inputs(degree)
	got = eval_sh(sh[:, :1].contiguous(), sh[:, 1:].contiguous(), dirs, degree, radii, None)
	want = eval_sh_reference(sh.double(), dirs.double(), degree, radii).float()
	assert (want == 0).any(), "no color clamped, so the clamp is untested"
	torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_rgb_to_sh_inverts_degree_zero():
	rgb = torch.rand(N, 3, device="cuda")
	_, dirs, _ = inputs(0)
	sh_dc = rgb_to_sh(rgb)[:, None].contiguous()
	sh_rest = torch.zeros(N, 15, 3, device="cuda")
	torch.testing.assert_close(eval_sh(sh_dc, sh_rest, dirs, 0, torch.ones(N, device="cuda"), None), rgb)


def adam_for(sh_dc: torch.Tensor, sh_rest: torch.Tensor) -> torch.optim.Adam:
	params = TrainingParams()
	return torch.optim.Adam([
		{"params": [sh_dc], "lr": params.sh_dc_lr, "name": "sh_dc"},
		{"params": [sh_rest], "lr": params.sh_rest_lr, "name": "sh_rest"},
	], eps=1e-15, fused=True)


@pytest.mark.parametrize("degree", [1, 3])
def test_fused_sh_adam_matches_autograd_and_torch_adam(degree):
	sh, dirs, radii = inputs(10 + degree)
	g = torch.Generator(device="cuda").manual_seed(degree)

	fused_dc = torch.nn.Parameter(sh[:, :1].clone())
	fused_rest = torch.nn.Parameter(sh[:, 1:].clone())
	fused_adam = adam_for(fused_dc, fused_rest)
	ref_dc = torch.nn.Parameter(sh[:, :1].clone())
	ref_rest = torch.nn.Parameter(sh[:, 1:].clone())
	ref_adam = adam_for(ref_dc, ref_rest)

	for _ in range(3):
		grad_rgb = torch.randn(N, 3, generator=g, device="cuda")

		fused_dirs = dirs.clone().requires_grad_(True)
		eval_sh(fused_dc, fused_rest, fused_dirs, degree, radii, fused_adam).backward(grad_rgb)
		assert fused_dc.grad is None and fused_rest.grad is None, "the fused backward must not leave SH gradients"

		ref_dirs = dirs.clone().requires_grad_(True)
		eval_sh_reference(torch.cat([ref_dc, ref_rest], dim=1), ref_dirs, degree, radii).backward(grad_rgb)
		ref_adam.step()
		ref_adam.zero_grad(set_to_none=True)

		torch.testing.assert_close(fused_dirs.grad, ref_dirs.grad, rtol=1e-4, atol=1e-5)

	for fused, ref in [(fused_dc, ref_dc), (fused_rest, ref_rest)]:
		torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-6)
		for key in ("exp_avg", "exp_avg_sq", "step"):
			torch.testing.assert_close(fused_adam.state[fused][key], ref_adam.state[ref][key], rtol=1e-5, atol=1e-7)
