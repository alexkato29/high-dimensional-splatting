import numpy as np
import pytest
import torch

from gaussian_splatting.config import TrainingParams
from gaussian_splatting.utils.dataset import PointCloud
from gaussian_splatting.utils.gaussian import GaussianModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the model lives on CUDA")

N = 400
EXTENT = 10.0


def model_with_optimizer() -> tuple[GaussianModel, torch.optim.Adam, TrainingParams]:
	"""A small model after one Adam step, with every Adam moment row tagged by its gaussian's id.

	The id lives in sh_rest[:, 0, 0], which no densification rule reads, so every copy of a gaussian
	carries it. Each Adam moment row is filled with that id, so a moment that ends up on the wrong
	gaussian shows up as a mismatch.
	"""
	rng = np.random.default_rng(0)
	cloud = PointCloud(rng.uniform(-1, 1, (N, 3)).astype(np.float32), rng.uniform(0, 1, (N, 3)).astype(np.float32))
	model = GaussianModel(cloud)
	params = TrainingParams()
	optimizer = torch.optim.Adam(model.get_optimizer_params(params, EXTENT), eps=1e-15, fused=True)
	for p in model.params.values():
		p.grad = torch.ones_like(p)
	optimizer.step()
	optimizer.zero_grad(set_to_none=True)

	ids = torch.arange(N, device="cuda", dtype=torch.float32) + 1
	with torch.no_grad():
		model.params["sh_rest"][:, 0, 0] = ids
	for p in model.params.values():
		for key in ("exp_avg", "exp_avg_sq"):
			optimizer.state[p][key].copy_(ids.view(-1, *[1] * (p.dim() - 1)).expand_as(p))
	return model, optimizer, params


def test_densify_and_prune_keeps_params_and_moments_together():
	model, optimizer, params = model_with_optimizer()
	threshold_scale = params.percent_dense * EXTENT
	i = torch.arange(N, device="cuda")

	clone = i < 50
	split = (i >= 50) & (i < 100)
	prune = (i >= 100) & (i < 150)
	with torch.no_grad():
		model.params["scales"][:] = torch.log(torch.full((N, 3), 0.5 * threshold_scale, device="cuda"))
		model.params["scales"][split] = torch.log(torch.full((3,), 4.0 * threshold_scale, device="cuda"))
		model.params["opacities"][:] = torch.logit(torch.tensor(0.5))
		model.params["opacities"][prune] = torch.logit(torch.tensor(0.5 * params.min_opacity))
	model.means2D_grad_accum[clone | split] = 10 * params.densify_grad_threshold
	model.denom[:] = 1
	old_scales = model.scales.detach().clone()

	model.densify_and_prune(optimizer, params, EXTENT, max_screen_size=None)

	ids = (i + 1).float()
	kept = ids[~split & ~prune]
	expected = torch.cat([kept, ids[clone], ids[split], ids[split]])
	got = model.params["sh_rest"][:, 0, 0]
	torch.testing.assert_close(got, expected, rtol=0, atol=0)

	for group in optimizer.param_groups:
		p = group["params"][0]
		assert p is model.params[group["name"]], f"{group['name']}: optimizer and model hold different tensors"
		for key in ("exp_avg", "exp_avg_sq"):
			moment = optimizer.state[p][key]
			assert moment.shape == p.shape, f"{group['name']} {key}: shape {tuple(moment.shape)} vs {tuple(p.shape)}"
			rows = moment.reshape(len(p), -1)
			assert (rows[:len(kept)] == kept[:, None]).all(), f"{group['name']} {key}: survivor moments moved"
			assert (rows[len(kept):] == 0).all(), f"{group['name']} {key}: new gaussians inherited momentum"

	# Children are smaller than their parent by 1.6 on every axis.
	children = model.scales[len(kept) + int(clone.sum()):]
	torch.testing.assert_close(children, old_scales[split].repeat(2, 1) / 1.6)
	assert len(model.max_radii2D) == len(model.denom) == len(model.means2D_grad_accum) == len(got)


def test_reset_opacities_zeroes_only_opacity_moments():
	model, optimizer, _ = model_with_optimizer()
	model.reset_opacities(optimizer)

	assert (model.opacities <= 0.01 + 1e-6).all()
	ids = torch.arange(N, device="cuda", dtype=torch.float32) + 1
	for group in optimizer.param_groups:
		p = group["params"][0]
		assert p is model.params[group["name"]]
		for key in ("exp_avg", "exp_avg_sq"):
			rows = optimizer.state[p][key].reshape(N, -1)
			want = torch.zeros_like(rows) if group["name"] == "opacities" else ids[:, None].expand_as(rows)
			assert torch.equal(rows, want), f"{group['name']} {key}"
