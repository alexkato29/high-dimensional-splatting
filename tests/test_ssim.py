import pytest
import torch

from gaussian_splatting.utils.loss import ssim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused SSIM needs CUDA")


def images(seed: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
	"""A random target and a render near it, so SSIM sits in a realistic range rather than near 0."""
	g = torch.Generator(device="cuda").manual_seed(seed)
	gt = torch.rand(height, width, 3, generator=g, device="cuda")
	rendered = (gt + 0.1 * torch.randn(height, width, 3, generator=g, device="cuda")).clamp(0, 1)
	return rendered, gt


@pytest.mark.parametrize("height, width", [(64, 64), (37, 53), (11, 11), (546, 979)])
def test_fused_ssim_matches_torch_reference(height, width):
	rendered, gt = images(height * width, height, width)

	fused_input = rendered.clone().requires_grad_(True)
	fused = ssim(fused_input, gt)
	fused.backward()

	ref_input = rendered.double().requires_grad_(True)
	ref = ssim(ref_input, gt.double())
	ref.backward()

	assert 0.2 < ref.item() < 0.99
	torch.testing.assert_close(fused.double(), ref, rtol=1e-5, atol=1e-6)
	torch.testing.assert_close(fused_input.grad.double(), ref_input.grad, rtol=1e-3, atol=1e-9)


def test_ssim_of_an_image_with_itself_is_one():
	_, gt = images(0, 64, 64)
	torch.testing.assert_close(ssim(gt, gt), torch.tensor(1.0, device="cuda"))
