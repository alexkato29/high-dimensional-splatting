import torch

# Must match common.cuh.
MIN_ALPHA = 1.0 / 255.0
MAX_ALPHA = 0.99
MIN_TRANSMITTANCE = 1e-4


def rasterize_reference(
	means2D: torch.Tensor,
	depths: torch.Tensor,
	radii: torch.Tensor,
	conics: torch.Tensor,
	colors: torch.Tensor,
	opacities: torch.Tensor,
	width: int,
	height: int,
	background: torch.Tensor | None = None,
) -> torch.Tensor:
	"""Dense float64 reference for rasterize(), every pixel against every gaussian.

	No tiles, sorting tricks or culling beyond radii, just the blend as the paper writes it. Autograd
	through it is the reference for the CUDA backward. Works for any number of channels D.

	Args:
		means2D: [N, 2] centers in pixels.
		depths: [N] camera space depths, blended front to back.
		radii: [N] footprint radii, 0 for culled gaussians.
		conics: [N, 3] inverse 2D covariances as (xx, xy, yy).
		colors: [N, D] values to blend.
		opacities: [N, 1] in (0, 1).
		width: Output width in pixels.
		height: Output height in pixels.
		background: [D] shown through whatever transmittance is left, zeros by default.

	Returns:
		[H, W, D] image in the dtype of means2D.
	"""
	f64 = torch.float64
	if background is None:
		background = torch.zeros(colors.shape[1], device=means2D.device, dtype=means2D.dtype)

	# Stable so equal depths keep index order like the CUDA radix sort.
	order = torch.sort(depths, stable=True).indices
	order = order[radii[order] > 0]
	mean, conic, color = means2D[order].to(f64), conics[order].to(f64), colors[order].to(f64)
	opacity = opacities[order, 0].to(f64)

	ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
	# Pixel centers sit at +0.5 as in the CUDA kernels.
	pixels = torch.stack([xs, ys], dim=-1).reshape(-1, 2).to(means2D.device, f64) + 0.5
	# Caps each [pixels, gaussians] float64 intermediate at 128 MB. Autograd still keeps every
	# chunk's saved tensors so only small scenes fit.
	chunk = max(1, 2**24 // max(1, len(order)))
	out = []
	for px in pixels.split(chunk):
		d = px[:, None] - mean
		dx, dy = d[..., 0], d[..., 1]
		mahalanobis = dx * (conic[:, 0] * dx + conic[:, 1] * dy) + dy * (conic[:, 1] * dx + conic[:, 2] * dy)
		alpha = (opacity * torch.exp(-0.5 * mahalanobis)).clamp(max=MAX_ALPHA)
		alpha = torch.where(alpha >= MIN_ALPHA, alpha, torch.zeros_like(alpha))

		post_transmittance = torch.cumprod(1 - alpha, dim=1)
		transmittance = torch.cat([
			torch.ones_like(post_transmittance[:, :1]), post_transmittance[:, :-1]
		], dim=1)

		# Only blend gaussians that actually leave any room for transmittance behind it.
		w = alpha * transmittance
		w = torch.where(post_transmittance >= MIN_TRANSMITTANCE, w, torch.zeros_like(w))
		remaining_light = 1 - torch.sum(w, dim=1, keepdim=True)

		# Background is impenetrable, so it's opacity is 1.
		bg = background * remaining_light
		out.append(w @ color + bg)

	return torch.cat(out).reshape(height, width, colors.shape[1]).to(means2D.dtype)
