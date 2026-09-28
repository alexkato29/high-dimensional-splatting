#include <torch/extension.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "api.h"

#define SSIM_RADIUS 5
#define SSIM_C1 (0.01f * 0.01f)
#define SSIM_C2 (0.03f * 0.03f)

// Normalized 11 tap gaussian with sigma 1.5, which is the window the SSIM reference uses.
__constant__ float G[2 * SSIM_RADIUS + 1] = {
	1.028380084e-03f, 7.598758135e-03f, 3.600077213e-02f, 1.093606895e-01f, 2.130055377e-01f,
	2.660117249e-01f,
	2.130055377e-01f, 1.093606895e-01f, 3.600077213e-02f, 7.598758135e-03f, 1.028380084e-03f,
};


__global__ void ssim_blur_rows(int n, int width, const float* a, const float* b,
							   float* h_a, float* h_b, float* h_aa, float* h_bb, float* h_ab) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	int x = (i / 3) % width;

	float sa = 0.0f, sb = 0.0f, saa = 0.0f, sbb = 0.0f, sab = 0.0f;
	#pragma unroll
	for (int k = -SSIM_RADIUS; k <= SSIM_RADIUS; k++) {
		if (x + k < 0 || x + k >= width) continue;
		float w = G[k + SSIM_RADIUS];
		float va = a[i + 3 * k], vb = b[i + 3 * k];
		sa += w * va;
		sb += w * vb;
		saa += w * va * va;
		sbb += w * vb * vb;
		sab += w * va * vb;
	}
	h_a[i] = sa; h_b[i] = sb; h_aa[i] = saa; h_bb[i] = sbb; h_ab[i] = sab;
}

__global__ void ssim_blur_cols_forward(int n, int width, int height,
									   const float* h_a, const float* h_b, const float* h_aa,
									   const float* h_bb, const float* h_ab,
									   float* ssim_map, float* d_mu, float* d_aa, float* d_ab) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	int y = i / (3 * width);
	int row = 3 * width;

	float mu_a = 0.0f, mu_b = 0.0f, e_aa = 0.0f, e_bb = 0.0f, e_ab = 0.0f;
	#pragma unroll
	for (int k = -SSIM_RADIUS; k <= SSIM_RADIUS; k++) {
		if (y + k < 0 || y + k >= height) continue;
		float w = G[k + SSIM_RADIUS];
		int j = i + k * row;
		mu_a += w * h_a[j];
		mu_b += w * h_b[j];
		e_aa += w * h_aa[j];
		e_bb += w * h_bb[j];
		e_ab += w * h_ab[j];
	}

	float a1 = 2.0f * mu_a * mu_b + SSIM_C1;
	float a2 = 2.0f * (e_ab - mu_a * mu_b) + SSIM_C2;
	float b1 = mu_a * mu_a + mu_b * mu_b + SSIM_C1;
	float b2 = (e_aa - mu_a * mu_a) + (e_bb - mu_b * mu_b) + SSIM_C2;
	float inv_b = 1.0f / (b1 * b2);
	float s = a1 * a2 * inv_b;

	ssim_map[i] = s;
	d_mu[i] = (2.0f * mu_b * (a2 - a1) - 2.0f * mu_a * s * (b2 - b1)) * inv_b;
	d_aa[i] = -s / b2;
	d_ab[i] = 2.0f * a1 * inv_b;
}

__global__ void ssim_blur_rows_backward(int n, int width, const float* d_mu, const float* d_aa,
										const float* d_ab, float* h_mu, float* h_aa, float* h_ab) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	int x = (i / 3) % width;

	float s_mu = 0.0f, s_aa = 0.0f, s_ab = 0.0f;
	#pragma unroll
	for (int k = -SSIM_RADIUS; k <= SSIM_RADIUS; k++) {
		if (x + k < 0 || x + k >= width) continue;
		float w = G[k + SSIM_RADIUS];
		s_mu += w * d_mu[i + 3 * k];
		s_aa += w * d_aa[i + 3 * k];
		s_ab += w * d_ab[i + 3 * k];
	}
	h_mu[i] = s_mu; h_aa[i] = s_aa; h_ab[i] = s_ab;
}

__global__ void ssim_blur_cols_backward(int n, int width, int height, const float* grad_mean,
										const float* a, const float* b,
										const float* h_mu, const float* h_aa, const float* h_ab,
										float* grad_a) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	int y = i / (3 * width);
	int row = 3 * width;

	float s_mu = 0.0f, s_aa = 0.0f, s_ab = 0.0f;
	#pragma unroll
	for (int k = -SSIM_RADIUS; k <= SSIM_RADIUS; k++) {
		if (y + k < 0 || y + k >= height) continue;
		float w = G[k + SSIM_RADIUS];
		int j = i + k * row;
		s_mu += w * h_mu[j];
		s_aa += w * h_aa[j];
		s_ab += w * h_ab[j];
	}
	float scale = grad_mean[0] / n;
	grad_a[i] = scale * (s_mu + 2.0f * a[i] * s_aa + b[i] * s_ab);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ssim_forward(
	torch::Tensor rendered,
	torch::Tensor gt
) {
	TORCH_CHECK(rendered.is_cuda() && gt.is_cuda(), "ssim inputs must be CUDA tensors");
	TORCH_CHECK(rendered.is_contiguous() && gt.is_contiguous(), "ssim inputs must be contiguous");
	TORCH_CHECK(rendered.dim() == 3 && rendered.size(2) == 3 && rendered.sizes() == gt.sizes(),
				"ssim inputs must both be [H, W, 3]");

	const int height = rendered.size(0);
	const int width = rendered.size(1);
	const int n = rendered.numel();
	const int threads = 256;
	const int blocks = (n + threads - 1) / threads;

	auto h = torch::empty({5, height, width, 3}, rendered.options());
	auto ssim_map = torch::empty_like(rendered);
	auto d_mu = torch::empty_like(rendered);
	auto d_aa = torch::empty_like(rendered);
	auto d_ab = torch::empty_like(rendered);
	float* hp = h.data_ptr<float>();

	ssim_blur_rows<<<blocks, threads>>>(n, width, rendered.data_ptr<float>(), gt.data_ptr<float>(),
										hp, hp + n, hp + 2 * n, hp + 3 * n, hp + 4 * n);
	check_cuda_error("ssim_blur_rows");
	ssim_blur_cols_forward<<<blocks, threads>>>(n, width, height, hp, hp + n, hp + 2 * n, hp + 3 * n,
												hp + 4 * n, ssim_map.data_ptr<float>(),
												d_mu.data_ptr<float>(), d_aa.data_ptr<float>(),
												d_ab.data_ptr<float>());
	check_cuda_error("ssim_blur_cols_forward");

	return {ssim_map, d_mu, d_aa, d_ab};
}

torch::Tensor ssim_backward(
	torch::Tensor grad_mean,
	torch::Tensor rendered,
	torch::Tensor gt,
	torch::Tensor d_mu,
	torch::Tensor d_aa,
	torch::Tensor d_ab
) {
	const int height = rendered.size(0);
	const int width = rendered.size(1);
	const int n = rendered.numel();
	const int threads = 256;
	const int blocks = (n + threads - 1) / threads;

	auto h = torch::empty({3, height, width, 3}, rendered.options());
	auto grad_rendered = torch::empty_like(rendered);
	float* hp = h.data_ptr<float>();

	ssim_blur_rows_backward<<<blocks, threads>>>(n, width, d_mu.data_ptr<float>(), d_aa.data_ptr<float>(),
												 d_ab.data_ptr<float>(), hp, hp + n, hp + 2 * n);
	check_cuda_error("ssim_blur_rows_backward");
	ssim_blur_cols_backward<<<blocks, threads>>>(n, width, height, grad_mean.data_ptr<float>(),
												 rendered.data_ptr<float>(), gt.data_ptr<float>(),
												 hp, hp + n, hp + 2 * n, grad_rendered.data_ptr<float>());
	check_cuda_error("ssim_blur_cols_backward");

	return grad_rendered;
}
