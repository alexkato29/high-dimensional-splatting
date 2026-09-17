#include <torch/extension.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "api.h"

__constant__ float C[16] = {
	0.28209479177387814f,
	-0.4886025119029199f, 0.4886025119029199f, -0.4886025119029199f,
	1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f,
	-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
	-0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f,
};


__device__ inline void basis(float x, float y, float z, int k, float* b) {
	float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
	b[0] = C[0];
	if (k <= 1) return;
	b[1] = C[1] * y; b[2] = C[2] * z; b[3] = C[3] * x;
	if (k <= 4) return;
	b[4] = C[4] * xy; b[5] = C[5] * yz; b[6] = C[6] * (2 * zz - xx - yy); b[7] = C[7] * xz; b[8] = C[8] * (xx - yy);
	if (k <= 9) return;
	b[9]  = C[9]  * y * (3 * xx - yy);   b[10] = C[10] * xy * z;
	b[11] = C[11] * y * (4 * zz - xx - yy);
	b[12] = C[12] * z * (2 * zz - 3 * xx - 3 * yy);
	b[13] = C[13] * x * (4 * zz - xx - yy);
	b[14] = C[14] * z * (xx - yy);       b[15] = C[15] * x * (xx - 3 * yy);
}


// Partial derivatives of each basis function with respect to x, y and z.
__device__ inline void basis_grad(float x, float y, float z, int k, float* bx, float* by, float* bz) {
	float xx = x * x, yy = y * y, zz = z * z;
	bx[0] = by[0] = bz[0] = 0.0f;
	if (k <= 1) return;
	bx[1] = 0;      by[1] = C[1];   bz[1] = 0;
	bx[2] = 0;      by[2] = 0;      bz[2] = C[2];
	bx[3] = C[3];   by[3] = 0;      bz[3] = 0;
	if (k <= 4) return;
	bx[4] = C[4] * y;            by[4] = C[4] * x;            bz[4] = 0;
	bx[5] = 0;                   by[5] = C[5] * z;            bz[5] = C[5] * y;
	bx[6] = C[6] * (-2 * x);     by[6] = C[6] * (-2 * y);     bz[6] = C[6] * (4 * z);
	bx[7] = C[7] * z;            by[7] = 0;                   bz[7] = C[7] * x;
	bx[8] = C[8] * (2 * x);      by[8] = C[8] * (-2 * y);     bz[8] = 0;
	if (k <= 9) return;
	bx[9]  = C[9] * (6 * x * y);            by[9]  = C[9] * (3 * xx - 3 * yy);      bz[9]  = 0;
	bx[10] = C[10] * y * z;                 by[10] = C[10] * x * z;                 bz[10] = C[10] * x * y;
	bx[11] = C[11] * (-2 * x * y);          by[11] = C[11] * (4 * zz - xx - 3 * yy); bz[11] = C[11] * (8 * y * z);
	bx[12] = C[12] * (-6 * x * z);          by[12] = C[12] * (-6 * y * z);          bz[12] = C[12] * (6 * zz - 3 * xx - 3 * yy);
	bx[13] = C[13] * (4 * zz - 3 * xx - yy); by[13] = C[13] * (-2 * x * y);          bz[13] = C[13] * (8 * x * z);
	bx[14] = C[14] * (2 * x * z);           by[14] = C[14] * (-2 * y * z);          bz[14] = C[14] * (xx - yy);
	bx[15] = C[15] * (3 * xx - 3 * yy);     by[15] = C[15] * (-6 * x * y);          bz[15] = 0;
}


// Degree 0 lives in sh_dc and the rest in sh_rest, so neither has to be copied into one tensor.
#define WEIGHT(j, c) ((j) == 0 ? dc[c] : rest[((j) - 1) * 3 + (c)])


__global__ void eval_sh_forward(int n, int k, const float* sh_dc, const float* sh_rest,
								const float3* dirs, float* rgb) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	float b[16];
	float3 d = dirs[i];
	basis(d.x, d.y, d.z, k, b);

	const float* dc = sh_dc + i * 3;
	const float* rest = sh_rest + i * 45;
	for (int c = 0; c < 3; c++) {
		float value = 0.5f;
		for (int j = 0; j < k; j++) value += b[j] * WEIGHT(j, c);
		rgb[i * 3 + c] = fmaxf(value, 0.0f);
	}
}


__global__ void eval_sh_backward(int n, int k, const float* sh_dc, const float* sh_rest, const float3* dirs,
								 const float* grad_rgb, float* grad_dc, float* grad_rest, float3* grad_dirs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	float b[16], bx[16], by[16], bz[16];
	float3 d = dirs[i];
	basis(d.x, d.y, d.z, k, b);
	basis_grad(d.x, d.y, d.z, k, bx, by, bz);

	// clamp_min(0) passes no gradient where the color was clipped, so recompute it rather than
	// saving a mask.
	const float* dc = sh_dc + i * 3;
	const float* rest = sh_rest + i * 45;
	float g[3];
	for (int c = 0; c < 3; c++) {
		float value = 0.5f;
		for (int j = 0; j < k; j++) value += b[j] * WEIGHT(j, c);
		g[c] = value > 0.0f ? grad_rgb[i * 3 + c] : 0.0f;
	}

	float* ddc = grad_dc + i * 3;
	float* drest = grad_rest + i * 45;
	float3 dd = make_float3(0.0f, 0.0f, 0.0f);
	for (int j = 0; j < 16; j++) {
		float t = 0.0f;
		for (int c = 0; c < 3; c++) {
			float grad = j < k ? b[j] * g[c] : 0.0f;
			if (j == 0) ddc[c] = grad; else drest[(j - 1) * 3 + c] = grad;
			if (j < k) t += WEIGHT(j, c) * g[c];
		}
		if (j < k) { dd.x += t * bx[j]; dd.y += t * by[j]; dd.z += t * bz[j]; }
	}
	grad_dirs[i] = dd;
}


torch::Tensor eval_sh(torch::Tensor sh_dc, torch::Tensor sh_rest, torch::Tensor dirs, int degree) {
	int n = sh_dc.size(0);
	torch::Tensor rgb = torch::empty({n, 3}, sh_dc.options());
	if (n > 0) {
		eval_sh_forward<<<(n + 255) / 256, 256>>>(
			n, (degree + 1) * (degree + 1), sh_dc.data_ptr<float>(), sh_rest.data_ptr<float>(),
			reinterpret_cast<const float3*>(dirs.data_ptr<float>()), rgb.data_ptr<float>());
		check_cuda_error("eval_sh_forward");
	}
	return rgb;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> eval_sh_backward_host(
	torch::Tensor grad_rgb, torch::Tensor sh_dc, torch::Tensor sh_rest, torch::Tensor dirs, int degree
) {
	int n = sh_dc.size(0);
	torch::Tensor grad_dc = torch::empty_like(sh_dc);
	torch::Tensor grad_rest = torch::empty_like(sh_rest);
	torch::Tensor grad_dirs = torch::empty_like(dirs);
	if (n > 0) {
		eval_sh_backward<<<(n + 255) / 256, 256>>>(
			n, (degree + 1) * (degree + 1), sh_dc.data_ptr<float>(), sh_rest.data_ptr<float>(),
			reinterpret_cast<const float3*>(dirs.data_ptr<float>()), grad_rgb.data_ptr<float>(),
			grad_dc.data_ptr<float>(), grad_rest.data_ptr<float>(),
			reinterpret_cast<float3*>(grad_dirs.data_ptr<float>()));
		check_cuda_error("eval_sh_backward");
	}
	return {grad_dc, grad_rest, grad_dirs};
}
