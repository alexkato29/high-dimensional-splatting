#pragma once
#include <cuda_runtime.h>


__device__ inline float3 load_float3(const float* ptr, int idx) {
	return make_float3(ptr[idx*3 + 0], ptr[idx*3 + 1], ptr[idx*3 + 2]);
}

__device__ inline float4 load_float4(const float* ptr, int idx) {
	return make_float4(ptr[idx*4 + 0], ptr[idx*4 + 1], ptr[idx*4 + 2], ptr[idx*4 + 3]);
}

__device__ inline float3 world_to_camera(const float* mat, float3 v) {
	return make_float3(
		mat[0]*v.x + mat[1]*v.y + mat[2]*v.z + mat[3],
		mat[4]*v.x + mat[5]*v.y + mat[6]*v.z + mat[7],
		mat[8]*v.x + mat[9]*v.y + mat[10]*v.z + mat[11]
	);
}

__device__ inline bool is_behind_camera(float depth, float near_plane = 0.2f) {
	return depth <= near_plane;
}

__device__ inline float2 pinhole_projection(float3 p_cam, float focal_x, float focal_y, float c_x, float c_y) {
	float u = focal_x * (p_cam.x / p_cam.z) + c_x;
	float v = focal_y * (p_cam.y / p_cam.z) + c_y;
	return make_float2(u, v);
}

__device__ inline bool is_off_screen(float2 p, int width, int height, int margin = 100) {
	return (p.x < -margin || p.x >= width + margin ||
			p.y < -margin || p.y >= height + margin);
}

__device__ inline float evaluate_gaussian_2d(float2 pixel, float2 mean, float3 cov) {
	float dx = pixel.x - mean.x;
	float dy = pixel.y - mean.y;

	float det = cov.x * cov.z - cov.y * cov.y;
	if (det == 0.0f) return 0.0f;

	float det_inv = 1.0f / det;
	float cov_inv_xx = cov.z * det_inv;
	float cov_inv_xy = -cov.y * det_inv;
	float cov_inv_yy = cov.x * det_inv;

	float mahalanobis = dx * (cov_inv_xx * dx + cov_inv_xy * dy) +
						dy * (cov_inv_xy * dx + cov_inv_yy * dy);

	return expf(-0.5f * mahalanobis);
}

__device__ inline void quat_to_rotmat(float4 q, float* R) {
	float w = q.x, x = q.y, y = q.z, z = q.w;

	// If we don't normalize the quaternion, the rotation matrix will scale
	float norm = sqrtf(w*w + x*x + y*y + z*z);
	w /= norm; x /= norm; y /= norm; z /= norm;

	// This is a harcoded formula
	R[0] = 1.0f - 2.0f*(y*y + z*z);
	R[1] = 2.0f*(x*y - w*z);
	R[2] = 2.0f*(x*z + w*y);

	R[3] = 2.0f*(x*y + w*z);
	R[4] = 1.0f - 2.0f*(x*x + z*z);
	R[5] = 2.0f*(y*z - w*x);

	R[6] = 2.0f*(x*z - w*y);
	R[7] = 2.0f*(y*z + w*x);
	R[8] = 1.0f - 2.0f*(x*x + y*y);
}

__device__ inline void compute_cov3D(float3 scale, const float* R, float* cov3D) {
	float M[9];
	M[0] = R[0] * scale.x; M[1] = R[1] * scale.y; M[2] = R[2] * scale.z;
	M[3] = R[3] * scale.x; M[4] = R[4] * scale.y; M[5] = R[5] * scale.z;
	M[6] = R[6] * scale.x; M[7] = R[7] * scale.y; M[8] = R[8] * scale.z;

	cov3D[0] = M[0]*M[0] + M[1]*M[1] + M[2]*M[2];
	cov3D[1] = M[0]*M[3] + M[1]*M[4] + M[2]*M[5];
	cov3D[2] = M[0]*M[6] + M[1]*M[7] + M[2]*M[8];
	cov3D[3] = M[3]*M[3] + M[4]*M[4] + M[5]*M[5];
	cov3D[4] = M[3]*M[6] + M[4]*M[7] + M[5]*M[8];
	cov3D[5] = M[6]*M[6] + M[7]*M[7] + M[8]*M[8];
}

__device__ inline float3 project_cov_matrix(
	const float* cov3D,
	float3 p_cam,
	const float* viewmatrix,
	float focal_x,
	float focal_y
) {
	float z_inv = 1.0f / (p_cam.z + 1e-6f);
	float z_inv2 = z_inv * z_inv;

	float J[6];
	J[0] = focal_x * z_inv;
	J[1] = 0.0f;
	J[2] = -focal_x * p_cam.x * z_inv2;
	J[3] = 0.0f;
	J[4] = focal_y * z_inv;
	J[5] = -focal_y * p_cam.y * z_inv2;

	float W[9];
	W[0] = viewmatrix[0]; W[1] = viewmatrix[1]; W[2] = viewmatrix[2];
	W[3] = viewmatrix[4]; W[4] = viewmatrix[5]; W[5] = viewmatrix[6];
	W[6] = viewmatrix[8]; W[7] = viewmatrix[9]; W[8] = viewmatrix[10];

	float Sigma3D[9];
	Sigma3D[0] = cov3D[0]; Sigma3D[1] = cov3D[1]; Sigma3D[2] = cov3D[2];
	Sigma3D[3] = cov3D[1]; Sigma3D[4] = cov3D[3]; Sigma3D[5] = cov3D[4];
	Sigma3D[6] = cov3D[2]; Sigma3D[7] = cov3D[4]; Sigma3D[8] = cov3D[5];

	float WS[9];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			WS[i*3 + j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				WS[i*3 + j] += W[i*3 + k] * Sigma3D[k*3 + j];
			}
		}
	}

	float T[9];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			T[i*3 + j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				T[i*3 + j] += WS[i*3 + k] * W[j*3 + k];
			}
		}
	}

	float JT[6];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			JT[i*3 + j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				JT[i*3 + j] += J[i*3 + k] * T[k*3 + j];
			}
		}
	}

	float xx = JT[0]*J[0] + JT[1]*J[1] + JT[2]*J[2];
	float xy = JT[0]*J[3] + JT[1]*J[4] + JT[2]*J[5];
	float yy = JT[3]*J[3] + JT[4]*J[4] + JT[5]*J[5];

	return make_float3(xx + 0.3f, xy, yy + 0.3f);
}
