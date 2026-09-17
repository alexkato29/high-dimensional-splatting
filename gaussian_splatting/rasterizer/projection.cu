#include <torch/extension.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "api.h"

struct Camera {
	float fx, fy, cx, cy;
	int width, height;
	float near_plane, low_pass, fov_scale;
};


// Rotation matrix of a unit quaternion (w, x, y, z).
__device__ inline void quat_to_rotmat(const float* q, float rot[3][3]) {
	float w = q[0], x = q[1], y = q[2], z = q[3];
	rot[0][0] = 1 - 2 * (y * y + z * z); rot[0][1] = 2 * (x * y - w * z);     rot[0][2] = 2 * (x * z + w * y);
	rot[1][0] = 2 * (x * y + w * z);     rot[1][1] = 1 - 2 * (x * x + z * z); rot[1][2] = 2 * (y * z - w * x);
	rot[2][0] = 2 * (x * z - w * y);     rot[2][1] = 2 * (y * z + w * x);     rot[2][2] = 1 - 2 * (x * x + y * y);
}


// Everything the forward computes and the backward needs to walk back through.
struct Projected {
	float R[3][3];
	float t[3];
	float p[3];
	float z, z_inv, jx, jy;
	bool clamped_x, clamped_y, behind;
	float jw[2][3];
	float M[3][3];
	float T[2][3];
	float a, b, c, det;
};


__device__ inline void project(const Camera& cam, const float* world_to_camera, const float* mean,
							   const float* scale, const float* quat, Projected& g) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) g.R[i][j] = world_to_camera[i * 4 + j];
		g.t[i] = world_to_camera[i * 4 + 3];
		g.p[i] = g.R[i][0] * mean[0] + g.R[i][1] * mean[1] + g.R[i][2] * mean[2] + g.t[i];
	}
	g.behind = g.p[2] <= cam.near_plane;
	g.z = fmaxf(g.p[2], cam.near_plane);
	g.z_inv = 1.0f / (g.z + 1e-6f);

	float limit_x = cam.fov_scale * (0.5f * cam.width / cam.fx);
	float limit_y = cam.fov_scale * (0.5f * cam.height / cam.fy);
	float rx = g.p[0] * g.z_inv, ry = g.p[1] * g.z_inv;
	g.clamped_x = fabsf(rx) > limit_x;
	g.clamped_y = fabsf(ry) > limit_y;
	g.jx = fminf(fmaxf(rx, -limit_x), limit_x) * g.z;
	g.jy = fminf(fmaxf(ry, -limit_y), limit_y) * g.z;

	float zz_inv = g.z_inv * g.z_inv;
	for (int j = 0; j < 3; j++) {
		g.jw[0][j] = cam.fx * g.z_inv * g.R[0][j] - cam.fx * g.jx * zz_inv * g.R[2][j];
		g.jw[1][j] = cam.fy * g.z_inv * g.R[1][j] - cam.fy * g.jy * zz_inv * g.R[2][j];
	}

	float rot[3][3];
	quat_to_rotmat(quat, rot);
	for (int j = 0; j < 3; j++) {
		for (int k = 0; k < 3; k++) g.M[j][k] = rot[j][k] * scale[k];
	}

	for (int r = 0; r < 2; r++) {
		for (int k = 0; k < 3; k++) {
			g.T[r][k] = g.jw[r][0] * g.M[0][k] + g.jw[r][1] * g.M[1][k] + g.jw[r][2] * g.M[2][k];
		}
	}

	g.a = g.T[0][0] * g.T[0][0] + g.T[0][1] * g.T[0][1] + g.T[0][2] * g.T[0][2] + cam.low_pass;
	g.b = g.T[0][0] * g.T[1][0] + g.T[0][1] * g.T[1][1] + g.T[0][2] * g.T[1][2];
	g.c = g.T[1][0] * g.T[1][0] + g.T[1][1] * g.T[1][1] + g.T[1][2] * g.T[1][2] + cam.low_pass;
	g.det = g.a * g.c - g.b * g.b;
}


__global__ void project_forward(int n, Camera cam, const float* world_to_camera, const float* means,
								const float* scales, const float* quats, float* means2D, float* depths,
								float* radii, float* conics) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	Projected g;
	project(cam, world_to_camera, means + i * 3, scales + i * 3, quats + i * 4, g);

	float u = cam.fx * g.p[0] / g.z + cam.cx;
	float v = cam.fy * g.p[1] / g.z + cam.cy;
	means2D[i * 2 + 0] = u;
	means2D[i * 2 + 1] = v;
	depths[i] = g.p[2];

	float scale_inv = g.det > 0.0f ? 1.0f / g.det : 1.0f;
	conics[i * 3 + 0] = g.c * scale_inv;
	conics[i * 3 + 1] = -g.b * scale_inv;
	conics[i * 3 + 2] = g.a * scale_inv;

	float trace = g.a + g.c;
	float radius = 3.0f * sqrtf(0.5f * (trace + sqrtf(fmaxf(trace * trace - 4.0f * g.det, 0.0f))));
	bool visible = !g.behind && g.det > 0.0f
		&& u - radius <= cam.width && u + radius >= 0.0f
		&& v - radius <= cam.height && v + radius >= 0.0f;
	radii[i] = visible ? radius : 0.0f;
}


__global__ void project_backward(int n, Camera cam, const float* world_to_camera, const float* means,
								 const float* scales, const float* quats, const float* grad_means2D,
								 const float* grad_conics, float* grad_means, float* grad_scales,
								 float* grad_quats) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	const float* mean = means + i * 3;
	const float* scale = scales + i * 3;
	const float* quat = quats + i * 4;
	Projected g;
	project(cam, world_to_camera, mean, scale, quat, g);

	float g0 = grad_conics[i * 3 + 0], g1 = grad_conics[i * 3 + 1], g2 = grad_conics[i * 3 + 2];
	float da, db, dc;
	if (g.det > 0.0f) {
		float inv = 1.0f / g.det, inv2 = inv * inv;
		da = inv2 * (-g0 * g.c * g.c + g1 * g.b * g.c - g2 * g.b * g.b);
		db = inv2 * (2.0f * g0 * g.b * g.c + 2.0f * g2 * g.a * g.b) - g1 * (inv + 2.0f * g.b * g.b * inv2);
		dc = inv2 * (-g0 * g.b * g.b + g1 * g.a * g.b - g2 * g.a * g.a);
	} else {
		da = g2; db = -g1; dc = g0;
	}

	float dT[2][3];
	for (int k = 0; k < 3; k++) {
		dT[0][k] = 2.0f * da * g.T[0][k] + db * g.T[1][k];
		dT[1][k] = 2.0f * dc * g.T[1][k] + db * g.T[0][k];
	}

	float rot[3][3];
	quat_to_rotmat(quat, rot);
	float djw[2][3], drot[3][3], dscale[3] = {0.0f, 0.0f, 0.0f};
	for (int j = 0; j < 3; j++) {
		djw[0][j] = dT[0][0] * g.M[j][0] + dT[0][1] * g.M[j][1] + dT[0][2] * g.M[j][2];
		djw[1][j] = dT[1][0] * g.M[j][0] + dT[1][1] * g.M[j][1] + dT[1][2] * g.M[j][2];
		for (int k = 0; k < 3; k++) {
			float dM = g.jw[0][j] * dT[0][k] + g.jw[1][j] * dT[1][k];
			dscale[k] += dM * rot[j][k];
			drot[j][k] = dM * scale[k];
		}
	}

	float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
	grad_quats[i * 4 + 0] = 2.0f * (-drot[0][1] * z + drot[0][2] * y + drot[1][0] * z
									- drot[1][2] * x - drot[2][0] * y + drot[2][1] * x);
	grad_quats[i * 4 + 1] = 2.0f * (drot[0][1] * y + drot[0][2] * z + drot[1][0] * y - 2.0f * drot[1][1] * x
									- drot[1][2] * w + drot[2][0] * z + drot[2][1] * w - 2.0f * drot[2][2] * x);
	grad_quats[i * 4 + 2] = 2.0f * (-2.0f * drot[0][0] * y + drot[0][1] * x + drot[0][2] * w + drot[1][0] * x
									+ drot[1][2] * z - drot[2][0] * w + drot[2][1] * z - 2.0f * drot[2][2] * y);
	grad_quats[i * 4 + 3] = 2.0f * (-2.0f * drot[0][0] * z - drot[0][1] * w + drot[0][2] * x + drot[1][0] * w
									- 2.0f * drot[1][1] * z + drot[1][2] * y + drot[2][0] * x + drot[2][1] * y);
	for (int k = 0; k < 3; k++) grad_scales[i * 3 + k] = dscale[k];

	// jw depends on the projection jacobian, so unwind it back to the camera space point.
	float zz_inv = g.z_inv * g.z_inv;
	float dz_inv = 0.0f, djx = 0.0f, djy = 0.0f;
	for (int j = 0; j < 3; j++) {
		dz_inv += djw[0][j] * (cam.fx * g.R[0][j] - 2.0f * cam.fx * g.jx * g.z_inv * g.R[2][j]);
		dz_inv += djw[1][j] * (cam.fy * g.R[1][j] - 2.0f * cam.fy * g.jy * g.z_inv * g.R[2][j]);
		djx += djw[0][j] * (-cam.fx * zz_inv * g.R[2][j]);
		djy += djw[1][j] * (-cam.fy * zz_inv * g.R[2][j]);
	}

	float dp[3] = {0.0f, 0.0f, 0.0f};
	float dz = 0.0f;
	// jx is clamp(p.x / z, limit) * z, so the ratio only sees gradient when it is inside the clamp.
	float rx = g.jx / g.z, ry = g.jy / g.z;
	dz += djx * rx + djy * ry;
	if (!g.clamped_x) { dp[0] += djx * g.z * g.z_inv; dz_inv += djx * g.z * g.p[0]; }
	if (!g.clamped_y) { dp[1] += djy * g.z * g.z_inv; dz_inv += djy * g.z * g.p[1]; }

	float gu = grad_means2D[i * 2 + 0], gv = grad_means2D[i * 2 + 1];
	dp[0] += gu * cam.fx / g.z;
	dp[1] += gv * cam.fy / g.z;
	dz += -(gu * cam.fx * g.p[0] + gv * cam.fy * g.p[1]) / (g.z * g.z);

	dz += dz_inv * (-zz_inv);
	if (!g.behind) dp[2] += dz;

	for (int k = 0; k < 3; k++) {
		grad_means[i * 3 + k] = g.R[0][k] * dp[0] + g.R[1][k] * dp[1] + g.R[2][k] * dp[2];
	}
}


static Camera make_camera(float fx, float fy, float cx, float cy, int width, int height,
						  float near_plane, float low_pass, float fov_scale) {
	Camera cam;
	cam.fx = fx; cam.fy = fy; cam.cx = cx; cam.cy = cy;
	cam.width = width; cam.height = height;
	cam.near_plane = near_plane; cam.low_pass = low_pass; cam.fov_scale = fov_scale;
	return cam;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> project_gaussians(
	torch::Tensor means, torch::Tensor scales, torch::Tensor quats, torch::Tensor world_to_camera,
	float fx, float fy, float cx, float cy, int width, int height,
	float near_plane, float low_pass, float fov_scale
) {
	int n = means.size(0);
	Camera cam = make_camera(fx, fy, cx, cy, width, height, near_plane, low_pass, fov_scale);
	torch::Tensor pose = world_to_camera.contiguous();
	torch::Tensor means2D = torch::empty({n, 2}, means.options());
	torch::Tensor depths = torch::empty({n}, means.options());
	torch::Tensor radii = torch::empty({n}, means.options());
	torch::Tensor conics = torch::empty({n, 3}, means.options());
	if (n > 0) {
		project_forward<<<(n + 255) / 256, 256>>>(
			n, cam, pose.data_ptr<float>(), means.data_ptr<float>(), scales.data_ptr<float>(), quats.data_ptr<float>(),
			means2D.data_ptr<float>(), depths.data_ptr<float>(), radii.data_ptr<float>(),
			conics.data_ptr<float>());
		check_cuda_error("project_forward");
	}
	return {means2D, depths, radii, conics};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project_gaussians_backward(
	torch::Tensor grad_means2D, torch::Tensor grad_conics, torch::Tensor means, torch::Tensor scales,
	torch::Tensor quats, torch::Tensor world_to_camera, float fx, float fy, float cx, float cy,
	int width, int height, float near_plane, float low_pass, float fov_scale
) {
	int n = means.size(0);
	Camera cam = make_camera(fx, fy, cx, cy, width, height, near_plane, low_pass, fov_scale);
	torch::Tensor pose = world_to_camera.contiguous();
	torch::Tensor grad_means = torch::empty_like(means);
	torch::Tensor grad_scales = torch::empty_like(scales);
	torch::Tensor grad_quats = torch::empty_like(quats);
	if (n > 0) {
		project_backward<<<(n + 255) / 256, 256>>>(
			n, cam, pose.data_ptr<float>(), means.data_ptr<float>(), scales.data_ptr<float>(), quats.data_ptr<float>(),
			grad_means2D.data_ptr<float>(), grad_conics.data_ptr<float>(), grad_means.data_ptr<float>(),
			grad_scales.data_ptr<float>(), grad_quats.data_ptr<float>());
		check_cuda_error("project_backward");
	}
	return {grad_means, grad_scales, grad_quats};
}
