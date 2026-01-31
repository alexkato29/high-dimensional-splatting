#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "helpers.h"

inline void check_cuda_error(const char* kernel_name) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(
			std::string("CUDA kernel launch error (") + kernel_name + "): " +
			cudaGetErrorString(err)
		);
	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(
			std::string("CUDA kernel execution error (") + kernel_name + "): " +
			cudaGetErrorString(err)
		);
	}
}


__global__ void project_gaussians(
	int num_gaussians,
	const float* means3D,
	const float* scales,
	const float* quaternions,
	const float* world_to_cam_matrix,
	float focal_x,
	float focal_y,
	float c_x,
	float c_y,
	int image_width,
	int image_height,
	float* means2D,
	float* cov2D,
	float* depths
) {
	// Step 1: We must project the gaussian mean to the screen space (u, v)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_gaussians) return;

	float3 p_world = load_float3(means3D, idx);
	float3 p_cam = world_to_camera(world_to_cam_matrix, p_world);

	// We cull along the way to avoid any and all unnecessary work
	// This causes warp divergence, so we will have to see if it impacts perf
	if (is_behind_camera(p_cam.z, 0.2f)) {
		depths[idx] = 1e10f;
		return;
	}

	float2 uv = pinhole_projection(p_cam, focal_x, focal_y, c_x, c_y);

	if (is_off_screen(uv, image_width, image_height)) {
		depths[idx] = 1e10f;
		return;
	}

	((float2*)means2D)[idx] = uv;
	depths[idx] = p_cam.z;

	// Step 2: We project the 3D covariance matrix to 2D
	float3 scale = load_float3(scales, idx);
	float4 quat = load_float4(quaternions, idx);

	float R[9];
	quat_to_rotmat(quat, R);

	float cov3D[6];
	compute_cov3D(scale, R, cov3D);

	float3 cov2D_out = project_cov_matrix(
		cov3D,
		p_cam,
		world_to_cam_matrix,
		focal_x,
		focal_y
	);

	((float3*)cov2D)[idx] = cov2D_out;
	// TODO: Optionally cull more gaussians...
}

__global__ void render_gaussians(
	int num_gaussians,
	const int* sorted_indices,
	const float* means2D,
	const float* cov2D,
	const float* colors,
	const float* opacities,
	int image_width,
	int image_height,
	float* output
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;

	if (px >= image_width || py >= image_height) return;

	float2 pixel = make_float2(px + 0.5f, py + 0.5f);

	float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
	float transmittance = 1.0f;

	for (int i = 0; i < num_gaussians; i++) {
		int idx = sorted_indices[i];

		float2 mean = ((float2*)means2D)[idx];
		float3 cov = ((float3*)cov2D)[idx];
		float3 color = ((float3*)colors)[idx];
		float opacity = opacities[idx];

		float weight = evaluate_gaussian_2d(pixel, mean, cov);
		float alpha = opacity * weight;

		if (alpha < 1e-4f) continue;

		accumulated_color.x += alpha * transmittance * color.x;
		accumulated_color.y += alpha * transmittance * color.y;
		accumulated_color.z += alpha * transmittance * color.z;

		transmittance *= (1.0f - alpha);

		if (transmittance < 1e-3f) break;
	}

	int pixel_idx = py * image_width + px;
	output[pixel_idx * 3 + 0] = accumulated_color.x;
	output[pixel_idx * 3 + 1] = accumulated_color.y;
	output[pixel_idx * 3 + 2] = accumulated_color.z;
}

torch::Tensor rasterize(
	torch::Tensor means3D,
	torch::Tensor scales,
	torch::Tensor quaternions,
	torch::Tensor opacities,
	torch::Tensor colors,
	torch::Tensor world_to_cam_matrix,
	float focal_x,
	float focal_y,
	float c_x,
	float c_y,
	int image_width,
	int image_height
) {
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor output = torch::zeros({image_height, image_width, 3}, options);

	int num_gaussians = means3D.size(0);

	// Covariance matrices are symmetric, so we can store 3 elements for 2D and 6 for 3D
	torch::Tensor means2D = torch::zeros({num_gaussians, 2}, options);
	torch::Tensor cov2D = torch::zeros({num_gaussians, 3}, options);
	torch::Tensor depths = torch::zeros({num_gaussians}, options);

	float* means3D_ptr = means3D.data_ptr<float>();
	float* scales_ptr = scales.data_ptr<float>();
	float* quaternions_ptr = quaternions.data_ptr<float>();
	float* opacities_ptr = opacities.data_ptr<float>();
	float* colors_ptr = colors.data_ptr<float>();
	float* world_to_cam_matrix_ptr = world_to_cam_matrix.data_ptr<float>();
	float* means2D_ptr = means2D.data_ptr<float>();
	float* cov2D_ptr = cov2D.data_ptr<float>();
	float* depths_ptr = depths.data_ptr<float>();

	int threads = 256;
	int blocks = (num_gaussians + threads - 1) / threads;
	project_gaussians<<<blocks, threads>>>(
		num_gaussians,
		means3D_ptr,
		scales_ptr,
		quaternions_ptr,
		world_to_cam_matrix_ptr,
		focal_x,
		focal_y,
		c_x,
		c_y,
		image_width,
		image_height,
		means2D_ptr,
		cov2D_ptr,
		depths_ptr
	);
	check_cuda_error("project_gaussians");

	// Step 2: sort gaussians by depth
	torch::Tensor indices = torch::arange(num_gaussians, options.dtype(torch::kInt32));
	int* indices_ptr = indices.data_ptr<int>();

	thrust::device_ptr<float> depth_ptr(depths_ptr);
	thrust::device_ptr<int> idx_ptr(indices_ptr);

	thrust::sort_by_key(depth_ptr, depth_ptr + num_gaussians, idx_ptr);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(
			std::string("Thrust sort error: ") + cudaGetErrorString(err)
		);
	}

	float* output_ptr = output.data_ptr<float>();

	dim3 block(16, 16);
	dim3 grid(
		(image_width + block.x - 1) / block.x,
		(image_height + block.y - 1) / block.y
	);

	render_gaussians<<<grid, block>>>(
		num_gaussians,
		indices_ptr,
		means2D_ptr,
		cov2D_ptr,
		colors_ptr,
		opacities_ptr,
		image_width,
		image_height,
		output_ptr
	);
	check_cuda_error("render_gaussians");

	return output;
}