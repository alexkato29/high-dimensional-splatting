#include <torch/extension.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "api.h"

__device__ inline float warp_octet_sum(float v) {
	for (int offset = 4; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffffu, v, offset);
	return v;
}


__global__ void render_gaussians_backward(
	const uint2* tile_ranges,
	const int* tiled_gaussian_values_sorted,
	const float2* means2D,
	const float4* conic,
	const float4* color_opacity,
	const float* background,
	const float* final_transmittance,
	const int* n_contrib,
	const float* grad_image,
	int image_width,
	int image_height,
	float* grad_means2D,
	float* grad_conics,
	float* grad_colors,
	float* grad_opacities
) {
	const int block_size = blockDim.x * blockDim.y;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;

	bool inside = (px < image_width && py < image_height);
	float2 pixel = make_float2(px + 0.5f, py + 0.5f);

	__shared__ float2 s_mean[RENDER_BATCH];
	__shared__ float4 s_conic[RENDER_BATCH];
	__shared__ float4 s_color[RENDER_BATCH];
	__shared__ int s_id[RENDER_BATCH];

	uint2 range = tile_ranges[tile_idx];
	int num_todo = (int)range.y - (int)range.x;

	int pixel_idx = py * image_width + px;
	float transmittance = inside ? final_transmittance[pixel_idx] : 0.0f;
	int last_contributor = inside ? n_contrib[pixel_idx] : 0;
	float3 dL_dpixel = make_float3(0.0f, 0.0f, 0.0f);
	if (inside) {
		dL_dpixel = make_float3(
			grad_image[pixel_idx * 3 + 0],
			grad_image[pixel_idx * 3 + 1],
			grad_image[pixel_idx * 3 + 2]);
	}

	// The loop walks back to front, so everything past the block's deepest contributor fails the
	// range gate for every pixel here. Start below it instead of burning iterations on the tail.
	__shared__ int s_last_contributor;
	if (tid == 0) s_last_contributor = 0;
	__syncthreads();
	atomicMax(&s_last_contributor, last_contributor);
	__syncthreads();
	num_todo = min(num_todo, s_last_contributor);

	const float final_T = transmittance;
	const float dL_dfinal_T = background[0] * dL_dpixel.x
							+ background[1] * dL_dpixel.y
							+ background[2] * dL_dpixel.z;

	float3 accum_rec = make_float3(0.0f, 0.0f, 0.0f);
	float3 last_color = make_float3(0.0f, 0.0f, 0.0f);
	float last_alpha = 0.0f;

	for (int batch_end = num_todo; batch_end > 0; batch_end -= block_size) {
		int batch_count = min(block_size, batch_end);
		int batch_start = batch_end - batch_count;

		__syncthreads();
		int fetch = batch_start + tid;
		if (fetch < batch_end) {
			int g = tiled_gaussian_values_sorted[range.x + fetch];
			s_id[tid] = g;
			s_mean[tid] = means2D[g];
			s_conic[tid] = conic[g];
			s_color[tid] = color_opacity[g];
		}
		__syncthreads();

		for (int j = batch_count - 1; j >= 0; j--) {
			bool contributes = inside && batch_start + j + 1 <= last_contributor;
			float dL_dcolor = 0.0f, dL_dopacity = 0.0f;
			float2 dL_dmean2D = make_float2(0.0f, 0.0f);
			float3 dL_dconic = make_float3(0.0f, 0.0f, 0.0f);

			if (contributes) {
				float2 mean = s_mean[j];
				float4 c = s_conic[j];
				float dx = pixel.x - mean.x;
				float dy = pixel.y - mean.y;
				float mahalanobis = dx * (c.x * dx + c.y * dy) +
									dy * (c.y * dx + c.z * dy);
				float G = __expf(-0.5f * mahalanobis);

				float4 color = s_color[j];
				float raw_alpha = color.w * G;
				float alpha = fminf(MAX_ALPHA, raw_alpha);
				contributes = alpha >= MIN_ALPHA;

				if (contributes) {
					transmittance /= (1.0f - alpha);

					accum_rec.x = last_alpha * last_color.x + (1.0f - last_alpha) * accum_rec.x;
					accum_rec.y = last_alpha * last_color.y + (1.0f - last_alpha) * accum_rec.y;
					accum_rec.z = last_alpha * last_color.z + (1.0f - last_alpha) * accum_rec.z;
					last_color = make_float3(color.x, color.y, color.z);
					last_alpha = alpha;

					float dL_dalpha = ((color.x - accum_rec.x) * dL_dpixel.x
									 + (color.y - accum_rec.y) * dL_dpixel.y
									 + (color.z - accum_rec.z) * dL_dpixel.z) * transmittance
									- final_T / (1.0f - alpha) * dL_dfinal_T;

					dL_dcolor = alpha * transmittance;

					// A clamped alpha no longer depends on the gaussian, so nothing flows past it.
					if (raw_alpha <= MAX_ALPHA) {
						dL_dopacity = G * dL_dalpha;
						float dL_dG = color.w * dL_dalpha;
						dL_dmean2D = make_float2(dL_dG * G * (c.x * dx + c.y * dy),
												 dL_dG * G * (c.y * dx + c.z * dy));
						dL_dconic = make_float3(dL_dG * G * -0.5f * dx * dx,
												dL_dG * G * -dx * dy,
												dL_dG * G * -0.5f * dy * dy);
					}
				}
			}

			unsigned int active = __ballot_sync(0xffffffffu, contributes);
			if (active) {
				float grad_color_x = warp_octet_sum(dL_dcolor * dL_dpixel.x);
				float grad_color_y = warp_octet_sum(dL_dcolor * dL_dpixel.y);
				float grad_color_z = warp_octet_sum(dL_dcolor * dL_dpixel.z);
				float grad_opacity = warp_octet_sum(dL_dopacity);
				float grad_mean_x = warp_octet_sum(dL_dmean2D.x);
				float grad_mean_y = warp_octet_sum(dL_dmean2D.y);
				float grad_conic_x = warp_octet_sum(dL_dconic.x);
				float grad_conic_y = warp_octet_sum(dL_dconic.y);
				float grad_conic_z = warp_octet_sum(dL_dconic.z);

				if ((tid & 7) == 0 && ((active >> (tid & 31)) & 0xffu)) {
					int g = s_id[j];
					// On an L4, there are no vectorized atomic instructions.
					atomicAdd(&grad_colors[g * 3 + 0], grad_color_x);
					atomicAdd(&grad_colors[g * 3 + 1], grad_color_y);
					atomicAdd(&grad_colors[g * 3 + 2], grad_color_z);
					atomicAdd(&grad_opacities[g], grad_opacity);
					atomicAdd(&grad_means2D[g * 2 + 0], grad_mean_x);
					atomicAdd(&grad_means2D[g * 2 + 1], grad_mean_y);
					atomicAdd(&grad_conics[g * 3 + 0], grad_conic_x);
					atomicAdd(&grad_conics[g * 3 + 1], grad_conic_y);
					atomicAdd(&grad_conics[g * 3 + 2], grad_conic_z);
				}
			}
		}
	}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward(
	torch::Tensor grad_image,
	torch::Tensor means2D,
	torch::Tensor conic,
	torch::Tensor color_opacity,
	torch::Tensor values_sorted,
	torch::Tensor tile_ranges,
	torch::Tensor final_transmittance,
	torch::Tensor n_contrib,
	torch::Tensor background,
	int image_width,
	int image_height
) {
	auto check_input = [](const torch::Tensor& t, const char* name) {
		TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
		TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
	};
	check_input(grad_image, "grad_image");
	check_input(means2D, "means2D");
	check_input(conic, "conic");
	check_input(color_opacity, "color_opacity");
	check_input(values_sorted, "values_sorted");
	check_input(tile_ranges, "tile_ranges");
	check_input(final_transmittance, "final_transmittance");
	check_input(n_contrib, "n_contrib");
	check_input(background, "background");

	const int num_gaussians = means2D.size(0);
	const int num_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
	const int num_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;

	auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor grad_means2D = torch::zeros({num_gaussians, 2}, f32);
	torch::Tensor grad_conics = torch::zeros({num_gaussians, 3}, f32);
	torch::Tensor grad_colors = torch::zeros({num_gaussians, 3}, f32);
	torch::Tensor grad_opacities = torch::zeros({num_gaussians, 1}, f32);

	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(num_tiles_x, num_tiles_y);

	render_gaussians_backward<<<grid, block>>>(
		reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
		values_sorted.data_ptr<int>(),
		reinterpret_cast<const float2*>(means2D.data_ptr<float>()),
		reinterpret_cast<const float4*>(conic.data_ptr<float>()),
		reinterpret_cast<const float4*>(color_opacity.data_ptr<float>()),
		background.data_ptr<float>(),
		final_transmittance.data_ptr<float>(),
		n_contrib.data_ptr<int>(),
		grad_image.data_ptr<float>(),
		image_width,
		image_height,
		grad_means2D.data_ptr<float>(),
		grad_conics.data_ptr<float>(),
		grad_colors.data_ptr<float>(),
		grad_opacities.data_ptr<float>()
	);
	check_cuda_error("render_gaussians_backward");

	return {grad_means2D, grad_conics, grad_colors, grad_opacities};
}
