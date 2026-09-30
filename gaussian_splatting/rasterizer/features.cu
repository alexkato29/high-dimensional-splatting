#include <torch/extension.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "api.h"

#define FULL_MASK 0xffffffffu


// The alpha_extent box widened slightly so fast-math rounding never culls a gaussian some pixel might accept.
__device__ inline float2 culling_extent(float4 c, float opacity) {
	float2 extent = alpha_extent(c.x, c.y, c.z, opacity);
	float ex = extent.x * 1.001f + 0.01f;
	float ey = extent.y * 1.001f + 0.01f;
	// A degenerate conic gives NaN, but we keep it rather than dropping it just in case.
	if (!(ex >= 0.0f)) ex = 1e30f;
	if (!(ey >= 0.0f)) ey = 1e30f;
	return make_float2(ex, ey);
}

__device__ inline bool box_hits(float2 mean, float2 extent, float4 rect) {
	return mean.x + extent.x >= rect.x && mean.x - extent.x <= rect.y
		&& mean.y + extent.y >= rect.z && mean.y - extent.y <= rect.w;
}

// Every lane agrees that the top left pixel is l0, and bottom right is l31.
__device__ inline float4 warp_rect(float2 pixel) {
	return make_float4(__shfl_sync(FULL_MASK, pixel.x, 0), __shfl_sync(FULL_MASK, pixel.x, 31),
					   __shfl_sync(FULL_MASK, pixel.y, 0), __shfl_sync(FULL_MASK, pixel.y, 31));
}

template <int D, int LANES_PER_PIXEL, int PIXELS_PER_BLOCK, int BATCH>
__global__ void __launch_bounds__(PIXELS_PER_BLOCK * LANES_PER_PIXEL) render_features(
	const uint2* tile_ranges,
	const int* values_sorted,
	const float2* means2D,
	const float4* conic,
	const float4* color_opacity,
	const float4* features,
	int image_width,
	int image_height,
	float* output,
	float* final_transmittance,
	int* n_contrib
) {
	constexpr int VEC4_PER_ROW = D / 4;
	constexpr int VEC4_PER_LANE = VEC4_PER_ROW / LANES_PER_PIXEL;
	constexpr int THREADS = PIXELS_PER_BLOCK * LANES_PER_PIXEL;
	constexpr int BLOCKS_PER_TILE = RENDER_BATCH / PIXELS_PER_BLOCK;
	static_assert(D % 4 == 0 && VEC4_PER_ROW % LANES_PER_PIXEL == 0, "each lane must own whole float4s");
	static_assert(BATCH % 32 == 0, "culling tests 32 gaussians per ballot");
	static_assert(RENDER_BATCH % PIXELS_PER_BLOCK == 0, "blocks must split a tile evenly");
	static_assert(PIXELS_PER_BLOCK % TILE_SIZE == 0, "blocks must cover whole tile rows");

	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int slice = tid % LANES_PER_PIXEL;

	// We use LANES_PER_PIXEL to avoid register pressure. As D increases, # of registers required to
	// keep the running sums rapidly reduces occupancy and hurts performance.
	const int local_pixel = (blockIdx.x % BLOCKS_PER_TILE) * PIXELS_PER_BLOCK + tid / LANES_PER_PIXEL;
	const int tile_idx = blockIdx.x / BLOCKS_PER_TILE;
	const int tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;

	const int px = (tile_idx % tiles_x) * TILE_SIZE + local_pixel % TILE_SIZE;
	const int py = (tile_idx / tiles_x) * TILE_SIZE + local_pixel / TILE_SIZE;
	const bool inside = px < image_width && py < image_height;
	const float2 pixel = make_float2(px + 0.5f, py + 0.5f);
	const float4 rect = warp_rect(pixel);

	__shared__ int s_id[BATCH];
	__shared__ float2 s_mean[BATCH];
	__shared__ float4 s_conic[BATCH];
	__shared__ float2 s_extent[BATCH];
	__shared__ float4 s_feat[BATCH * VEC4_PER_ROW];

	uint2 range = tile_ranges[tile_idx];
	int num_todo = (int)range.y - (int)range.x;

	// Arrays only live in registers if every index is a compile time constant, thus the unroll.
	float acc[VEC4_PER_LANE * 4];
	#pragma unroll
	for (int i = 0; i < VEC4_PER_LANE * 4; i++) acc[i] = 0.0f;

	float transmittance = 1.0f;
	bool done = !inside;
	int last_contributor = 0;

	// We process the gaussians in batches so we can stage them in shared memory.
	for (int batch_start = 0; batch_start < num_todo; batch_start += BATCH) {
		// Doubles as the barrier protecting last batch's shared memory reads from this batch's writes.
		if (__syncthreads_count(done) == THREADS) break;

		int batch_count = min(BATCH, num_todo - batch_start);
		for (int i = tid; i < batch_count; i += THREADS) {
			int g = values_sorted[range.x + batch_start + i];
			s_id[i] = g;
			s_mean[i] = means2D[g];
			float4 c = conic[g];
			float opacity = color_opacity[g].w;
			s_conic[i] = make_float4(c.x, c.y, c.z, opacity);
			s_extent[i] = culling_extent(c, opacity);
		}
		__syncthreads();
		for (int i = tid; i < batch_count * VEC4_PER_ROW; i += THREADS) {
			int k = i / VEC4_PER_ROW;
			s_feat[i] = features[(size_t)s_id[k] * VEC4_PER_ROW + (i - k * VEC4_PER_ROW)];
		}
		__syncthreads();

		// We cull gaussians at the warp level since some gaussians, while in a given warp's tile, might not touch
		// any of the given warp's actual pixels.
		for (int j0 = 0; j0 < batch_count; j0 += 32) {
			if (__all_sync(FULL_MASK, done)) break;
			int jl = j0 + lane;
			// Bit j of mask is 1 if the gaussian at j0 + j touches at least one pixel in this warp's rectangle.
			unsigned mask = __ballot_sync(FULL_MASK, jl < batch_count && box_hits(s_mean[jl], s_extent[jl], rect));
			while (mask) {
				int j = j0 + __ffs(mask) - 1;
				mask &= mask - 1;

				if (done) continue;

				float2 mean = s_mean[j];
				float4 c = s_conic[j];
				float dx = pixel.x - mean.x;
				float dy = pixel.y - mean.y;

				float mahalanobis = dx * (c.x * dx + c.y * dy) + dy * (c.y * dx + c.z * dy);
				float alpha = fminf(MAX_ALPHA, c.w * __expf(-0.5f * mahalanobis));
				if (alpha < MIN_ALPHA) continue;

				float test_transmittance = transmittance * (1.0f - alpha);
				if (test_transmittance < MIN_TRANSMITTANCE) {
					done = true;
					continue;
				}

				float w = alpha * transmittance;
				const float4* row = &s_feat[j * VEC4_PER_ROW + slice];
				#pragma unroll
				for (int i = 0; i < VEC4_PER_LANE; i++) {
					float4 f = row[i * LANES_PER_PIXEL];
					acc[4 * i + 0] += w * f.x;
					acc[4 * i + 1] += w * f.y;
					acc[4 * i + 2] += w * f.z;
					acc[4 * i + 3] += w * f.w;
				}
				transmittance = test_transmittance;
				last_contributor = batch_start + j + 1;
			}
		}
	}

	if (inside) {
		int pixel_idx = py * image_width + px;
		int plane = image_width * image_height;
		#pragma unroll
		for (int i = 0; i < VEC4_PER_LANE; i++) {
			int channel = 4 * (slice + LANES_PER_PIXEL * i);
			#pragma unroll
			for (int e = 0; e < 4; e++) output[(channel + e) * plane + pixel_idx] = acc[4 * i + e];
		}
		if (slice == 0) {
			final_transmittance[pixel_idx] = transmittance;
			n_contrib[pixel_idx] = last_contributor;
		}
	}
}

template <int D, int LANES_PER_PIXEL, int PIXELS_PER_BLOCK, int BATCH>
static void launch_render_features(int num_tiles, const uint2* tile_ranges, const int* values_sorted,
								   const float2* means2D, const float4* conic, const float4* color_opacity,
								   const float4* features, int width, int height, float* output,
								   float* final_transmittance, int* n_contrib) {
	const int blocks = num_tiles * (RENDER_BATCH / PIXELS_PER_BLOCK);
	const int threads = PIXELS_PER_BLOCK * LANES_PER_PIXEL;
	render_features<D, LANES_PER_PIXEL, PIXELS_PER_BLOCK, BATCH><<<blocks, threads>>>(
		tile_ranges, values_sorted, means2D, conic, color_opacity, features, width, height, output,
		final_transmittance, n_contrib
	);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_features(
	torch::Tensor features,
	torch::Tensor means2D,
	torch::Tensor conic,
	torch::Tensor color_opacity,
	torch::Tensor values_sorted,
	torch::Tensor tile_ranges,
	int image_width,
	int image_height
) {
	TORCH_CHECK(features.is_cuda() && features.is_contiguous() && features.scalar_type() == torch::kFloat32,
				"features must be a contiguous float32 CUDA tensor");
	const int D = features.size(1);
	const int num_tiles = ((image_width + TILE_SIZE - 1) / TILE_SIZE) * ((image_height + TILE_SIZE - 1) / TILE_SIZE);

	auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	torch::Tensor output = torch::empty({D, image_height, image_width}, f32);
	torch::Tensor final_transmittance = torch::empty({image_height, image_width}, f32);
	torch::Tensor n_contrib = torch::empty({image_height, image_width}, i32);

	auto args = std::make_tuple(
		num_tiles,
		reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
		values_sorted.data_ptr<int>(),
		reinterpret_cast<const float2*>(means2D.data_ptr<float>()),
		reinterpret_cast<const float4*>(conic.data_ptr<float>()),
		reinterpret_cast<const float4*>(color_opacity.data_ptr<float>()),
		reinterpret_cast<const float4*>(features.data_ptr<float>()),
		image_width, image_height,
		output.data_ptr<float>(), final_transmittance.data_ptr<float>(), n_contrib.data_ptr<int>()
	);

	// The fastest launch layout measured on an L4 for each D.
	switch (D) {
		case 16: std::apply(launch_render_features<16, 1, 256, 128>, args); break;
		case 32: std::apply(launch_render_features<32, 1, 256, 128>, args); break;
		case 64: std::apply(launch_render_features<64, 1, 256, 128>, args); break;
		case 128: std::apply(launch_render_features<128, 2, 128, 64>, args); break;
		default: TORCH_CHECK(false, "features must have 16, 32, 64 or 128 channels, got ", D);
	}
	check_cuda_error("render_features");
	return {output, final_transmittance, n_contrib};
}
