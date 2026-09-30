#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define TILE_SIZE 16
#define RENDER_BATCH (TILE_SIZE * TILE_SIZE)

#define MIN_TRANSMITTANCE 1e-4f

#define MIN_ALPHA (1.0f / 255.0f)
#define MAX_ALPHA 0.99f


__device__ inline int2 pixel_to_tile(float2 pixel) {
	return make_int2(static_cast<int>(floorf(pixel.x / TILE_SIZE)), static_cast<int>(floorf(pixel.y / TILE_SIZE)));
}

__device__ inline float2 alpha_extent(float a_conic, float b_conic, float c_conic, float opacity) {
	float k = sqrtf(2.0f * logf(opacity / MIN_ALPHA));
	float det_conic = a_conic * c_conic - b_conic * b_conic;
	return make_float2(k * sqrtf(c_conic / det_conic), k * sqrtf(a_conic / det_conic));
}

__device__ inline bool tile_rect(float2 uv, float a_conic, float b_conic, float c_conic, float opacity,
								 int num_tiles_x, int num_tiles_y, int2& tile_min, int2& tile_size) {
	if (!(opacity > MIN_ALPHA)) return false;

	float2 extent = alpha_extent(a_conic, b_conic, c_conic, opacity);
	int2 lo = pixel_to_tile(make_float2(uv.x - extent.x, uv.y - extent.y));
	int2 hi = pixel_to_tile(make_float2(uv.x + extent.x, uv.y + extent.y));
	tile_min = make_int2(max(0, lo.x), max(0, lo.y));
	tile_size = make_int2(min(num_tiles_x - 1, hi.x) - tile_min.x + 1, min(num_tiles_y - 1, hi.y) - tile_min.y + 1);
	return tile_size.x > 0 && tile_size.y > 0;
}

inline bool debug_enabled() {
	static const bool on = (std::getenv("GS_DEBUG") != nullptr);
	return on;
}

inline void check_cuda_error(const char* kernel_name) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(
			std::string("CUDA kernel launch error (") + kernel_name + "): " +
			cudaGetErrorString(err)
		);
	}

	// Execution errors otherwise surface at the next synchronizing call, which may be a later kernel.
	if (!debug_enabled()) return;
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(
			std::string("CUDA kernel execution error (") + kernel_name + "): " +
			cudaGetErrorString(err)
		);
	}
}

inline void debug_print_duplicates(int num_gaussians, int num_tiles_x, int num_tiles_y, int total_duplicates) {
	if (!debug_enabled()) return;
	int num_tiles = num_tiles_x * num_tiles_y;
	std::fprintf(stderr,
		"[gs] gaussians=%d  tiles=%dx%d(%d)  duplicates=%d  dup/gaussian=%.1f  dup/tile=%.1f\n",
		num_gaussians, num_tiles_x, num_tiles_y, num_tiles, total_duplicates,
		(double)total_duplicates / num_gaussians,
		(double)total_duplicates / num_tiles);
}
