#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Tile size in pixels. Compile-time so that RENDER_BATCH is derived from it rather than duplicated.
// divide instruction, so dividing by a runtime value costs ~15-20 instructions, always avoid).
#define TILE_SIZE 16

// Number of gaussians loaded into SMEM per render batch. It must be equal to the block
// size since each block cooperatively loads gaussians into SMEM.
#define RENDER_BATCH (TILE_SIZE * TILE_SIZE)

#define MIN_TRANSMITTANCE 1e-4f

#define MIN_ALPHA (1.0f / 255.0f)
#define MAX_ALPHA 0.99f


__device__ inline int2 pixel_to_tile(float2 pixel) {
	return make_int2(static_cast<int>(pixel.x) / TILE_SIZE, static_cast<int>(pixel.y) / TILE_SIZE);
}

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
