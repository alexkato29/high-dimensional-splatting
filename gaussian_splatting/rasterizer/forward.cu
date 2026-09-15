#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <cub/device/device_radix_sort.cuh>
#include "helpers.h"


// Tile size in pixels. Compile-time so that RENDER_BATCH is derived from it rather than duplicated.
// Also, the compiler can optimize pixel_to_tile's integer divisions  into shifts (GPUs have no integer
// divide instruction, so dividing by a runtime value costs ~15-20 instructions, always avoid).
#define TILE_SIZE 16

// Number of gaussians loaded into SMEM per render batch. It must be equal to the block
// size since each block cooperatively loads gaussians into SMEM.
#define RENDER_BATCH (TILE_SIZE * TILE_SIZE)


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
	const float* opacities,
	const float* colors,
	const float* world_to_cam_matrix,
	float focal_x,
	float focal_y,
	float c_x,
	float c_y,
	int image_width,
	int image_height,
	int num_tiles_x,
	int num_tiles_y,
	float4* gaussian_data,
	float2* means2D,
	float4* conic,
	float4* color_opacity,
	int* tiles_touched
) {
	// Step 1: We must project the gaussian mean to the screen space (u, v)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_gaussians) return;

	float3 p_world = load_float3(means3D, idx);
	float3 p_cam = world_to_camera(world_to_cam_matrix, p_world);

	// We cull along the way to avoid any and all unnecessary work
	// This causes warp divergence, so we will have to see if it impacts perf
	if (is_behind_camera(p_cam.z, 0.2f)) {
		gaussian_data[idx] = make_float4(0.0f, 0.0f, 0.0f, 1e10f);
		means2D[idx] = make_float2(0.0f, 0.0f);
		tiles_touched[idx] = 0;
		return;
	}

	float2 uv = pinhole_projection(p_cam, focal_x, focal_y, c_x, c_y);

	if (is_centered_off_screen(uv, image_width, image_height)) {
		gaussian_data[idx] = make_float4(0.0f, 0.0f, 0.0f, 1e10f);
		means2D[idx] = make_float2(0.0f, 0.0f);
		tiles_touched[idx] = 0;
		return;
	}

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

	float radius = compute_radius_from_cov2D(cov2D_out);

	if (is_completely_off_screen(uv, image_width, image_height, radius)) {
		gaussian_data[idx] = make_float4(0.0f, 0.0f, 0.0f, 1e10f);
		means2D[idx] = make_float2(0.0f, 0.0f);
		tiles_touched[idx] = 0;
		return;
	}

	// Seems repetitive, but we do this because duplicate_gaussians is extremely memory bound and
	// benefits from one vectorized representation, while render_gaussians only needs means.
	gaussian_data[idx] = make_float4(uv.x, uv.y, radius, p_cam.z);
	means2D[idx] = uv;

	// Invert the 2x2 covariance here, once per gaussian, rather than once per (pixel, gaussian)
	// pair in the render loop. det/det_inv/the three conic products depend only on cov2D, so they
	// are loop-invariant across pixels.
	float det = cov2D_out.x * cov2D_out.z - cov2D_out.y * cov2D_out.y;
	if (det <= 0.0f) {
		gaussian_data[idx] = make_float4(0.0f, 0.0f, 0.0f, 1e10f);
		tiles_touched[idx] = 0;
		return;
	}
	float det_inv = 1.0f / det;
	conic[idx] = make_float4(
		cov2D_out.z * det_inv, -cov2D_out.y * det_inv, cov2D_out.x * det_inv, 0.0f);

	// Color is 12B at 12B alignment, which the compiler must split into three LDG.E.
	// Padding it to a float4 and parking opacity in the spare lane allows vectorized
	// access in the render loop.
	color_opacity[idx] = make_float4(
		colors[idx*3 + 0], colors[idx*3 + 1], colors[idx*3 + 2], opacities[idx]);

	float2 min_pixel = make_float2(uv.x - radius, uv.y - radius);
	float2 max_pixel = make_float2(uv.x + radius, uv.y + radius);

	int2 tile_min_coords = pixel_to_tile(min_pixel, TILE_SIZE);
	int2 tile_max_coords = pixel_to_tile(max_pixel, TILE_SIZE);

	// This should handle partially on screen gaussians. We just clamp it to only on screen tiles,
	// then we just need to make sure we do the same when duplicating the gaussians.
	tile_min_coords.x = max(0, tile_min_coords.x);
	tile_min_coords.y = max(0, tile_min_coords.y);
	tile_max_coords.x = min(num_tiles_x - 1, tile_max_coords.x);
	tile_max_coords.y = min(num_tiles_y - 1, tile_max_coords.y);

	tiles_touched[idx] = (tile_max_coords.x - tile_min_coords.x + 1) * (tile_max_coords.y - tile_min_coords.y + 1);
}

__device__ int find_gaussian_binary_search(
	const int* offsets,
	int num_gaussians,
	int output_idx
) {
	// Cheap despite the scattered look: early probes hit the same few offsets across all threads, so
	// they stay in L1/L2. See docs/rasterizer.md.
	int left = 0;
	int right = num_gaussians - 1;

	while (left < right) {
		int mid = (left + right + 1) / 2;
		if (offsets[mid] <= output_idx) {
			left = mid;
		} else {
			right = mid - 1;
		}
	}
	return left;
}

__global__ void duplicate_gaussians(
	int num_duplicates,
	int num_gaussians,
	const int* offsets,
	const float4* gaussian_data,
	int num_tiles_x,
	int num_tiles_y,
	uint64_t* tiled_gaussian_keys,
	int* tiled_gaussian_values
) {
	// One thread per output (duplicate) index, not per gaussian: ~5.5x faster, since writes stay
	// coalesced. Alternatives tried and why they lost: docs/rasterizer.md.
	int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (output_idx >= num_duplicates) return;

	int gaussian_idx = find_gaussian_binary_search(offsets, num_gaussians, output_idx);

	float4 data = gaussian_data[gaussian_idx];
	float2 uv = make_float2(data.x, data.y);
	float radius = data.z;
	float depth = data.w;
	uint32_t depth_bits = *((uint32_t*)&depth);

	float2 min_pixel = make_float2(uv.x - radius, uv.y - radius);
	float2 max_pixel = make_float2(uv.x + radius, uv.y + radius);

	int2 gaussian_min_tile_coords = pixel_to_tile(min_pixel, TILE_SIZE);
	int2 gaussian_max_tile_coords = pixel_to_tile(max_pixel, TILE_SIZE);

	gaussian_min_tile_coords.x = max(0, gaussian_min_tile_coords.x);
	gaussian_min_tile_coords.y = max(0, gaussian_min_tile_coords.y);
	gaussian_max_tile_coords.x = min(num_tiles_x - 1, gaussian_max_tile_coords.x);
	gaussian_max_tile_coords.y = min(num_tiles_y - 1, gaussian_max_tile_coords.y);

	int local_idx = output_idx - offsets[gaussian_idx];
	int gaussian_width = gaussian_max_tile_coords.x - gaussian_min_tile_coords.x + 1;
	int tile_y = gaussian_min_tile_coords.y + local_idx / gaussian_width;
	int tile_x = gaussian_min_tile_coords.x + local_idx % gaussian_width;

	int tile_idx = tile_y * num_tiles_x + tile_x;
	uint64_t key = ((uint64_t)tile_idx << 32) | depth_bits;

	tiled_gaussian_keys[output_idx] = key;
	tiled_gaussian_values[output_idx] = gaussian_idx;
}

__global__ void identify_tile_ranges(
	int num_duplicates,
	const uint64_t* sorted_keys,
	uint2* tile_ranges
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_duplicates) return;

	uint64_t key = sorted_keys[idx];
	uint32_t curr_tile = key >> 32;

	if (idx == 0) {
		tile_ranges[curr_tile].x = 0;
	} else {
		uint32_t prev_tile = sorted_keys[idx - 1] >> 32;
		if (curr_tile != prev_tile) {
			tile_ranges[prev_tile].y = idx;
			tile_ranges[curr_tile].x = idx;
		}
	}

	if (idx == num_duplicates - 1) {
		tile_ranges[curr_tile].y = num_duplicates;
	}
}

__global__ void render_gaussians(
	const uint2* tile_ranges,
	const int* tiled_gaussian_values_sorted,
	const float2* means2D,
	const float4* conic,
	const float4* color_opacity,
	int image_width,
	int image_height,
	float* output
) {
	// All pixels in a block share one gaussian list, so each batch is staged through SMEM once per
	// block instead of every warp re-requesting it from L1. See docs/rasterizer.md.
	const int block_size = blockDim.x * blockDim.y;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;

	// Out-of-bounds threads cannot return early: they still have to help fetch and, more
	// importantly, they have to keep hitting the __syncthreads barriers below.
	bool inside = (px < image_width && py < image_height);

	float2 pixel = make_float2(px + 0.5f, py + 0.5f);

	__shared__ float2 s_mean[RENDER_BATCH];
	__shared__ float4 s_conic[RENDER_BATCH];
	__shared__ float4 s_color[RENDER_BATCH];

	uint2 range = tile_ranges[tile_idx];
	int num_todo = (int)range.y - (int)range.x;

	float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float transmittance = 1.0f;
	bool done = !inside;

	for (int batch_start = 0; batch_start < num_todo; batch_start += block_size) {
		// Doubles as the barrier protecting last iteration's SMEM reads from this iteration's writes.
		if (__syncthreads_count(done) == block_size) break;

		int fetch = batch_start + tid;
		if (fetch < num_todo) {
			int g = tiled_gaussian_values_sorted[range.x + fetch];
			s_mean[tid] = means2D[g];
			s_conic[tid] = conic[g];
			s_color[tid] = color_opacity[g];
		}
		__syncthreads();

		int batch_count = min(block_size, num_todo - batch_start);
		for (int j = 0; j < batch_count && !done; j++) {
			float2 mean = s_mean[j];
			float4 c = s_conic[j];

			// c = (conic.xx, conic.xy, conic.yy, unused), already inverted in project_gaussians.
			float dx = pixel.x - mean.x;
			float dy = pixel.y - mean.y;
			float mahalanobis = dx * (c.x * dx + c.y * dy) +
								dy * (c.y * dx + c.z * dy);
			float weight = __expf(-0.5f * mahalanobis);

			float4 color = s_color[j];
			float alpha = color.w * weight;

			if (alpha < 1e-4f) continue;

			accumulated_color.x += alpha * transmittance * color.x;
			accumulated_color.y += alpha * transmittance * color.y;
			accumulated_color.z += alpha * transmittance * color.z;

			transmittance *= (1.0f - alpha);

			if (transmittance < 1e-3f) done = true;
		}
	}

	if (inside) {
		((float4*)output)[py * image_width + px] = accumulated_color;
	}
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
	auto check_input = [](const torch::Tensor& t, const char* name) {
		TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
		TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
	};
	check_input(means3D, "means3D");
	check_input(scales, "scales");
	check_input(quaternions, "quaternions");
	check_input(opacities, "opacities");
	check_input(colors, "colors");
	check_input(world_to_cam_matrix, "world_to_cam_matrix");

	const int num_gaussians = means3D.size(0);
	const int num_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
	const int num_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;
	const int num_tiles = num_tiles_x * num_tiles_y;

	// These came from python and thus are torch::Tensor types
	float* means3D_ptr = means3D.data_ptr<float>();
	float* scales_ptr = scales.data_ptr<float>();
	float* quaternions_ptr = quaternions.data_ptr<float>();
	float* opacities_ptr = opacities.data_ptr<float>();
	float* colors_ptr = colors.data_ptr<float>();
	float* world_to_cam_matrix_ptr = world_to_cam_matrix.data_ptr<float>();

	auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
	auto u8  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

	torch::Tensor gaussian_data = torch::empty({num_gaussians, 4}, f32);  // xy=means2D, z=radius, w=depth
	torch::Tensor means2D       = torch::empty({num_gaussians, 2}, f32);
	torch::Tensor conic         = torch::empty({num_gaussians, 4}, f32);
	torch::Tensor color_opacity = torch::empty({num_gaussians, 4}, f32);
	torch::Tensor tiles_touched = torch::empty({num_gaussians}, i32);

	float4* gaussian_data_ptr = reinterpret_cast<float4*>(gaussian_data.data_ptr<float>());
	float2* means2D_ptr       = reinterpret_cast<float2*>(means2D.data_ptr<float>());
	float4* conic_ptr         = reinterpret_cast<float4*>(conic.data_ptr<float>());
	float4* color_opacity_ptr = reinterpret_cast<float4*>(color_opacity.data_ptr<float>());
	int*    tiles_touched_ptr = tiles_touched.data_ptr<int>();

	// Step 1: Project gaussians from 3D world space to 2D screen space
	const int threads = 256;
	auto blocks_for = [threads](int n) { return (n + threads - 1) / threads; };

	project_gaussians<<<blocks_for(num_gaussians), threads>>>(
		num_gaussians,
		means3D_ptr,
		scales_ptr,
		quaternions_ptr,
		opacities_ptr,
		colors_ptr,
		world_to_cam_matrix_ptr,
		focal_x,
		focal_y,
		c_x,
		c_y,
		image_width,
		image_height,
		num_tiles_x,
		num_tiles_y,
		gaussian_data_ptr,
		means2D_ptr,
		conic_ptr,
		color_opacity_ptr,
		tiles_touched_ptr
	);
	check_cuda_error("project_gaussians");

	// Step 2: bin gaussians into tiles, sorted by depth within each tile. Walkthrough of the
	// offsets -> duplicate -> sort -> ranges pipeline: docs/rasterizer.md.
	torch::Tensor offsets = torch::empty({num_gaussians}, i32);
	int* offsets_ptr = offsets.data_ptr<int>();

	thrust::device_ptr<int> tiles_touched_thrust(tiles_touched_ptr);
	thrust::device_ptr<int> offsets_thrust(offsets_ptr);
	// This is prefix sum despite the fancy name
	thrust::exclusive_scan(tiles_touched_thrust,
						   tiles_touched_thrust + num_gaussians,
						   offsets_thrust);

	// We avoid unnecessary transfers by bringing back ONLY the last element of each
	int last_offset;
	int last_tiles_touched;
	cudaMemcpy(&last_offset, offsets_ptr + num_gaussians - 1, sizeof(int),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(&last_tiles_touched, tiles_touched_ptr + num_gaussians - 1, sizeof(int),
			   cudaMemcpyDeviceToHost);
	int total_duplicates = last_offset + last_tiles_touched;

	if (total_duplicates == 0) {
		return torch::zeros({image_height, image_width, 3}, f32);
	}



	torch::Tensor keys   = torch::empty({total_duplicates}, i64);
	torch::Tensor values = torch::empty({total_duplicates}, i32);

	uint64_t* tiled_gaussian_keys   = reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>());
	int*      tiled_gaussian_values = values.data_ptr<int>();

	// How this kernel's parallelism was chosen: docs/rasterizer.md.
	duplicate_gaussians<<<blocks_for(total_duplicates), threads>>>(
		total_duplicates,
		num_gaussians,
		offsets_ptr,
		gaussian_data_ptr,
		num_tiles_x,
		num_tiles_y,
		tiled_gaussian_keys,
		tiled_gaussian_values
	);
	check_cuda_error("duplicate_gaussians");

	torch::Tensor keys_sorted   = torch::empty({total_duplicates}, i64);
	torch::Tensor values_sorted = torch::empty({total_duplicates}, i32);

	uint64_t* tiled_gaussian_keys_sorted   = reinterpret_cast<uint64_t*>(keys_sorted.data_ptr<int64_t>());
	int*      tiled_gaussian_values_sorted = values_sorted.data_ptr<int>();

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	// Keys are (tile_idx << 32 | depth_bits), so everything above bit 32+ceil(log2(num_tiles))
	// is always zero. CUB's onesweep does 8 bits per pass, so sorting the full 64 bits
	// can be wasteful when we aren't using anywhere close to 2^33 - 1 tiles.
	int tile_bits = 0;
	// Use a loop to find the upper bound power of two. Floating point precision makes using
	// a logarithm directly a bit risky, this is stable.
	while ((1 << tile_bits) < num_tiles) tile_bits++;
	const int sort_end_bit = 32 + tile_bits;

	// Radix sort needs temporary storage. We don't know exactly how much, but if you pass it
	// a nullptr it will automatically decide how much it needs.
	cub::DeviceRadixSort::SortPairs(
		d_temp_storage, temp_storage_bytes,
		tiled_gaussian_keys, tiled_gaussian_keys_sorted,
		tiled_gaussian_values, tiled_gaussian_values_sorted,
		total_duplicates, 0, sort_end_bit
	);

	torch::Tensor temp_storage = torch::empty({(int64_t)temp_storage_bytes}, u8);
	d_temp_storage = temp_storage.data_ptr();

	cub::DeviceRadixSort::SortPairs(
		d_temp_storage, temp_storage_bytes,
		tiled_gaussian_keys, tiled_gaussian_keys_sorted,
		tiled_gaussian_values, tiled_gaussian_values_sorted,
		total_duplicates, 0, sort_end_bit
	);

	check_cuda_error("radix_sort");

	torch::Tensor tile_ranges_t = torch::zeros({num_tiles, 2}, i32);
	uint2* tile_ranges = reinterpret_cast<uint2*>(tile_ranges_t.data_ptr<int>());

	identify_tile_ranges<<<blocks_for(total_duplicates), threads>>>(
		total_duplicates,
		tiled_gaussian_keys_sorted,
		tile_ranges
	);
	check_cuda_error("identify_tile_ranges");

	// Step 3: alpha blend/render the gaussians
	// We use a 4th channel to allow vectorized writes, but we don't actually care about it.
	torch::Tensor output = torch::empty({image_height, image_width, 4}, f32);

	float* output_ptr = output.data_ptr<float>();

	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(num_tiles_x, num_tiles_y);

	render_gaussians<<<grid, block>>>(
		tile_ranges,
		tiled_gaussian_values_sorted,
		means2D_ptr,
		conic_ptr,
		color_opacity_ptr,
		image_width,
		image_height,
		output_ptr
	);
	check_cuda_error("render_gaussians");

	// Drop the useless 4th channel here, torch can do this for free functionally
	return output.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
}
