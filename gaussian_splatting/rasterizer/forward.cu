#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <cub/device/device_radix_sort.cuh>
#include "helpers.h"


// TODO: Once backward pass working, we can consider optimizing memory further via lower
// precision storage of gaussian data (e.g., uint4) to allow even higher vectorization during reads

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
	int tile_size,
	int num_tiles_x,
	int num_tiles_y,
	float4* gaussian_data,
	float2* means2D,
	float4* cov2D,
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

	cov2D[idx] = make_float4(cov2D_out.x, cov2D_out.y, cov2D_out.z, 0.0f);

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

	float2 min_pixel = make_float2(uv.x - radius, uv.y - radius);
	float2 max_pixel = make_float2(uv.x + radius, uv.y + radius);

	int2 tile_min_coords = pixel_to_tile(min_pixel, tile_size);
	int2 tile_max_coords = pixel_to_tile(max_pixel, tile_size);

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
	/*
	The first N iterations of binary search check 2^(N-1) unique elements. That makes
	this ridiculously cache friendly. The most frequent elements (like offsets[(num_gaussians - 1) / 2])
	can stay on the L1 cache. The rest of offsets also fits comfortably on the L2 cache.

	I had tried coercing the cache layout by using a coarse/fine binary search, but that
	didn't have any impact. Seems like the hardware is already being very smart.
	*/
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
	int tile_size,
	int num_tiles_x,
	int num_tiles_y,
	int tile_cols,
	uint64_t* tiled_gaussian_keys,
	int* tiled_gaussian_values
) {
	/*
	This kernel initially had "flipped" parallelism and mapped threads to gaussian ids
	rather than to output indices. That ate away at performance due to massive problems
	with shuffled writes to global memory. Instead, mapping threads to output indices 
	ended up running substantially faster (~x5.5).

	I also tried CSR. Basically, swap binary searches for writes to SMEM:
	1. Find the min and max gaussian ID that the block works on
	2. Make an array in smem of length blockDim.x (but must be known at compile time)
	3. For every GID this block touches, replicate the GID in smem on all the indices
	of output it is tied to
	4. Lookup the output_idx in gaussian_indices[output_idx] rather than making a binary
	search call

	However, this approach was ultimately slower since the writes to smem ended up leading
	to warp divergence. Also, gaussians have space in between them in the output (obviously,
	that is what this kernel solves), so the smem writes weren't coalesced. Plus, caching is 
	so effective that the binary search baseline is already hard to beat.
	*/
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

	int2 gaussian_min_tile_coords = pixel_to_tile(min_pixel, tile_size);
	int2 gaussian_max_tile_coords = pixel_to_tile(max_pixel, tile_size);

	gaussian_min_tile_coords.x = max(0, gaussian_min_tile_coords.x);
	gaussian_min_tile_coords.y = max(0, gaussian_min_tile_coords.y);
	gaussian_max_tile_coords.x = min(num_tiles_x - 1, gaussian_max_tile_coords.x);
	gaussian_max_tile_coords.y = min(num_tiles_y - 1, gaussian_max_tile_coords.y);

	int local_idx = output_idx - offsets[gaussian_idx];
	int gaussian_width = gaussian_max_tile_coords.x - gaussian_min_tile_coords.x + 1;
	int tile_y = gaussian_min_tile_coords.y + local_idx / gaussian_width;
	int tile_x = gaussian_min_tile_coords.x + local_idx % gaussian_width;

	int tile_idx = tile_y * tile_cols + tile_x;
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
	const float* means2D,
	const float4* cov2D,
	const float* colors,
	const float* opacities,
	int image_width,
	int image_height,
	float* output
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;

	if (px >= image_width || py >= image_height) return;

	float2 pixel = make_float2(px + 0.5f, py + 0.5f);

	float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float transmittance = 1.0f;

	uint2 range = tile_ranges[tile_idx];
	int start_idx = range.x;
	int end_idx = range.y;

	for (int i = start_idx; i < end_idx; i++) {
		int idx = tiled_gaussian_values_sorted[i];

		float2 mean = ((float2*)means2D)[idx];
		float4 cov = cov2D[idx];
		float weight = evaluate_gaussian_2d(pixel, mean, cov);

		float opacity = opacities[idx];
		float alpha = opacity * weight;

		if (alpha < 1e-4f) continue;

		float3 color = ((float3*) colors)[idx];

		accumulated_color.x += alpha * transmittance * color.x;
		accumulated_color.y += alpha * transmittance * color.y;
		accumulated_color.z += alpha * transmittance * color.z;

		transmittance *= (1.0f - alpha);

		if (transmittance < 1e-3f) break;
	}

	int pixel_idx = py * image_width + px;
	((float4*)output)[pixel_idx] = accumulated_color;
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
	const int TILE_SIZE = 16;

	int num_gaussians = means3D.size(0);
	int num_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
	int num_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;

	// These came from python and thus are torch::Tensor types
	float* means3D_ptr = means3D.data_ptr<float>();
	float* scales_ptr = scales.data_ptr<float>();
	float* quaternions_ptr = quaternions.data_ptr<float>();
	float* opacities_ptr = opacities.data_ptr<float>();
	float* colors_ptr = colors.data_ptr<float>();
	float* world_to_cam_matrix_ptr = world_to_cam_matrix.data_ptr<float>();

	// We just need the output to be a torch tensor, but the rest can be regular dtypes
	float4* gaussian_data_ptr;  // Packed for duplication: xy=means2D, z=radius, w=depth
	float2* means2D_ptr;  // Separate for render_gaussians
	float4* cov2D_ptr;
	int* tiles_touched_ptr;

	cudaMalloc(&gaussian_data_ptr, num_gaussians * sizeof(float4));
	cudaMalloc(&means2D_ptr, num_gaussians * sizeof(float2));
	cudaMalloc(&cov2D_ptr, num_gaussians * sizeof(float4));
	cudaMalloc(&tiles_touched_ptr, num_gaussians * sizeof(int));

	// Step 1: Project gaussians from 3D world space to 2D screen space
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
		TILE_SIZE,
		num_tiles_x,
		num_tiles_y,
		gaussian_data_ptr,
		means2D_ptr,
		cov2D_ptr,
		tiles_touched_ptr
	);
	check_cuda_error("project_gaussians");

	/*
	Step 2: Sort gaussians by depth per tile.

	Ok, this part is a bit confusing so buckle up. To render a pixel, we need to know which gaussians influence it.
	Duh. It also needs to be in depth order since we render front to back.

	Naively: you globally sort all the gaussians by depth. Then your renderer, for each pixel, goes through all 
	gaussians front to back to compute their impact. That is painfully slow. 4k image res * 50k gaussians is
	414 BILLION comparisons. I actually did that at first, and it took about 6 seconds to do one forward pass.

	But, pixels should only check gaussians that actually matter (e.g., top left isn't checking a bottom right
	gaussian). Then each pixel might only be checking tens of gaussians. Not tens of thousands. We do this by
	tiling the image and, per tile, checking what gaussians actually overlap it. Great, that sounds faster. How
	does it happen?

	The end goal: have some list tiled_gaussian_ids_sorted that contains gaussiand ids like:
	[tile0_depth0.1_gaussian123, tile0_depth1.5_gaussian456, ..., tileK_depth23.1_gaussian789]
	sorted first by tile, then by depth. Only gaussians that touch the tile should be in the list. In this way, 
	we've basically created per tile gaussian lists. Then, we may have another list tiled_ranges such that 
	tiled_ranges[tile_idx] tells us the start and end index in tiled_gaussian_ids_sorted that we need to check. 
	If we can have these two things, then we can check only	relevant gaussians within each tile.

	So how do we get those two lists? Like this (note that gaussian ID/GID just refers to the idx of the gaussian
	in means2D and cov2D):
	0. We know each gaussian touches tiles_touched[GID] and all times in a bounding box from tile_min[GID] to
	tile_max[GID].

	1. We compute a list offsets that is a prefix sum of tiles_touched. We also compute total_duplicates to be
	the sum of tiles_touched (or more efficiently, offsets[-1] + tiles_touched[-1]).

	2. We allocate keys and values lists of length total_duplicates that will store the key of a gaussian (see
	the next step) and the value, which will be its GID.

	3. We know which tiles each gaussian touches from tile_min and tile_max. We then can "duplicate" gaussians
	by creating total_duplicates keys like (tile_idx | depth). Keys should be like:
	[tile32_depth10.1, tile33_depth10.1, ...]
	When writing to keys, we can use offsets[GID] to know which index to start writing to. We'll use depth[GID]
	to know the depth, obviously. We also will store GID in the values list to map keys to gaussians.

	4. Sort those lists by sorting keys first on tile ID, then on depth. We now have tiled_gaussian_keys_sorted
	and tiled_gaussian_values_sorted.

	5. Traverse the tiled_gaussian_keys_sorted and extract the tile IDs to build tiled_ranges.

	Why tile at all and not just do it per pixel? We're already going to be making TONS of global memory calls.
	By tiling, we can take advantage of SMEM to ideally make this operation less memory intensie. Sure, some 
	pixels will ignore some gaussians. But at least we have far fewer unique DRAM calls and DRAM requirements.
	*/
	int* offsets_ptr;
	cudaMalloc(&offsets_ptr, num_gaussians * sizeof(int));

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

	uint64_t* tiled_gaussian_keys;
	int* tiled_gaussian_values;

	cudaMalloc(&tiled_gaussian_keys, total_duplicates * sizeof(uint64_t));
	cudaMalloc(&tiled_gaussian_values, total_duplicates * sizeof(int));

	// Optimization of this kernel was super fun, see the kernel for info
	int blocks_reverse = (total_duplicates + threads - 1) / threads;
	duplicate_gaussians<<<blocks_reverse, threads>>>(
		total_duplicates,
		num_gaussians,
		offsets_ptr,
		gaussian_data_ptr,
		TILE_SIZE,
		num_tiles_x,
		num_tiles_y,
		num_tiles_x,
		tiled_gaussian_keys,
		tiled_gaussian_values
	);
	check_cuda_error("duplicate_gaussians");

	uint64_t* tiled_gaussian_keys_sorted;
	int* tiled_gaussian_values_sorted;
	cudaMalloc(&tiled_gaussian_keys_sorted, total_duplicates * sizeof(uint64_t));
	cudaMalloc(&tiled_gaussian_values_sorted, total_duplicates * sizeof(int));

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	// Radix sort needs temporary storage. We don't know exactly how much, but if you pass it 
	// a nullptr it will automatically decide how much it needs.
	cub::DeviceRadixSort::SortPairs(
		d_temp_storage, temp_storage_bytes,
		tiled_gaussian_keys, tiled_gaussian_keys_sorted,
		tiled_gaussian_values, tiled_gaussian_values_sorted,
		total_duplicates
	);

	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(
		d_temp_storage, temp_storage_bytes,
		tiled_gaussian_keys, tiled_gaussian_keys_sorted,
		tiled_gaussian_values, tiled_gaussian_values_sorted,
		total_duplicates
	);

	cudaFree(d_temp_storage);
	check_cuda_error("radix_sort");

	uint2* tile_ranges;
	cudaMalloc(&tile_ranges, num_tiles_x * num_tiles_y * sizeof(uint2));
	cudaMemset(tile_ranges, 0, num_tiles_x * num_tiles_y * sizeof(uint2));

	int blocks_ranges = (total_duplicates + threads - 1) / threads;
	identify_tile_ranges<<<blocks_ranges, threads>>>(
		total_duplicates,
		tiled_gaussian_keys_sorted,
		tile_ranges
	);
	check_cuda_error("identify_tile_ranges");

	// Step 3: alpha blend/render the gaussians
	// We use a 4th channel to allow vectorized writes, but we don't actually care about it
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor output = torch::zeros({image_height, image_width, 4}, options);

	float* output_ptr = output.data_ptr<float>();

	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(num_tiles_x, num_tiles_y);

	render_gaussians<<<grid, block>>>(
		tile_ranges,
		tiled_gaussian_values_sorted,
		(float*)means2D_ptr,
		cov2D_ptr,
		colors_ptr,
		opacities_ptr,
		image_width,
		image_height,
		output_ptr
	);
	check_cuda_error("render_gaussians");

	cudaFree(gaussian_data_ptr);
	cudaFree(means2D_ptr);
	cudaFree(cov2D_ptr);
	cudaFree(tiles_touched_ptr);
	cudaFree(offsets_ptr);
	cudaFree(tiled_gaussian_keys);
	cudaFree(tiled_gaussian_values);
	cudaFree(tiled_gaussian_keys_sorted);
	cudaFree(tiled_gaussian_values_sorted);
	cudaFree(tile_ranges);

	// Drop the useless 4th channel here, torch can do this for free functionally
	return output.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
}