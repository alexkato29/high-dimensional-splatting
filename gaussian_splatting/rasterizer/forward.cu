#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <cub/device/device_radix_sort.cuh>
#include "common.cuh"


__global__ void prepare_gaussians(
	int num_gaussians,
	const float2* means2D,
	const float* depths,
	const float* radii,
	const float* conics,
	const float* colors,
	const float* opacities,
	int num_tiles_x,
	int num_tiles_y,
	float4* gaussian_data,
	float4* conic,
	float4* color_opacity,
	int* tiles_touched
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_gaussians) return;

	// Culled gaussians have radius 0. Written as !(radius > 0) so a NaN radius is culled too.
	float radius = radii[idx];
	if (!(radius > 0.0f)) {
		tiles_touched[idx] = 0;
		return;
	}

	// duplicate_gaussians is memory bound and wants everything it needs in one vectorized load,
	// while render_gaussians only reads the means, so the means live in both places.
	float2 uv = means2D[idx];
	gaussian_data[idx] = make_float4(uv.x, uv.y, radius, depths[idx]);

	// Color and conic are 12B at 12B alignment, which the compiler must split into three LDG.E.
	// Padding each to a float4 (opacity parked in the color's spare lane) allows vectorized
	// access in the render loop.
	conic[idx] = make_float4(conics[idx*3 + 0], conics[idx*3 + 1], conics[idx*3 + 2], 0.0f);
	color_opacity[idx] = make_float4(
		colors[idx*3 + 0], colors[idx*3 + 1], colors[idx*3 + 2], opacities[idx]);

	float2 min_pixel = make_float2(uv.x - radius, uv.y - radius);
	float2 max_pixel = make_float2(uv.x + radius, uv.y + radius);

	int2 tile_min_coords = pixel_to_tile(min_pixel);
	int2 tile_max_coords = pixel_to_tile(max_pixel);

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

	int2 gaussian_min_tile_coords = pixel_to_tile(min_pixel);
	int2 gaussian_max_tile_coords = pixel_to_tile(max_pixel);

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
	const float* background,
	int image_width,
	int image_height,
	float* output,
	float* final_transmittance,
	int* n_contrib
) {
	const int block_size = blockDim.x * blockDim.y;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;

	// Out-of-bounds threads cannot return early, they still have to help fetch and, more
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

	int contributor = 0;
	int last_contributor = 0;

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
			contributor++;
			float2 mean = s_mean[j];
			float4 c = s_conic[j];

			// c = (conic.xx, conic.xy, conic.yy, unused), already inverted in projection.py.
			float dx = pixel.x - mean.x;
			float dy = pixel.y - mean.y;
			float mahalanobis = dx * (c.x * dx + c.y * dy) +
								dy * (c.y * dx + c.z * dy);
			float weight = __expf(-0.5f * mahalanobis);

			float4 color = s_color[j];
			float alpha = fminf(MAX_ALPHA, color.w * weight);

			if (alpha < MIN_ALPHA) continue;

			float test_transmittance = transmittance * (1.0f - alpha);
			if (test_transmittance < MIN_TRANSMITTANCE) {
				done = true;
				continue;
			}

			accumulated_color.x += alpha * transmittance * color.x;
			accumulated_color.y += alpha * transmittance * color.y;
			accumulated_color.z += alpha * transmittance * color.z;

			transmittance = test_transmittance;
			last_contributor = contributor;
		}
	}

	if (inside) {
		int pixel_idx = py * image_width + px;
		((float4*)output)[pixel_idx] = make_float4(
			accumulated_color.x + transmittance * background[0],
			accumulated_color.y + transmittance * background[1],
			accumulated_color.z + transmittance * background[2],
			0.0f);
		final_transmittance[pixel_idx] = transmittance;
		n_contrib[pixel_idx] = last_contributor;
	}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize(
	torch::Tensor means2D,
	torch::Tensor depths,
	torch::Tensor radii,
	torch::Tensor conics,
	torch::Tensor colors,
	torch::Tensor opacities,
	torch::Tensor background,
	int image_width,
	int image_height
) {
	auto check_input = [](const torch::Tensor& t, const char* name) {
		TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
		TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
	};
	check_input(means2D, "means2D");
	check_input(depths, "depths");
	check_input(radii, "radii");
	check_input(conics, "conics");
	check_input(colors, "colors");
	check_input(opacities, "opacities");
	check_input(background, "background");

	const int num_gaussians = means2D.size(0);
	const int num_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
	const int num_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;
	const int num_tiles = num_tiles_x * num_tiles_y;

	const float2* means2D_ptr = reinterpret_cast<const float2*>(means2D.data_ptr<float>());

	auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
	auto u8  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

	torch::Tensor gaussian_data = torch::empty({num_gaussians, 4}, f32);  // xy=means2D, z=radius, w=depth
	torch::Tensor conic         = torch::empty({num_gaussians, 4}, f32);
	torch::Tensor color_opacity = torch::empty({num_gaussians, 4}, f32);
	torch::Tensor tiles_touched = torch::empty({num_gaussians}, i32);

	float4* gaussian_data_ptr = reinterpret_cast<float4*>(gaussian_data.data_ptr<float>());
	float4* conic_ptr         = reinterpret_cast<float4*>(conic.data_ptr<float>());
	float4* color_opacity_ptr = reinterpret_cast<float4*>(color_opacity.data_ptr<float>());
	int*    tiles_touched_ptr = tiles_touched.data_ptr<int>();

	// Step 1: pack the projected gaussians and count the tiles each one touches.
	const int threads = 256;
	auto blocks_for = [threads](int n) { return (n + threads - 1) / threads; };

	prepare_gaussians<<<blocks_for(num_gaussians), threads>>>(
		num_gaussians,
		means2D_ptr,
		depths.data_ptr<float>(),
		radii.data_ptr<float>(),
		conics.data_ptr<float>(),
		colors.data_ptr<float>(),
		opacities.data_ptr<float>(),
		num_tiles_x,
		num_tiles_y,
		gaussian_data_ptr,
		conic_ptr,
		color_opacity_ptr,
		tiles_touched_ptr
	);
	check_cuda_error("prepare_gaussians");

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
		return {background.reshape({1, 1, 3}).expand({image_height, image_width, 3}).contiguous(),
				torch::ones({image_height, image_width}, f32),
				torch::zeros({image_height, image_width}, i32),
				torch::empty({0}, i32),
				torch::zeros({num_tiles, 2}, i32),
				conic,
				color_opacity};
	}



	torch::Tensor keys   = torch::empty({total_duplicates}, i64);
	torch::Tensor values = torch::empty({total_duplicates}, i32);

	uint64_t* tiled_gaussian_keys   = reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>());
	int*      tiled_gaussian_values = values.data_ptr<int>();

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
	torch::Tensor final_transmittance = torch::empty({image_height, image_width}, f32);
	torch::Tensor n_contrib = torch::empty({image_height, image_width}, i32);

	float* output_ptr = output.data_ptr<float>();

	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(num_tiles_x, num_tiles_y);

	render_gaussians<<<grid, block>>>(
		tile_ranges,
		tiled_gaussian_values_sorted,
		means2D_ptr,
		conic_ptr,
		color_opacity_ptr,
		background.data_ptr<float>(),
		image_width,
		image_height,
		output_ptr,
		final_transmittance.data_ptr<float>(),
		n_contrib.data_ptr<int>()
	);
	check_cuda_error("render_gaussians");

	// Drop the useless 4th channel here, torch can do this for free functionally
	torch::Tensor image = output.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
	return {image, final_transmittance, n_contrib, values_sorted, tile_ranges_t, conic, color_opacity};
}
