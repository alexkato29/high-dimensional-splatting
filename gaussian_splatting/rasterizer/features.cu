#include <torch/extension.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "api.h"


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
		case 32: std::apply(launch_render_features<32, 1, 256, 128>, args); break;
		case 64: std::apply(launch_render_features<64, 1, 256, 128>, args); break;
		case 128: std::apply(launch_render_features<128, 2, 128, 64>, args); break;
		default: TORCH_CHECK(false, "features must have 32, 64 or 128 channels, got ", D);
	}
	check_cuda_error("render_features");
	return {output, final_transmittance, n_contrib};
}

// Each block owns one tile and WIDTH of its D channels. The geometry gradients are linear in the
// features, so the slices of a tile add theirs up through the atomics in any order.
template <int D, int WIDTH, int BATCH>
__global__ void __launch_bounds__(RENDER_BATCH) render_features_backward(
	const uint2* tile_ranges,
	const int* values_sorted,
	const float2* means2D,
	const float4* conic,
	const float4* color_opacity,
	const float4* features,
	const float* final_transmittance,
	const int* n_contrib,
	const float* grad_output,
	int image_width,
	int image_height,
	float* grad_means2D,
	float* grad_conics,
	float* grad_features,
	float* grad_opacities
) {
	constexpr int SLICES = D / WIDTH;
	constexpr int VEC4_PER_ROW = D / 4;
	constexpr int VEC4_PER_SLICE = WIDTH / 4;
	constexpr int CHANNELS_PER_LANE = WIDTH / 32;
	static_assert(D % WIDTH == 0, "slices must split the channels evenly");
	static_assert(WIDTH % 32 == 0, "each lane must own the same number of channels");
	static_assert(BATCH % 32 == 0, "culling tests 32 gaussians per ballot");

	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp = tid >> 5;

	const int tile_idx = blockIdx.x / SLICES;
	const int channel0 = (blockIdx.x % SLICES) * WIDTH;
	const int tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
	const int tile_x0 = (tile_idx % tiles_x) * TILE_SIZE;
	const int tile_y0 = (tile_idx / tiles_x) * TILE_SIZE;

	const int px = tile_x0 + tid % TILE_SIZE;
	const int py = tile_y0 + tid / TILE_SIZE;
	const bool inside = px < image_width && py < image_height;
	const float2 pixel = make_float2(px + 0.5f, py + 0.5f);
	const int pixel_idx = py * image_width + px;

	const float4 rect = warp_rect(pixel);

	__shared__ int s_id[BATCH];
	__shared__ float2 s_mean[BATCH];
	__shared__ float4 s_conic[BATCH];
	__shared__ float2 s_extent[BATCH];
	// BATCH * VEC4_PER_SLICE float4s, sized at launch because 128 gaussians of 128 channels pass the 48 KB static cap.
	extern __shared__ float4 s_feat4[];
	__shared__ float s_weight[RENDER_BATCH];
	__shared__ int s_last_contributor;
	const float* s_feat = reinterpret_cast<const float*>(s_feat4);

	// Cooperatively load dL/dpixel into registers s.t. each thread grabs multiple channels of each pixel.
	// grad[i][p] is dL/dpixel for channel lane + 32*i of the warp's pixel p. Remember that, when D >= 1,
	// we render two images of shape [H, W, 3] (the rgb image) and [H, W, D] (the feature map). We are,
	// in this kernel, only concerned with pixels from the second.
	float grad[CHANNELS_PER_LANE][32];
	#pragma unroll
	for (int p = 0; p < 32; p++) {
		int x = tile_x0 + (warp * 32 + p) % TILE_SIZE;
		int y = tile_y0 + (warp * 32 + p) / TILE_SIZE;
		bool in = x < image_width && y < image_height;
		#pragma unroll
		for (int i = 0; i < CHANNELS_PER_LANE; i++)
			grad[i][p] = in ? grad_output[D * (y * image_width + x) + channel0 + lane + 32 * i] : 0.0f;
	}
	float transmittance = inside ? final_transmittance[pixel_idx] : 0.0f;
	int last_contributor = inside ? n_contrib[pixel_idx] : 0;

	// Start below the block's deepest contributor, as render_gaussians_backward does.
	uint2 range = tile_ranges[tile_idx];
	if (tid == 0) s_last_contributor = 0;
	__syncthreads();
	atomicMax(&s_last_contributor, last_contributor);
	__syncthreads();
	int num_todo = min((int)range.y - (int)range.x, s_last_contributor);

	float rec = 0.0f;
	for (int batch_end = num_todo; batch_end > 0; batch_end -= BATCH) {
		int batch_count = min(BATCH, batch_end);
		int batch_start = batch_end - batch_count;

		__syncthreads();
		for (int i = tid; i < batch_count; i += RENDER_BATCH) {
			int g = values_sorted[range.x + batch_start + i];
			s_id[i] = g;
			s_mean[i] = means2D[g];
			float4 c = conic[g];
			float opacity = color_opacity[g].w;
			s_conic[i] = make_float4(c.x, c.y, c.z, opacity);
			s_extent[i] = culling_extent(c, opacity);
		}

		__syncthreads();
		for (int i = tid; i < batch_count * VEC4_PER_SLICE; i += RENDER_BATCH) {
			int k = i / VEC4_PER_SLICE;
			s_feat4[i] = features[(size_t)s_id[k] * VEC4_PER_ROW + (channel0 / 4) + (i - k * VEC4_PER_SLICE)];
		}
		__syncthreads();

		// Walk the batch back to front. (batch_count - 1) / 32 * 32 rounds down to the closest smaller
		// multiple of 32 so each lane has a
		for (int j0 = (batch_count - 1) / 32 * 32; j0 >= 0; j0 -= 32) {
			int jl = j0 + lane;

			// Each thread owns a set of channels from each gaussian. Thus, each thread must know if each
			// gaussian is active (thus why a ballot sync is required).
			unsigned mask = __ballot_sync(FULL_MASK, jl < batch_count && box_hits(s_mean[jl], s_extent[jl], rect));
			while (mask) {
			    // Find the deepest gaussian in our mask, then flip it to a zero to mark it as visited.
				int bit = 31 - __clz(mask);
				mask &= ~(1u << bit);
				int j = j0 + bit;

				// Skip out of bounds pixels or too deep of gaussians.
				bool contributes = inside && batch_start + j < last_contributor;
				float alpha = 0.0f, gauss = 0.0f, raw_alpha = 0.0f, dx = 0.0f, dy = 0.0f;
				float4 c = s_conic[j];

				// Skip gaussians that don't influence the thread's associated pixel.
				if (contributes) {
					float2 mean = s_mean[j];
					dx = pixel.x - mean.x;
					dy = pixel.y - mean.y;
					float mahalanobis = dx * (c.x * dx + c.y * dy) + dy * (c.y * dx + c.z * dy);
					gauss = __expf(-0.5f * mahalanobis);
					raw_alpha = c.w * gauss;
					alpha = fminf(MAX_ALPHA, raw_alpha);
					contributes = alpha >= MIN_ALPHA;
				}
				if (!__any_sync(FULL_MASK, contributes)) continue;

				// f \dot dL/dpixel for all 32 pixels. Each lane computes its partial over its channels, then
				// it's summed across lanes w/ a reduce scatter. Afterward, lane p holds the dot for pixel p.
				float f[CHANNELS_PER_LANE];
				#pragma unroll
				for (int i = 0; i < CHANNELS_PER_LANE; i++) f[i] = s_feat[j * WIDTH + lane + 32 * i];
				float partial[32];
				#pragma unroll
				for (int p = 0; p < 32; p++) {
					partial[p] = 0.0f;
					#pragma unroll
					for (int i = 0; i < CHANNELS_PER_LANE; i++) partial[p] += f[i] * grad[i][p];
				}
				float dot = reduce_scatter<32, 32>(partial, lane);

				// Compute dL/dAlpha, which the derivatives of opacity/means/conics depend on.
				float dL_dalpha = 0.0f;
				float weight = 0.0f;
				if (contributes) {
					float inv_one_minus_alpha = __fdividef(1.0f, 1.0f - alpha);
					transmittance *= inv_one_minus_alpha;
					weight = transmittance * alpha;
					dL_dalpha = transmittance * (dot - rec);
					rec = alpha * dot + (1.0f - alpha) * rec;
				}

				// Compute the partials of opacity, means, and conics for the gaussian wrt each pixel.
				// Last two elements of v are never used and exist purely as padding for reduce scatter,
				// which mandates the input value contain a power of two elements.
				float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
				// A clamped alpha no longer depends on the gaussian so nothing flows past it.
				if (contributes && raw_alpha <= MAX_ALPHA) {
				    // c.w is opacity.
					float dL_dG = c.w * dL_dalpha;
					v[0] = gauss * dL_dalpha;

					v[1] = dL_dG * gauss * (c.x * dx + c.y * dy);
					v[2] = dL_dG * gauss * (c.y * dx + c.z * dy);

					v[3] = dL_dG * gauss * -0.5f * dx * dx;
					v[4] = dL_dG * gauss * -dx * dy;
					v[5] = dL_dG * gauss * -0.5f * dy * dy;
				}

				// Total holds all partial derivates summed across the whole warp. for 0<=k<=5, each
				// lane 4k to 4k + 3 holds one of the derivatives. They're repeated since the
				// reduce scatter is undersaturated given its template parameters.
				float total = reduce_scatter<8, 32>(v, lane);
				int k = lane >> 2;

				int g = s_id[j];
				// k = 6, 7 will have no values because they were purely for padding the reduce scatter.
				if ((lane & 3) == 0 && k < 6) {
					float* dst = k == 0 ? &grad_opacities[g]
							   : k < 3 ? &grad_means2D[g * 2 + (k - 1)]
							   : &grad_conics[g * 3 + (k - 3)];
					atomicAdd(dst, total);
				}

				s_weight[tid] = weight;
				__syncwarp();
				float w[32];
				#pragma unroll
				for (int p = 0; p < 32; p++) w[p] = s_weight[warp * 32 + p];
				__syncwarp();

				#pragma unroll
				for (int i = 0; i < CHANNELS_PER_LANE; i++) {
					float dL_df = 0.0f;
					#pragma unroll
					for (int p = 0; p < 32; p++) dL_df += w[p] * grad[i][p];
					// Each lane writes to a different address with CHANNELS_PER_LANE values, so this
					// atomic is not as heavy hitting as it seems.
					atomicAdd(&grad_features[(size_t)g * D + channel0 + lane + 32 * i], dL_df);
				}
			}
		}
	}
}

template <int D, int WIDTH, int BATCH>
static void launch_render_features_backward(int num_tiles, const uint2* tile_ranges, const int* values_sorted,
											const float2* means2D, const float4* conic,
											const float4* color_opacity, const float4* features,
											const float* final_transmittance, const int* n_contrib,
											const float* grad_output, int width, int height,
											float* grad_means2D, float* grad_conics, float* grad_features,
											float* grad_opacities) {
	const int feature_bytes = BATCH * (WIDTH / 4) * sizeof(float4);
	auto kernel = render_features_backward<D, WIDTH, BATCH>;
	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, feature_bytes);
	kernel<<<num_tiles * (D / WIDTH), RENDER_BATCH, feature_bytes>>>(
		tile_ranges, values_sorted, means2D, conic, color_opacity, features, final_transmittance, n_contrib,
		grad_output, width, height, grad_means2D, grad_conics, grad_features, grad_opacities
	);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_features_backward(
	torch::Tensor grad_output,
	torch::Tensor features,
	torch::Tensor means2D,
	torch::Tensor conic,
	torch::Tensor color_opacity,
	torch::Tensor values_sorted,
	torch::Tensor tile_ranges,
	torch::Tensor final_transmittance,
	torch::Tensor n_contrib,
	int image_width,
	int image_height,
	int slice_width
) {
	TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(), "grad_output must be a contiguous CUDA tensor");

	const int num_gaussians = features.size(0);
	const int D = features.size(1);
	const int num_tiles = ((image_width + TILE_SIZE - 1) / TILE_SIZE) * ((image_height + TILE_SIZE - 1) / TILE_SIZE);

	auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor grad_features = torch::zeros({num_gaussians, D}, f32);
	torch::Tensor grad_means2D = torch::zeros({num_gaussians, 2}, f32);
	torch::Tensor grad_conics = torch::zeros({num_gaussians, 3}, f32);
	torch::Tensor grad_opacities = torch::zeros({num_gaussians, 1}, f32);

	auto args = std::make_tuple(
		num_tiles,
		reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
		values_sorted.data_ptr<int>(),
		reinterpret_cast<const float2*>(means2D.data_ptr<float>()),
		reinterpret_cast<const float4*>(conic.data_ptr<float>()),
		reinterpret_cast<const float4*>(color_opacity.data_ptr<float>()),
		reinterpret_cast<const float4*>(features.data_ptr<float>()),
		final_transmittance.data_ptr<float>(), n_contrib.data_ptr<int>(),
		grad_output.data_ptr<float>(), image_width, image_height,
		grad_means2D.data_ptr<float>(), grad_conics.data_ptr<float>(), grad_features.data_ptr<float>(),
		grad_opacities.data_ptr<float>()
	);

	switch (D * 1000 + slice_width) {
		case 32 * 1000 + 32: std::apply(launch_render_features_backward<32, 32, 128>, args); break;
		case 64 * 1000 + 64: std::apply(launch_render_features_backward<64, 64, 128>, args); break;
		case 64 * 1000 + 32: std::apply(launch_render_features_backward<64, 32, 128>, args); break;
		case 128 * 1000 + 128: std::apply(launch_render_features_backward<128, 128, 128>, args); break;
		case 128 * 1000 + 64: std::apply(launch_render_features_backward<128, 64, 128>, args); break;
		case 128 * 1000 + 32: std::apply(launch_render_features_backward<128, 32, 128>, args); break;
		default: TORCH_CHECK(false, "no feature backward for ", D, " channels in slices of ", slice_width);
	}
	check_cuda_error("render_features_backward");
	return {grad_features, grad_means2D, grad_conics, grad_opacities};
}
