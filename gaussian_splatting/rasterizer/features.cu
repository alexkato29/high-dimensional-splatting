#include <climits>
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

// Every lane agrees that the top left pixel is lane 0, and bottom right is lane 31.
__device__ inline float4 warp_rect(float2 pixel) {
	return make_float4(__shfl_sync(FULL_MASK, pixel.x, 0), __shfl_sync(FULL_MASK, pixel.x, 31),
					   __shfl_sync(FULL_MASK, pixel.y, 0), __shfl_sync(FULL_MASK, pixel.y, 31));
}

__device__ inline int2 warp_pixel(int warp, int lane) {
	return make_int2((warp & 1) * 8 + (lane & 7), (warp >> 1) * 4 + (lane >> 3));
}

// Hardcoded for a NVIDIA L4 GPU.
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

constexpr int A_VALS_PER_LANE = MMA_M * MMA_K / WARP_SIZE;
constexpr int C_VALS_PER_LANE = MMA_M * MMA_N / WARP_SIZE;
constexpr int C_COLS_PER_LANE = 2;

constexpr int LANES_PER_GROUP = 32 / MMA_K;
constexpr int FLOAT4_SIZE = 4;

constexpr int W_STRIDE = WARP_SIZE + 8;
constexpr int MMAS_PER_WARP = WARP_SIZE / MMA_M;

// c += a . b on one TF32 tile, register layouts per the PTX ISA "Matrix Fragments for mma.m16n8k8".
__device__ inline void mma_tf32(float* c, const float* a, float b0, float b1) {
	asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
		: "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])), "r"(__float_as_uint(a[2])), "r"(__float_as_uint(a[3])),
		  "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1)));
}

// Compute the pixels a given lane works on, with indicator variables `m` and `half`.
__device__ inline int fragment_pixel(int m, int half, int lane) {
	return m * MMA_M + half * (MMA_M / 2) + lane / LANES_PER_GROUP;
}

// Channel of column n in the j-th mma across.
template <int MMAS_PER_WIDTH>
__device__ inline int column_channel(int n, int j) {
	return n * MMAS_PER_WIDTH + j;
}

// Blends the warp's group of MMA_K gaussians into acc, acc += weights . features.
template <int MMAS_PER_WIDTH>
__device__ inline void blend_group(float (&acc)[MMAS_PER_WARP][MMAS_PER_WIDTH][C_VALS_PER_LANE],
								   const float* w, const int* slots, const float* feat, int lane) {
	constexpr int WIDTH = MMAS_PER_WIDTH * MMA_N;
	__syncwarp();
	const int group_id = lane / LANES_PER_GROUP;
	const int lane_in_group = lane % LANES_PER_GROUP;
	const int channel = column_channel<MMAS_PER_WIDTH>(group_id, 0);

	float b0[MMAS_PER_WIDTH], b1[MMAS_PER_WIDTH];

	const float* row0 = feat + slots[lane_in_group] * WIDTH + channel;
	const float* row1 = feat + slots[lane_in_group + MMA_K / 2] * WIDTH + channel;

	#pragma unroll
	for (int q = 0; q < MMAS_PER_WIDTH / FLOAT4_SIZE; q++) {
		float4 v0 = reinterpret_cast<const float4*>(row0)[q];
		float4 v1 = reinterpret_cast<const float4*>(row1)[q];
		const int k = FLOAT4_SIZE * q;
		b0[k] = v0.x; b0[k + 1] = v0.y; b0[k + 2] = v0.z; b0[k + 3] = v0.w;
		b1[k] = v1.x; b1[k + 1] = v1.y; b1[k + 2] = v1.z; b1[k + 3] = v1.w;
	}

	const float* w0 = w + lane_in_group * W_STRIDE;
	const float* w1 = w + (lane_in_group + MMA_K / 2) * W_STRIDE;
	#pragma unroll
	for (int m = 0; m < MMAS_PER_WARP; m++) {
		const int p0 = fragment_pixel(m, 0, lane);
		const int p1 = fragment_pixel(m, 1, lane);
		float a[A_VALS_PER_LANE] = {w0[p0], w0[p1], w1[p0], w1[p1]};
		#pragma unroll
		for (int j = 0; j < MMAS_PER_WIDTH; j++) mma_tf32(acc[m][j], a, b0[j], b1[j]);
	}
	__syncwarp();
}

template <int D, int BATCH, int WIDTH>
__global__ void __launch_bounds__(BLOCK_SIZE) render_features(
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
	constexpr int VEC4_PER_ROW = D / FLOAT4_SIZE;
	constexpr int VEC4_PER_SLICE = WIDTH / FLOAT4_SIZE;
	constexpr int SLICES = D / WIDTH;
	constexpr int MMAS_PER_WIDTH = WIDTH / MMA_N;
	static_assert(D % WIDTH == 0, "slices must split the channels evenly");
	static_assert(MMAS_PER_WIDTH % FLOAT4_SIZE == 0, "each lane must load its B channels as whole float4s");
	static_assert(BATCH % WARP_SIZE == 0, "culling tests 32 gaussians per ballot");

	const int tid = threadIdx.x;
	const int lane = tid % WARP_SIZE;
	const int warp = tid / WARP_SIZE;

	const int tile_idx = blockIdx.x / SLICES;
	const int channel0 = (blockIdx.x % SLICES) * WIDTH;
	const int tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;

	const int2 local = warp_pixel(warp, lane);
	const int px = (tile_idx % tiles_x) * TILE_SIZE + local.x;
	const int py = (tile_idx / tiles_x) * TILE_SIZE + local.y;
	const bool inside = px < image_width && py < image_height;
	const float2 pixel = make_float2(px + 0.5f, py + 0.5f);
	const float4 rect = warp_rect(pixel);

	__shared__ int s_id[BATCH];
	__shared__ float2 s_mean[BATCH];
	__shared__ float4 s_conic[BATCH];
	__shared__ float2 s_extent[BATCH];
	extern __shared__ float4 s_feat[];
	__shared__ float s_w[BLOCK_SIZE / WARP_SIZE][MMA_K * W_STRIDE];
	__shared__ int s_slots[BLOCK_SIZE / WARP_SIZE][MMA_K];
	const float* feat = reinterpret_cast<const float*>(s_feat);

	float* w_group = s_w[warp];
	int* slot_group = s_slots[warp];

	uint2 range = tile_ranges[tile_idx];
	int num_todo = (int)range.y - (int)range.x;

	float acc[MMAS_PER_WARP][MMAS_PER_WIDTH][C_VALS_PER_LANE];
	#pragma unroll
	for (int m = 0; m < MMAS_PER_WARP; m++)
		#pragma unroll
		for (int j = 0; j < MMAS_PER_WIDTH; j++)
			#pragma unroll
			for (int i = 0; i < C_VALS_PER_LANE; i++) acc[m][j][i] = 0.0f;

	float transmittance = 1.0f;
	bool done = !inside;
	int last_contributor = 0;
	int count = 0;

	for (int batch_start = 0; batch_start < num_todo; batch_start += BATCH) {
		if (__syncthreads_count(done) == BLOCK_SIZE) break;

		int batch_count = min(BATCH, num_todo - batch_start);
		for (int i = tid; i < batch_count; i += BLOCK_SIZE) {
			int gaussian_id = values_sorted[range.x + batch_start + i];
			s_id[i] = gaussian_id;
			s_mean[i] = means2D[gaussian_id];
			float4 c = conic[gaussian_id];
			float opacity = color_opacity[gaussian_id].w;
			s_conic[i] = make_float4(c.x, c.y, c.z, opacity);
			s_extent[i] = culling_extent(c, opacity);
		}
		__syncthreads();
		for (int i = tid; i < batch_count * VEC4_PER_SLICE; i += BLOCK_SIZE) {
			int slot = i / VEC4_PER_SLICE;
			s_feat[i] = features[(size_t)s_id[slot] * VEC4_PER_ROW + channel0 / FLOAT4_SIZE + i % VEC4_PER_SLICE];
		}
		__syncthreads();

		for (int group_start = 0; group_start < batch_count; group_start += WARP_SIZE) {
			if (__all_sync(FULL_MASK, done)) break;
			int candidate = group_start + lane;
			unsigned mask = __ballot_sync(FULL_MASK, candidate < batch_count && box_hits(s_mean[candidate], s_extent[candidate], rect));
			while (mask) {
				int slot = group_start + __ffs(mask) - 1;
				mask &= mask - 1;

				bool hit = false;
				float w = 0.0f;

				if (!done) {
					float2 mean = s_mean[slot];
					float4 c = s_conic[slot];
					float dx = pixel.x - mean.x;
					float dy = pixel.y - mean.y;
					float mahalanobis = dx * (c.x * dx + c.y * dy) + dy * (c.y * dx + c.z * dy);
					float alpha = fminf(MAX_ALPHA, c.w * __expf(-0.5f * mahalanobis));

					if (!(alpha < MIN_ALPHA)) {
						float test_transmittance = transmittance * (1.0f - alpha);
						if (test_transmittance < MIN_TRANSMITTANCE) {
							done = true;
						} else {
							hit = true;
							w = alpha * transmittance;
							transmittance = test_transmittance;
							last_contributor = batch_start + slot + 1;
						}
					}
				}

				if (!__any_sync(FULL_MASK, hit)) continue;
				w_group[count * W_STRIDE + lane] = w;
				slot_group[count++] = slot;

				if (count == MMA_K) {
					blend_group<MMAS_PER_WIDTH>(acc, w_group, slot_group, feat, lane);
					count = 0;
				}
			}
		}

		if (count == 0) continue;
		for (int i = count; i < MMA_K; i++) {
			w_group[i * W_STRIDE + lane] = 0.0f;
			slot_group[i] = 0;
		}
		count = 0;
		blend_group<MMAS_PER_WIDTH>(acc, w_group, slot_group, feat, lane);
	}

	const int lane_in_group = lane % LANES_PER_GROUP;
	const int tile_x0 = (tile_idx % tiles_x) * TILE_SIZE;
	const int tile_y0 = (tile_idx / tiles_x) * TILE_SIZE;

	#pragma unroll
	for (int m = 0; m < MMAS_PER_WARP; m++) {
		#pragma unroll
		for (int i = 0; i < C_VALS_PER_LANE; i++) {
			int2 p = warp_pixel(warp, fragment_pixel(m, i / C_COLS_PER_LANE, lane));
			int x = tile_x0 + p.x;
			int y = tile_y0 + p.y;
			if (x >= image_width || y >= image_height) continue;

			int column = C_COLS_PER_LANE * lane_in_group + i % C_COLS_PER_LANE;
			int channel = channel0 + column_channel<MMAS_PER_WIDTH>(column, 0);
			float4* dst = reinterpret_cast<float4*>(&output[(y * image_width + x) * D + channel]);
			#pragma unroll
			for (int q = 0; q < MMAS_PER_WIDTH / FLOAT4_SIZE; q++) {
				const int j = FLOAT4_SIZE * q;
				dst[q] = make_float4(acc[m][j][i], acc[m][j + 1][i], acc[m][j + 2][i], acc[m][j + 3][i]);
			}
		}
	}

	if (inside && blockIdx.x % SLICES == 0) {
		int pixel_idx = py * image_width + px;
		final_transmittance[pixel_idx] = transmittance;
		n_contrib[pixel_idx] = last_contributor;
	}
}

template <int D, int BATCH, int WIDTH>
static void launch_render_features(int num_tiles, const uint2* tile_ranges, const int* values_sorted,
								   const float2* means2D, const float4* conic, const float4* color_opacity,
								   const float4* features, int width, int height, float* output,
								   float* final_transmittance, int* n_contrib) {
	const int blocks = num_tiles * (D / WIDTH);
	const int feature_bytes = BATCH * (WIDTH / FLOAT4_SIZE) * sizeof(float4);
	auto kernel = render_features<D, BATCH, WIDTH>;
	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, feature_bytes);
	kernel<<<blocks, BLOCK_SIZE, feature_bytes>>>(
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
	// Both feature kernels index the [H, W, D] image with int arithmetic.
	TORCH_CHECK((int64_t)D * image_width * image_height <= INT_MAX,
				"a ", image_width, "x", image_height, " image of ", D, " channels overflows the feature kernels' int indices");

	auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	torch::Tensor output = torch::empty({image_height, image_width, D}, f32);
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

	// Slices of at most 64 channels keep a thread near 128 registers, so two blocks fit per SM on an L4.
	switch (D) {
		case 32: std::apply(launch_render_features<32, 128, 32>, args); break;
		case 64: std::apply(launch_render_features<64, 128, 64>, args); break;
		case 128: std::apply(launch_render_features<128, 128, 64>, args); break;
		case 256: std::apply(launch_render_features<256, 128, 64>, args); break;
		case 512: std::apply(launch_render_features<512, 128, 64>, args); break;
		default: TORCH_CHECK(false, "features must have 32, 64, 128, 256 or 512 channels, got ", D);
	}
	check_cuda_error("render_features");
	return {output, final_transmittance, n_contrib};
}

// Each block owns one tile and WIDTH of its D channels. The geometry gradients are linear in the
// features, so the slices of a tile add theirs up through the atomics in any order.
template <int D, int WIDTH, int BATCH>
__global__ void __launch_bounds__(BLOCK_SIZE) render_features_backward(
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
	constexpr int VEC4_PER_ROW = D / FLOAT4_SIZE;
	constexpr int VEC4_PER_SLICE = WIDTH / FLOAT4_SIZE;
	constexpr int CHANNELS_PER_LANE = WIDTH / WARP_SIZE;
	static_assert(D % WIDTH == 0, "slices must split the channels evenly");
	static_assert(WIDTH % WARP_SIZE == 0, "each lane must own the same number of channels");
	static_assert(BATCH % WARP_SIZE == 0, "culling tests 32 gaussians per ballot");

	const int tid = threadIdx.x;
	const int lane = tid % WARP_SIZE;
	const int warp = tid / WARP_SIZE;

	const int tile_idx = blockIdx.x / SLICES;
	const int channel0 = (blockIdx.x % SLICES) * WIDTH;
	const int tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
	const int tile_x0 = (tile_idx % tiles_x) * TILE_SIZE;
	const int tile_y0 = (tile_idx / tiles_x) * TILE_SIZE;

	const int2 local = warp_pixel(warp, lane);
	const int px = tile_x0 + local.x;
	const int py = tile_y0 + local.y;
	const bool pixel_inside = px < image_width && py < image_height;
	const float2 pixel = make_float2(px + 0.5f, py + 0.5f);
	const int pixel_idx = py * image_width + px;

	const float4 rect = warp_rect(pixel);

	__shared__ int s_id[BATCH];
	__shared__ float2 s_mean[BATCH];
	__shared__ float4 s_conic[BATCH];
	__shared__ float2 s_extent[BATCH];
	extern __shared__ float4 s_feat4[];
	__shared__ float s_weight[BLOCK_SIZE];
	__shared__ int s_last_contributor;
	const float* s_feat = reinterpret_cast<const float*>(s_feat4);

	float grad[CHANNELS_PER_LANE][WARP_SIZE];
	#pragma unroll
	for (int p = 0; p < WARP_SIZE; p++) {
		int2 lp = warp_pixel(warp, p);
		int x = tile_x0 + lp.x;
		int y = tile_y0 + lp.y;
		bool in = x < image_width && y < image_height;
		#pragma unroll
		for (int i = 0; i < CHANNELS_PER_LANE; i++)
			grad[i][p] = in ? grad_output[D * (y * image_width + x) + channel0 + lane + WARP_SIZE * i] : 0.0f;
	}
	float transmittance = pixel_inside ? final_transmittance[pixel_idx] : 0.0f;
	int last_contributor = pixel_inside ? n_contrib[pixel_idx] : 0;

	// We can skip any gaussians that no pixel in the block will ever touch.
	uint2 range = tile_ranges[tile_idx];
	if (tid == 0) s_last_contributor = 0;
	__syncthreads();
	atomicMax(&s_last_contributor, last_contributor);
	__syncthreads();
	int num_todo = min((int)range.y - (int)range.x, s_last_contributor);

	float behind_dot = 0.0f;
	for (int batch_end = num_todo; batch_end > 0; batch_end -= BATCH) {
		int batch_count = min(BATCH, batch_end);
		int batch_start = batch_end - batch_count;

		__syncthreads();
		for (int i = tid; i < batch_count; i += BLOCK_SIZE) {
			int gaussian_id = values_sorted[range.x + batch_start + i];
			s_id[i] = gaussian_id;
			s_mean[i] = means2D[gaussian_id];
			float4 c = conic[gaussian_id];
			float opacity = color_opacity[gaussian_id].w;
			s_conic[i] = make_float4(c.x, c.y, c.z, opacity);
			s_extent[i] = culling_extent(c, opacity);
		}

		__syncthreads();
		for (int i = tid; i < batch_count * VEC4_PER_SLICE; i += BLOCK_SIZE) {
			int slot = i / VEC4_PER_SLICE;
			s_feat4[i] = features[(size_t)s_id[slot] * VEC4_PER_ROW + (channel0 / FLOAT4_SIZE) + (i - slot * VEC4_PER_SLICE)];
		}
		__syncthreads();

		for (int group_start = (batch_count - 1) / WARP_SIZE * WARP_SIZE; group_start >= 0; group_start -= WARP_SIZE) {
			int candidate = group_start + lane;
			unsigned mask = __ballot_sync(FULL_MASK, candidate < batch_count && box_hits(s_mean[candidate], s_extent[candidate], rect));
			while (mask) {
				int bit = 31 - __clz(mask);
				mask &= ~(1u << bit);
				int slot = group_start + bit;

				bool contributes = pixel_inside && batch_start + slot < last_contributor;
				float alpha = 0.0f, gauss = 0.0f, raw_alpha = 0.0f, dx = 0.0f, dy = 0.0f;
				float4 c = s_conic[slot];

				// Skip gaussians that don't influence the thread's associated pixel.
				if (contributes) {
					float2 mean = s_mean[slot];
					dx = pixel.x - mean.x;
					dy = pixel.y - mean.y;
					float mahalanobis = dx * (c.x * dx + c.y * dy) + dy * (c.y * dx + c.z * dy);
					gauss = __expf(-0.5f * mahalanobis);
					raw_alpha = c.w * gauss;
					alpha = fminf(MAX_ALPHA, raw_alpha);
					contributes = alpha >= MIN_ALPHA;
				}
				if (!__any_sync(FULL_MASK, contributes)) continue;

				float f[CHANNELS_PER_LANE];
				#pragma unroll
				for (int i = 0; i < CHANNELS_PER_LANE; i++) f[i] = s_feat[slot * WIDTH + lane + WARP_SIZE * i];
				float partial[WARP_SIZE];
				#pragma unroll
				for (int px = 0; px < WARP_SIZE; px++) {
					partial[px] = 0.0f;
					#pragma unroll
					for (int i = 0; i < CHANNELS_PER_LANE; i++) partial[px] += f[i] * grad[i][px];
				}
				float dot = reduce_scatter<WARP_SIZE, WARP_SIZE>(partial, lane);

				float dL_dalpha = 0.0f;
				float weight = 0.0f;
				if (contributes) {
					float inv_one_minus_alpha = __fdividef(1.0f, 1.0f - alpha);
					transmittance *= inv_one_minus_alpha;
					weight = transmittance * alpha;
					dL_dalpha = transmittance * (dot - behind_dot);
					behind_dot = alpha * dot + (1.0f - alpha) * behind_dot;
				}

				// Compute the partials of opacity, means, and conics for the gaussian wrt each pixel.
				// Last two elements of v are never used and exist purely as padding for reduce scatter,
				// which mandates the input value contain a power of two elements.
				float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
				// A clamped alpha no longer depends on the gaussian, so nothing flows past it.
				if (contributes && raw_alpha <= MAX_ALPHA) {
					float dL_dG = c.w * dL_dalpha;
					v[0] = gauss * dL_dalpha;

					v[1] = dL_dG * gauss * (c.x * dx + c.y * dy);
					v[2] = dL_dG * gauss * (c.y * dx + c.z * dy);

					v[3] = dL_dG * gauss * -0.5f * dx * dx;
					v[4] = dL_dG * gauss * -dx * dy;
					v[5] = dL_dG * gauss * -0.5f * dy * dy;
				}

				float total = reduce_scatter<8, WARP_SIZE>(v, lane);
				int k = lane >> 2;

				int gaussian_id = s_id[slot];
				// Again, k = 6, 7 will have no values because they were purely for padding the reduce scatter.
				if ((lane & 3) == 0 && k < 6) {
					float* dst = k == 0 ? &grad_opacities[gaussian_id]
							   : k < 3 ? &grad_means2D[gaussian_id * 2 + (k - 1)]
							   : &grad_conics[gaussian_id * 3 + (k - 3)];
					atomicAdd(dst, total);
				}

				s_weight[tid] = weight;
				__syncwarp();
				float w[WARP_SIZE];
				#pragma unroll
				for (int p = 0; p < WARP_SIZE; p++) w[p] = s_weight[warp * WARP_SIZE + p];
				__syncwarp();

				#pragma unroll
				for (int i = 0; i < CHANNELS_PER_LANE; i++) {
					float dL_df = 0.0f;
					#pragma unroll
					for (int p = 0; p < WARP_SIZE; p++) dL_df += w[p] * grad[i][p];
					atomicAdd(&grad_features[(size_t)gaussian_id * D + channel0 + lane + WARP_SIZE * i], dL_df);
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
	const int feature_bytes = BATCH * (WIDTH / FLOAT4_SIZE) * sizeof(float4);
	auto kernel = render_features_backward<D, WIDTH, BATCH>;
	cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, feature_bytes);
	kernel<<<num_tiles * (D / WIDTH), BLOCK_SIZE, feature_bytes>>>(
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
		case 256 * 1000 + 128: std::apply(launch_render_features_backward<256, 128, 128>, args); break;
		case 256 * 1000 + 64: std::apply(launch_render_features_backward<256, 64, 128>, args); break;
		case 512 * 1000 + 128: std::apply(launch_render_features_backward<512, 128, 128>, args); break;
		case 512 * 1000 + 64: std::apply(launch_render_features_backward<512, 64, 128>, args); break;
		default: TORCH_CHECK(false, "no feature backward for ", D, " channels in slices of ", slice_width);
	}
	check_cuda_error("render_features_backward");
	return {grad_features, grad_means2D, grad_conics, grad_opacities};
}
