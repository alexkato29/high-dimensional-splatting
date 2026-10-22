@@ __device__ inline int2 warp_pixel(int warp, int lane) {
When performing warp culling, we use this function to ensure warp boxes are `8x4`. Why? It is the rectangle of area 32 with the smallest perimeter at 24. 

The worst case for a warp during is having a gaussian touch the edge. The whole warp must pursue a gaussian that only one of its pixels will touch. Minimizing the perimeter minimizes the size of the edge, so this change is mostly a theoretical "correctness". In practice, the observed speedup is small like 3-9%. But it does see 12.5% fewer passes (on the truck scene) on average... Always feels good when theory works.

@@ __global__ void __launch_bounds__(BLOCK_SIZE) render_features(
For long feature vectors (e.g., 512 dimensions), it is expensive on occupancy to stage the entire vector in shared memory. For a batch of 64 Gaussians, 64 Gaussians * 512 features / gaussian * 4 bytes / features = 131kb. Each SM only has 100kb total, so you can see how this becomes a problem... `WIDTH` splits up channels over blocks. When `WIDTH = 128`, and features are 512d, 4 blocks will each work on 128d of the feature vector simultaneously. Now, in SMEM: 64 Gaussians * 128 features / gaussian * 4 bytes / feature = 33kb. Far better, we can actually fit multiple blocks at this rate.

`WIDTH` also bounds register pressure, since `acc` holds `WIDTH` floats per thread in registers.

@@ float4 f = row[i];
Accumulate the overall rendered values of a given feature channel for a given pixel. This loop is pretty dead simple. Just scales each channel by the blending weight of the current gaussian in registers, reading the staged slice one `float4` at a time.

Parallelism: Each thread owns the `WIDTH` channels of its block's slice for a particular pixel.

@@ if (test_transmittance < MIN_TRANSMITTANCE) {
The parallelism here is deceptive, and this line actually can do repeated work. Why? `acc` holds final render values of a particular *pixel's* feature vector. `transmittance` is also per pixel. Now, for small `D`, the entire feature vector can fit comfortably into `acc`, so each thread really is just concerned with one unique pixel.

For large `D`, `acc` becomes only a subset of the whole feature vector, and multiple blocks (one per channel slice) now have to team up to take on a single pixel. But, `transmittance` necessarily only has one meaningful value per output pixel. There is no work to split up. That means that in the case where `k` slices are working on one pixel, they are *all computing the same `transmittance`/`done`/other pixel level values*. 

This is repeated work, though it has essentially 0 cost and unlocks the savings of dividing up channels.

@@ int slot = group_start + __ffs(mask) - 1;
`__ffs` is the find first set operation, and returns the 1-indexed location of the rightmost `1` bit in `mask`. `group_start + __ffs(mask) - 1` gives us the location of the first gaussian this warp is going to process in shared memory. 

@@ for (int group_start = 0; group_start < batch_count; group_start += WARP_SIZE) {
We just cooperatively loaded a batch of gaussians, and this loop traverses them. We cull these gaussians per-warp. If a gaussian influences none of the pixels the threads in a given warp cover, there is absolutely no reason to visit it. 

Parallelism: Each lane within a warp will verify if a particular gaussian influences any of the `rect` pixels belonging to the warp. 

@@ if (__syncthreads_count(done) == BLOCK_SIZE) break;
This line has two roles:
- Early exit the loop when every pixel in the tile has reached its maximum transmittance.
- Stall until all threads have finished the previous gaussian batch. Without the stall, a new batch may overwrite the previous before rendering is completed.

Parallelism: each thread is tracking the `done` status of a single pixel.

@@ float acc[WIDTH];
This will be the kernel's output: the `WIDTH` channels of this block's slice of the final feature vector of a given pixel.

Parallelism: Each thread is working on a subset of the final channels for a particular pixel.

@@ extern __shared__ float4 s_feat[];
Static shared memory arrays are capped to 48kb on an L4, but it allows up to 99 KB per block through a dynamic allocation. With `WIDTH = 64` and `BATCH = 128` a block stages 32 KB, so shared memory allows two blocks per SM, the same as registers.

@@ atomicAdd(&grad_features[(size_t)gaussian_id * D + channel0 + lane + WARP_SIZE * i], dL_df);
Because there are no vectorized atomics on the L4 GPU (which I wrote this on), we unfortunately must write each each gradient individually. Each lane, at this point, writes all the channels it was responsible for.


@@ for (int p = 0; p < WARP_SIZE; p++) w[p] = s_weight[warp * WARP_SIZE + p];
This is a sync to ensure that all 32 lanes in a warp hold all 32 weight values, which were computed per thread.

@@ float total = reduce_scatter<8, WARP_SIZE>(v, lane);
`total` holds all partial derivates summed across the whole warp. for 0<=k<=5, each lane 4k to 4k + 3 holds one of the derivatives. They're repeated since the reduce scatter is undersaturated given its template parameters.

Since each is repeated 4 times, we perform `k = lane >> 2` to divide it by 4. This gives tells us which lane holds which derivative.


@@ float dL_dG = c.w * dL_dalpha;
`c.w` is opacity.

@@ float dot = reduce_scatter<WARP_SIZE, WARP_SIZE>(partial, lane);
By the end, `dot` will hold the dot product of feature vector `f` with `dL/dpixel`. The reduce scatter ensures each lane holds the dot product owned by its pixel.

@@ float f[CHANNELS_PER_LANE];
Each thread touches `CHANNELS_PER_LANE` channels of a given gaussian. `f[i]` holds in registers the value of the feature vector's `i`-th channel, where `i` is a value this warp will touch (`lane + 32 * i`).

@@ bool contributes = pixel_inside && batch_start + slot < last_contributor;
Don't miss it: the parallelism has just changed again. Now, each thread is working with the SAME gaussian (`slot`), but computing for its own pixel.

Parallelism: Thread level. Each thread is working on its pixel, though within a warp all threads are doing so for the same gaussian.

@@ int bit = 31 - __clz(mask);
`__clz` is "count leading zeros". By performing `31 - __clz(mask)`, we get the index of the first non-zero bit in `mask`, which corresponds to the gaussian we are going to visit. 

`mask &= ~(1u << bit);` is going to flip "off" the gaussian in the mask so that it is marked as done. When all are marked done, we can move onto the next warp-sized chunk of gaussians in the batch.

@@ unsigned mask = __ballot_sync(FULL_MASK, candidate < batch_count && box_hits(s_mean[candidate], s_extent[candidate], rect));
This mask contains a bit for each of the 32 gaussians. The bit is 1 if the gaussian is used by at least one pixel, 0 if used by none of the pixels this warp handles. The goal being to reduce overall work: only visit gaussians where the mask is true.

@@ for (int group_start = (batch_count - 1) / WARP_SIZE * WARP_SIZE; group_start >= 0; group_start -= WARP_SIZE) {
This loop will walk the gaussians in the current batch back to front, but be warned that the parallelism paradigm changes inside the loop.

At the start, `(batch_count - 1) / 32 * 32` is `batch_count - 1` rounded down to the nearest multiple of 32. Why do that? Because we are about to kick off on warp-level parallelism. Each warp will compute a gradient for its 32 pixels with respect to all gaussians in the batch. A gaussian can touch multiple warps, so all warps will touch all gaussians. But, to speed things up, cull them at a warp level.

If gaussian `g` influences zero of a warp's 32 pixels, that warp can skip it entirely. This is explained further in the next note.

Parallelism: Warp level. We must cull gaussians that no lane will ultimately interact with.

@@ s_feat4[i] = features[(size_t)s_id[slot] * VEC4_PER_ROW + (channel0 / FLOAT4_SIZE) + (i - slot * VEC4_PER_SLICE)];
`(size_t)s_id` is purely defensive and casts the entire operation to `int64`. If `N` and `D` get large, this can become a huge chunk of memory. We may overflow `int32`, which can only hold up to 2^31 values (first bit is reserved for the sign).

@@ int slot = i / VEC4_PER_SLICE;
`VEC4_PER_SLICE` is a power of 2. During compilation, this will be a bit shift, not division.

@@ for (int i = tid; i < batch_count * VEC4_PER_SLICE; i += BLOCK_SIZE) {
Loads `WIDTH` feature values for the `BATCH` gaussians. This loop is a bit complicated to wrap your head around at first... Recall:
- `RENDER_BATCH` is the number of pixels in a tile.
- `BATCH` is the number of gaussians we are trying to cover right now.
- `VEC4_PER_SLICE = WIDTH / 4` is the amount of float4s we must load to have `WIDTH` channels.

This is really a nested loop in disguise. Written simpler *and sequentially*:
```
for (int slot = 0; slot < batch_count; slot++)            // each gaussian in the batch
      for (int m = 0; m < VEC4_PER_SLICE; m++)   // each float4 of its slice
          s_feat4[slot * VEC4_PER_SLICE + m] = features[s_id[slot] * VEC4_PER_ROW + channel0 / 4 + m];
```

It's a simple load when viewed that way. Though, to make it parallel, of course different threads must handle different gaussians and channels. 

`slot = i / VEC4_PER_SLICE` is the trick that makes the parallel version work. Since we need `VEC4_PER_SLICE` float4s per gaussian, `slot` is the gaussian that this thread is currently visiting. The indexing follows:
- `s_id[slot] * VEC4_PER_ROW` is the gaussian's starting index in memory.
- `channel0 / 4` adds the channel offset (since blocks cover different channels).
- `i - slot * VEC4_PER_SLICE` tells each thread to grab a different float4, with `slot * VEC4_PER_SLICE` making sure we start at 0 for each gaussian.

Parallelism: Thread level. One thread, per loop iteration, is handling a (gaussian, channel) pair in the cooperative load.

@@ const int channel0 = (blockIdx.x % SLICES) * WIDTH;
Will hold the index of the first channel that this block is responsible for. Blocks only hold width channels. Say `WIDTH = 128` but `D = 256`, two blocks will be needed per feature vectors. As such, `channel0 = 0/128` depending on which block we're in.

@@ for (int batch_end = num_todo; batch_end > 0; batch_end -= BATCH) {
A for each `BATCH` of gaussians in `num_todo`. `num_todo` is just the total number of gaussians this tile needs to walk.

@@ for (int i = tid; i < batch_count; i += BLOCK_SIZE) {
Cooperatively load a `BATCH` of gaussians (from deepest to shallowest) into shared memory.

Parallelism: Thread level. One thread takes care of one gaussian, so we can load up to `TILE_SIZE * TILE_SIZE = RENDER_BATCH` at a time.

@@ if (tid == 0) s_last_contributor = 0;
`s_last_contributor` is shared across the tile. Why use `tid == 0`? 

Instructions are issued at the warp level. While every thread would in theory write the same value (so it would be a one instruction broadcasted write), every warp would still issue an instruction. 

That, for a tile size of 256 threads, is 8 writes where 7 of those are redundant. In practice, this is so small it doesn't matter. But, it's right.

@@ #pragma unroll
For an array to live in registers, the compiler has to know at compile time which element every access touches. E.g., `arr[i] = 0.0f` with an unknown `i` forces the whole array into (far slower) local memory. Where local memory is just a per-thread reserved slice of global memory on an NVIDIA GPU.

`#pragma unroll` asks the compiler to literally unroll the loop into repeated instructions. For this to actually work, the # of loop iterations must also be a compile time constant (otherwise, how many times should the compiler repeat it?). 

Without this, the compiler silently will put these register values in local memory. The compiler needs exact instructions like "add register 7 to register 12". You can't say: "use register i", that's not a machine instruction. Unrolling replaces i with a real value.

@@ for (int p = 0; p < WARP_SIZE; p++) {
Cooperatively load `dL/dpixel` into registers s.t. each thread grabs multiple channels of each pixel. `grad[i][p]` is `dL/dpixel` for channel `lane + 32 * i` of the warp's pixel p. 

Remember that, when D >= 1, we render two images of shape `[H, W, 3]` (the rgb image) and `[H, W, D]` (the feature map). We are, in this kernel, only concerned with pixels from the second.

Parallelism: Warp level. Each warp is responsible for 32 pixels, so that's why we iterate over 32 values of `p`. Each thread in this section is assigned particular channels of `dL/dpixel` to track, which it does so across all the warp's pixels.
