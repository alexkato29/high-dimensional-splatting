@@ asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
Tensor cores on the L4 operate on tf32 representations of floats. The difference: fp32 has 23 mantissa bits, tf32 has 10. Handing the tensor core raw fp32 bits truncates them, which meaningfully increased the error of our renderings compared to the oracle. This line rounds to nearest instead. While it's intuitive why truncation hurts (it always biases to round down), the math is still interesting.

I won't cover the exact math here, but this table has the gist:

|                  | Mean                 | Variance | Relative error of a blend of `N_eff` gaussians |
| ---------------- | ------------------------------------ | ----------------- | ---------------------------------------------- |
| Truncate         | `-ulp / 2`, about `-3.5e-4` relative | `ulp^2 / 12`      | about `7e-4`, stuck there no matter `N_eff`    |
| Round to nearest | `0`                                  | `ulp^2 / 12`      | about `2.9e-4 / sqrt(N_eff)`                   |

The variance is the same, but the mean differs. Errors with no bias partially cancel as more gaussians blend into a pixel, while a bias just continues to add up. 

@@ blend_group<MMAS_PER_WIDTH>(acc, w_group, slot_group, feat, lane);
The tensor core blending operation for a group of `MMA_K` gaussians. Perhaps the most interesting fact: this operation has nothing to do with being compute bound. Before this change, the kernel was actually severely underutilizing compute...

The tensor core swap, rather, was motivated by avoiding a different issue: warp divergence. Before, each lane would compute an updated per pixel-gaussian `transmittance`. If the `transmittance < MIN_TRANSMITTANCE` for a given lane, it sat around idle while the others ran on to do their blending.

With tensor cores, all lanes stay involved during all blends (since MMAs are warp level ops). This removed a large cause of warp divergence, and I think it's pretty interesting that this change wasn't actually motivated by the typical "we need faster math" argument.

@@ asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
Absolute unit of an instruction... First, why are we writing inline assembly? Well, the CUDA tensor core API `WMMA` isn't really flexible enough to do what we are trying to do here. It loads each matrix with one fixed stride, and its register layouts are hidden. Our `B` rows are whichever gaussians happened to be heavy, scattered around `s_feat`, and the whole column mapping trick depends on knowing exactly which lane holds what. PTX gives us both.

What does this instruction actually say? Reading the name left to right:

- `mma.sync.aligned`: a matrix multiply-accumulate that the whole warp executes together, so all 32 lanes must reach it at once (hence all the care to avoid divergence before it).
- `m16n8k8`: the shape. `A` is 16 x 8, `B` is 8 x 8, and the output is 16 x 8.
- `row.col`: `A` is read row-major, `B` column-major. For TF32 at this shape it's the only option PTX offers.
- `f32.tf32.tf32.f32`: the types of the output, `A`, `B`, and the input accumulator, in that order. Products in TF32, sums in full FP32.

Then the operands, each a brace list of this lane's registers:

- `{%0,%1,%2,%3}` first: the output `D`, this lane's 4 values (`C_VALS_PER_LANE`).
- `{%4,%5,%6,%7}`: this lane's 4 values of `A`.
- `{%8,%9}`: this lane's 2 values of `B`.
- `{%0,%1,%2,%3}` again: the accumulator `C`. Same registers as `D`, so the result adds onto `c` in place, `c = a . b + c`.

The lines below bind those `%` numbers to our variables. `"+f"` means a float register that is both read and written, which is exactly what an accumulator is. `"r"` means a plain 32-bit register, and `__float_as_uint` hands over the float's raw bits without converting anything. Why bits? PTX takes TF32 operands as 32-bit patterns and the tensor core only reads the top 19 of them (sign, exponent, 10 mantissa bits). The low 13 bits are simply ignored, which is why we round to nearest beforehand (see `to_tf32`). `volatile` just tells the compiler not to drop or merge the instruction.

@@ float b0[MMAS_PER_WIDTH], b1[MMAS_PER_WIDTH];
This matmul will perform `A @ B = C`. Each lane loads 2 values of `B` per matmul (since `B` holds 64 values and there are 32 lanes). `b0` and `b1` will ultimately store those 2 values, for each of the `MMAS_PER_WIDTH` matmuls.

@@ for (int j = 0; j < MMAS_PER_WIDTH; j++) mma_tf32(acc[m][j], a, b0[j], b1[j]);
Tensor cores require all hands on deck to cooperatively load values. `A` is a `16 x 8 = 128` value matrix, so each lane passes in 4 values. `B` is an `8 x 8 = 64` matrix, each lane will pass in two values.

@@ if (inside && blockIdx.x % SLICES == 0) {
This is at a per-pixel level, and we wouldn't want multiple blocks to potentially overwrite this! Recall that two blocks may work on the same pixel, though a different subset of its feature channels.

@@ int channel = channel0 + column_channel<MMAS_PER_WIDTH>(column, 0);
Map that column to a channel, but remember that the block starts at `channel0`.

@@ return n * MMAS_PER_WIDTH + j;
Unfortunately, this line has deceptively large implications... The idea of this function is to map column n in the j-th mma across to one channel. You would think it would be as simple as: `j * MMAS_PER_WIDTH + n`, which would map column `i` to channel `i`. But, notice that the actual function swaps `j` and `n`... The channels are actually ordered like:

|           | col 0 | col 1 | … | col 7 |
| --------- | ----- | ----- | - | ----- |
| mma j = 0 | ch 0  | ch 8  | … | ch 56 |
| mma j = 1 | ch 1  | ch 9  | … | ch 57 |
| …         |       |       |   |       |
| mma j = 7 | ch 7  | ch 15 | … | ch 63 |

Columns are strided in how they represent channels. We are allowed to do this: all the `mma` instruction cares about is that dimensions line up. But *why* do we do this, given it's much harder to think about? Rather than explain it here, I motivate why this layout makes sense on each of its caller lines.

@@ int column = C_COLS_PER_LANE * lane_in_group + i % C_COLS_PER_LANE;
Computes the column a lane is actively working on within one row of an `mma` output.

@@ return m * MMA_M + half * (MMA_M / 2) + lane / LANES_PER_GROUP;
 An `mma` tile is 16 rows (pixels) by 8 columns (channels). NVIDIA specifies this layout, and it looks like:
```
          col 0  1  | 2  3  | 4  5  | 6  7
row 0  (g=0)  L0 L0 | L1 L1 | L2 L2 | L3 L3
row 1  (g=1)  L4 L4 | ...
...
row 8  (g=0)  L0 L0 | L1 L1 | L2 L2 | L3 L3
row 9  (g=1)  L4 L4 | ...
```
Each lane gets 2 adjacent columns on 2 rows, 8 rows apart. But, for a given lane, which rows will it touch? Breaking down the formula:

- `m * MMA_M`: a warp's 32 pixels take two tiles stacked, so `m (0 or 1)` indicates if we are starting at pixel 0 or 16.
- `half * (MMA_M / 2)`: each lane appears on two rows, so `half (0 or 1)` indicates if we are in the top or bottom 8 rows of the `mma` tile.
- `lane / LANES_PER_GROUP`: lanes come in groups of 4 that share a row, so this is the row within those 8.

E.g., `fragment_pixel(m=1, half=1, lane=7)` would return 16 + 8 + 1 = 25.

@@ for (int i = count; i < MMA_K; i++) {
Sometimes, the batch may finish before the weight and slot buffers are full (`count < MMA_K`). In this case, we flush the old values and perform a blend with the present buffer. When we start a new batch, we overwrite `s_feat`, meaning we must blend now.

@@ if (!done) {
Nested ifs instead of continue, so every lane reaches the group step below.

@@ constexpr int MMAS_PER_WARP = WARP_SIZE / MMA_M;
Warps are 32 threads, but we can't actually do a matrix multiply on all 32 pixels at once due to the `MMA_M` size constraint. As such, we must perform `MMAS_PER_WARP` matrix multiplies to get values for all 32 pixels.

@@ float acc[MMAS_PER_WARP][MMAS_PER_WIDTH][C_VALS_PER_LANE];
Each lane holds part of its warp's output. But which part? And where on Earth does this shape come from?

A warp renders 32 pixels for its block's `WIDTH` channels, so it'll owe a `32 pixels x WIDTH channels` output. That's `WIDTH` channels per lane, which in the scalar kernel (see commit history), a lane simply owned. With tensor cores the split is... not so nice...

One `mma.m16n8k8` produces `16 pixels x 8 channels`, so the warp's output is tiled from several of them:
- `MMAS_PER_WARP = 32 / 16 = 2` tiles stacked down cover the pixels.
- `MMAS_PER_WIDTH = WIDTH / 8` tiles side by side cover the channels.
- Each mma tile's 128 output values are distributed to lanes. So, `C_VALS_PER_LANE = 128 / 32 = 4`.

So `acc[m][j][i]` is value `i` of this lane's share of the tile `m` down and `j` across. Which pixels and channels those are is fixed by the `C` fragment layout (see the output write).

@@ __shared__ int s_slots[BLOCK_SIZE / WARP_SIZE][MMA_K];
This keeps track of which of the `MMA_K` gaussians a given warp is working with, since warp culling allows warps to operate on gaussians independently.

@@ constexpr int A_VALS_PER_LANE = MMA_M * MMA_K / WARP_SIZE;
Tensor cores will take `A: m x k` and `B: k x n` as inputs to provide a `C: m x n` output. The matrix values are split among registers within a warp, so this tells us how many values each lane within the warp holds for `A` and `C`.

@@ constexpr int MMA_M = 16;
Shape of one mma.m16n8k8, rows (pixels) by columns (channels) by inner dimension (gaussians).

@@ __shared__ float s_w[BLOCK_SIZE / WARP_SIZE][MMA_K * W_STRIDE];
To fill up the tensor core matrices, we will accumulate the per-pixel-gaussian weights for numerous gaussians. In this case, `BLOCK_SIZE / WARP_SIZE` is the number of warps in the block (since each warp operates independently). `MMA_K * W_STRIDE` (see their definition sidenotes) gives space for the pixel-gaussian weights.

@@ constexpr int MMAS_PER_WIDTH = WIDTH / MMA_N;
Tensor cores require a fixed shape. On a NVIDIA L4 with the particular core we will be using, it's `M=16, N=8, K=8` where the matrix inputs `A` and `B` are $M \times K$ and $K \times N$, respectively.

The cores are ultimately going to compute `32 pixels x 8 weights @ 8 gaussians x WIDTH channels` (the warp's local output). But, because `N=8` does not necessarily equal width, we'll have to split it into a few separate MMAs. Luckily, channels are completely independent. We can just perform `WIDTH / N` multiplies.

@@ __device__ inline int2 warp_pixel(int warp, int lane) {
When performing warp culling, we use this function to ensure warp boxes are `8x4`. Why? It is the rectangle of area 32 with the smallest perimeter at 24. 

The worst case for a warp is having a gaussian touch the edge. The whole warp must pursue a gaussian that only one of its pixels will touch. Minimizing the perimeter minimizes the size of the edge, so this change is mostly a theoretical "correctness". In practice, the observed speedup is small like 3-9%. But it does see 12.5% fewer passes (on the truck scene) on average... Always feels good when intuition works.

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
Because there are no vectorized atomics on the L4 GPU (which I wrote this on), we unfortunately must write each gradient individually. Each lane, at this point, writes all the channels it was responsible for.


@@ for (int p = 0; p < WARP_SIZE; p++) w[p] = s_weight[warp * WARP_SIZE + p];
This is a sync to ensure that all 32 lanes in a warp hold all 32 weight values, which were computed per thread.

@@ float total = reduce_scatter<8, WARP_SIZE>(v, lane);
`total` holds all partial derivatives summed across the whole warp. for 0<=k<=5, each lane 4k to 4k + 3 holds one of the derivatives. They're repeated since the reduce scatter is undersaturated given its template parameters.

Since each is repeated 4 times, we perform `k = lane >> 2` to divide it by 4. This tells us which lane holds which derivative.


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
- `BLOCK_SIZE` is the number of pixels in a tile.
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

Parallelism: Thread level. One thread takes care of one gaussian, so we can load up to `TILE_SIZE * TILE_SIZE = BLOCK_SIZE` at a time.

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

@@ for (int i = 0; i < C_VALS_PER_LANE; i++) {
Each iteration will write an output. A lane holds 64 values: 16 channels across 4 pixels. Thanks to `column_channel`, they *are not* scattered. They form 8 runs of 8 contiguous channels that each belong to one pixel. Consider lane 5:

| pixel | runs |
| ----- | ---- |
| 1     | ch 16-23, ch 24-31 |
| 9     | ch 16-23, ch 24-31 |
| 17    | ch 16-23, ch 24-31 |
| 25    | ch 16-23, ch 24-31 |

The loop is two levels: the outer `m` and `i` pick a run (`2 x 4 = 8` of them), and the inner loop will write it. `m` is the tile down (pixels 0-15 or 16-31). `i` is one of this lane's 4 C values per tile, `i / 2` picks which of its 2 rows (pixels) and `i % 2` which of its 2 columns.

@@ int2 p = warp_pixel(warp, fragment_pixel(m, i / C_COLS_PER_LANE, lane));
`fragment_pixel` turns (tile down, which row) into the warp pixel 0-31, and `warp_pixel` turns that into x, y inside the tile. For lane 5 and `m = 0`:
- `i = 0, 1` land on pixel 1.
- `i = 2, 3` land on pixel 9.

@@ dst[q] = make_float4(acc[m][j][i], acc[m][j + 1][i], acc[m][j + 2][i], acc[m][j + 3][i]);
Four `acc` values in a row become one `float4` of output. But why are those four values adjacent channels? To see it, we first need to remember what `acc[m][j][i]` holds.

- `m`: which `mma` down. The warp's 32 pixels take two stacked tiles, so this is pixels 0-15 or 16-31.
- `j`: which `mma` across. Each covers 8 of the slice's channels.
- `i`: one `mma` writes 128 values, and each of the 32 lanes only gets 4 of them. `i` picks which of the 4.

Which 4, though? That's NVIDIA's call, not ours. From the layout diagram (see `fragment_pixel`), a lane shows up in 4 cells of every tile: 2 rows 8 apart, and 2 adjacent columns. Lane 5, for example:

```
            col 2   col 3
row 1       i = 0   i = 1
row 9       i = 2   i = 3
```

So `i / 2` says top or bottom row (which pixel), and `i % 2` says left or right column.

Now hold `m` and `i` fixed. We're on the same pixel and the same column `c` in every tile across, only `j` changes. So which channel is column `c` of the j-th `mma`? That one *is* our choice. The B load fed column `c` of `mma` `j` from channel `8c + j` (see `column_channel`), and since C's column `c` is just `A . B[:, c]`, it carries that same channel.

```
acc[m][j][i]  ->  channel  8c + j,   c = 2 * lane_in_group + i % 2
```

Each step in `j` is one channel over. Four in a row is a `float4`, and two of them cover all 8 channels of that column. With the "natural" mapping (`8j + c`), a lane would instead hold pairs of channels spaced 8 apart... 8 `float2` stores per pixel instead of 4 `float4`s.

@@ const float* row0 = feat + slots[lane_in_group] * WIDTH + channel;
Time to hand the tensor core its `B`, the features. Per `mma`, `B` is 8 gaussians (rows) by 8 channels (columns), and each lane only supplies 2 of those 64 values. Which 2? NVIDIA's `B` layout says both sit in column `group_id`, one on row `lane_in_group` and one on row `lane_in_group + 4`:

| value | `B` row (gaussian in the group) | `B` column |
| ----- | ------------------------------- | ---------- |
| `b0`  | `lane_in_group` (0-3)           | `group_id` |
| `b1`  | `lane_in_group + 4` (4-7)       | `group_id` |

So `row0` points at that first gaussian's feature row in `s_feat` (`slots` says where it was staged), and `row1` at the second. The `+ channel` jumps to this lane's column. Lane 5 (`group_id = 1`, `lane_in_group = 1`) reads group gaussians 1 and 5, both starting at channel 8.

@@ float4 v0 = reinterpret_cast<const float4*>(row0)[q];
But that's one column of one `mma`. We run `MMAS_PER_WIDTH` of them across, and the lane needs its column in every one. Thanks to `column_channel`, column `group_id` of the j-th `mma` is channel `8 * group_id + j`, so across all `j` the lane's values are 8 channels in a row. That's two `float4` loads per row instead of 8 scalar ones, and `b0[j]` ends up being exactly the value the j-th `mma` wants.

@@ const float* w0 = w + lane_in_group * W_STRIDE;
Now `A`, the weights. Per `mma`, `A` is 16 pixels (rows) by 8 gaussians (columns), and each lane supplies 4 values. NVIDIA's `A` layout puts them on columns `lane_in_group` and `lane_in_group + 4`, and on rows `group_id` and `group_id + 8`.

Notice anything? The lane's `A` columns are the same two gaussians as its `B` rows. That's no accident, it's the shared `k` dimension lining up. So `w0` and `w1` are the weight rows (one per gaussian, see `s_w`) for those two gaussians.

@@ float a[A_VALS_PER_LANE] = {w0[p0], w0[p1], w1[p0], w1[p1]};
And the rows of `A` are pixels, the same two rows the lane holds in `C` (see `fragment_pixel`). Picking those two pixels out of both gaussians' weight rows gives the 4 values, in the order PTX expects:

| value  | pixel (row) | gaussian (column)   |
| ------ | ----------- | ------------------- |
| `a[0]` | `p0`        | `lane_in_group`     |
| `a[1]` | `p1`        | `lane_in_group`     |
| `a[2]` | `p0`        | `lane_in_group + 4` |
| `a[3]` | `p1`        | `lane_in_group + 4` |

Unlike `B`, `A` doesn't change across `j`. Every `mma` across uses the same weights, only the channels differ, so we build `a` once per `m` and reuse it.
