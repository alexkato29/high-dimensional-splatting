# Rasterizer Design Notes

Per chunk of the program, I explain an optimization, how I found it, how I addressed it, and what it did. The number is the order in which they occurred.

## Forward Pass

### Project Gaussians

##### (1) Understand the shape of your matrices
I noticed that this part of the pass was far and away the longest. We were calling 3 batched matmuls of shapes like [1.1M, 3, 3]. That is a ridiculously bad shape for cuBLAS... The `ampere_sgemm_128x128_nt` kernels were taking `~3.4ms` each. Sounds fast, but when we think about the work:
- On the L4, max of 65536 matrices per BMM
- 3.4ms / 65536 matrices = ~52ns per 3x3 multiply
- 18 reads, and 9 writes with 300 GB/s of memory bandwidth = ~0.36 ns of computation

Abysmal ratio. I swapped matmuls for hardcoded expressions with Claude and saw a $\times3$ speedup!

##### (2) Don't repeat work
I saw `aten::inverse` was taking `~18.3ms` per forward pass. Where is an inverse computed?? Turns out, once:
```
@property
def center(self) -> torch.Tensor:
    return self.world_to_camera.inverse()[:3, 3]
```
This property recomputes the camera center every time called. But, a camera's center *never* changes. This is a complete waste of time. Plus, from optimization (1), you can imagine that the kernel dispatched by torch to take the inverse of a `4x4` matrix is going to be all overhead.

Though, the inverse wasn't even a slow target. I realized I was looking at the CPU line (`CUDA API`) which was actually just stuck in a barrier synchronization (`cudaStreamSynchronize`). The CPU is waiting on a value, but the GPU is so busy that I can revisit these syncs.

### Spherical Harmonics

##### (2) PyTorch isn't all magic
For large matrix multiplies, it's hard to beat torch. But, there are some things eager-mode PyTorch genuinely wasn't designed for. Spherical harmonics is an excellent example.

To perform spherical harmonics, we must perform tons of tiny computations on lots of data. All in, ~49 multiplies and subtracts across `N` gaussians. In eager-mode, torch's own docs say:
>PyTorch eager-mode initiates a separate kernel for each operation, which involves loading data from memory, executing the operation (often not the most time-consuming step), and writing the results back to memory.

What one forward pass writes to produce the colors, at `N = 1,182,597` gaussians (fp32, so one `[N]` tensor is 4.73 MB):

| File + loc | Code (shorthand) | Shape in → out | Memory written |
|---|---|---|---|
| `utils/sh.py:34-43` (`sh_basis`) | `xx = x * x`, `C3[3] * z * (2*zz - 3*xx - 3*yy)`, ... (49 pointwise ops) | `[N]` → `[N]` each | ~232 MB (49 × 4.73 MB) |
| `utils/sh.py:36-43` (`sh_basis`) | `basis = torch.stack([16 basis terms], dim=-1)` | 16 × `[N]` → `[N, 16]` | 76 MB |
| `utils/gaussian.py:108` (`colors`) | `sh = torch.cat([sh_dc, sh_rest], dim=1)` | `[N, 1, 3]` + `[N, 15, 3]` → `[N, 16, 3]` | 227 MB |
| `utils/sh.py:61` (`eval_sh`) | `rgb = torch.einsum("nk,nkc->nc", basis, sh)` | `[N, 16]`, `[N, 16, 3]` → `[N, 3]` | 14 MB |

For millions of gaussians, that's hundreds of megabytes of repeated reads and writes with little computation. `torch.compile()` was introduced to save the day, and generally that works. But, in our case, we later call `torch.einsum(...)`. Under the hood, that will leverage cuBLAS, and `torch.compile()` will not fuse it. The inputs and outputs will need to exist in memory, there's no way around it.

Looking back at the table, that's where the largest remaining writes come from. `torch.compile()` can fuse the ~49 pointwise ops into a single kernel, but it can't see past the einsum, so both of its inputs still have to be materialized:
- Stacked basis, `[N, 16]` (76 MB)
- Concatenated coefficients, `[N, 16, 3]` (227 MB)

Neither holds new information, they're just copies rearranged so cuBLAS can take one tensor each. And, autograd saves both for the backward pass. All we actually need is `rgb`, `[N, 3]`, which is only 14 MB. As long as einsum is involved, we can't do better. The fix options then become:
1. Rewrite the algorithm pointwise so `torch.compile` can fuse all ops
2. Hand write the CUDA kernel

I first tried (1) to get more PyTorch practice, and the forward total time ~halved after doing this.
I ended up pursuing (2), because `torch.compile` can't easily handle varying parameter shapes. Every densification/culling pass, we change `N`, so compilation wasn't the right call.

### Forward render

##### Parallelize where it matters (3)

The original authors cleverly batch gaussians into tiles to make rendering faster. To achieve this, a kernel `duplicate_gaussians` must makes $N_g$ copies of each gaussian in memory, where $N_g$ is the number of tiles a given gaussian $g$ appears in.

On the first write, I per thread:
- Chose a gaussian id
- Read the number of times it must be duplicated
- Wrote the gaussian that many times to the output

Abysmal. So what was going wrong? Two problems:
1. Instructions are issued at the warp level. If one thread needs to create tons of duplicates, the rest of its warp will have to sit and wait.
2. By nature of the kernel, outputs had to be in gaussian id order. Meaning, each thread is performing a strided write in memory.

Well, we certainly don't want a stalled threads. And, if we are performing far more writes than reads, wouldn't we rather coalesce the writes?

To address both, I rewrote the algorithm around coalescing output writes (with the added advantage of solving the warp stall problem).

## Backward Pass

##### (3) Atomic operations don't cleanly stack
Every thread in the backward pass computes gradients of a different pixel w.r.t a particular gaussian. When it's time to accumulate gradients via `atomicAdd(&grad_colors[g * 3 + 0], grad_color_x)`, the atomic property fortunately prevents race conditions in the memory writes! But, rather unfortunately, it does so by queueing memory write requests.

When all 256 threads in the block at roughly the same time call `atomicAdd` on the same gaussian `g`, the L2 slice owning that memory address will have to queue each write to guarantee the atomic property. That leads to very long memory ops. Notably:
```
Runtime: 31.82ms
Compute (SM) Throughput [%]	25.06
Memory Throughput [%]	63.75
L1/TEX Cache Throughput [%]	50.95
L2 Cache Throughput [%]	63.75
DRAM Throughput [%]	2.13
L2 Hit Rate [%]	99.28
Mem Pipes Busy [%]	25.06
```
L2 throughput at `63.75%` is our limiting factor, so clearly our atomic operations are taking too long.

To fix this, we can make use of `__shfl_down_sync` to share values within a warp. The idea is that, before actually issuing a write, sum the gradients within the warp *first*, then write the value *once*. Sharing values between threads in the warp is called a shuffle, and in our case it may decrease the # of queued writes by 32x! In reality, it almost surely won't.
```
// As we walk gaussians back to front, did this gaussian impact the pixel this thread is responsible for? If not, don't bother writing anything.
bool contributes = inside && batch_start + j + 1 <= last_contributor;
if (!contributes) continue;
...
// If that gaussian was in range, did it contribute at least more than some negligble `MIN_ALPHA` amount to final transmittance?
float alpha = fminf(MAX_ALPHA, raw_alpha);
if (alpha < MIN_ALPHA) continue;
...
// If you made it this far, do the 9 atomic add operations
```
There are two gates that might stop a given thread from actually doing the `atomicAdd`. We therefore don't expect to see 1/32 the memory traffic, and it will be extremely hard to predict what sort of gain we will see. Let's just run it and see:
```
Runtime: 10.75ms
Compute (SM) Throughput [%]	89.49
Memory Throughput [%]	89.49
L1/TEX Cache Throughput [%]	92.06
L2 Cache Throughput [%]	17.13
DRAM Throughput [%]	6.36
Mem Pipes Busy [%]	89.49
```
A `~3x` of a speedup, and the limiting factor at some point swapped from L2 cache throughput to memory pipe queueing. Let's see if we can derive why the speedup was x3.

| counter | what it counts | all lanes | warp reduction |
| --- | --- | ---: | ---: |
| `gpu__time_duration.sum` (ns) | wall time of the kernel | 31,816,736 | 10,748,608 |
| `gpc__cycles_elapsed.max` | GPU clock cycles the kernel spanned | 26,214,542 | 8,944,626 |
| `smsp__inst_executed_op_global_red.sum` | atomic instructions issued, one per warp regardless of how many lanes are awake | 29,409,057 | 29,409,057 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum` | 32 byte sectors those atomics sent to L2, one per awake lane | 376,550,778 | 29,409,057 |
| awake lanes per atomic (sectors / instructions) | derived, how many of the 32 lanes actually wrote | 12.80 | 1.00 |
| `lts__d_atomic_input_cycles_active` [%] | cycles the L2 atomic unit was busy, as a share of peak | 59.67 | 13.65 |

The atomic instruction count is identical in both runs. Not surprising, the SIMT model issues instructions per warp, not thread, and the number of warps hasn't changed. But, `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum` fell dramatically. By `~12.80x`! This counter measures the number of *sectors sent* per instruction. We haven't changed the instruction count, but we are sending `12.80x` less data, on average. And in that lies our answer, only 12.80 of the 32 lanes were reaching the `atomicAdd`, on average.

As a bonus, L2 work fell 12.80x but the kernel also finished 2.93x sooner. The reported active kernel cycles could only fall by the ratio of the two:

```
work: 29,409,057 / 376,550,778 = 0.0781
rate: 13.649 / 59.674          = 0.2287

cycles predicted:  0.0781 / 0.2287        = 0.3415
cycles measured:   8,944,626 / 26,214,542 = 0.3412     (0.1% off)
```


##### (4) Relieving SMEM pressure
We decreased atomic operations, but it wasn't free...
```
Compute (SM) Throughput [%]	89.49
Memory Throughput [%]	89.49
Mem Pipes Busy [%]	89.49
```
Now, for some reason, memory pipes are limiting our overall throughput. Okay... well what is a memory pipe and why is our's busy? From [Nvidia](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-workload-analysis):
> Memory can become a limiting factor for the overall kernel performance when fully utilizing the involved hardware units (Mem Busy), exhausting the available communication bandwidth between those units (Max Bandwidth), or by reaching the maximum throughput of issuing memory instructions (Mem Pipes Busy).

We have "reached the maximum throughput of issuing memory instructions". Intuitively, a memory instruction is probably just global/shared memory read/write instructions. But... something is suspicious. In optimization (3), we did not change the # of issued memory instructions at all and only saw a `2.93x` speedup. Let's do the math:

| counter | what it counts | all lanes | warp reduction |
| --- | --- | ---: | ---: |
| `smsp__sass_inst_executed_op_shared_ld.sum` | shared memory loads | 52,922,071 | 52,922,071 |
| `smsp__sass_inst_executed_op_shared_st.sum` | shared memory stores | 303,836 | 303,836 |
| `smsp__sass_inst_executed_op_global_ld.sum` | global loads | 457,906 | 457,906 |
| `smsp__inst_executed_op_global_red.sum` | global atomics | 29,409,057 | 29,409,057 |
| **total memory instructions** |  | **83,092,870** | **83,092,870** |
| `gpc__cycles_elapsed.max` | cycles the kernel spanned | 26,214,542 | 8,944,626 |
| `sm__inst_executed_pipe_lsu` [%] | memory instruction issue rate | 11.03 | 89.49 |
| `sm__memory_throughput` [%] | Mem Pipes Busy | 25.06 | 89.49 |

Instructions are identical to the digit. Issue rate is instructions over cycles, so with the numerator fixed the only thing left to move it is the cycle count:
```
instructions: x1.00
cycles:       x0.341
predicted rate: 11.03 / 0.341 = 32.3%
measured rate:                  89.5%
```

The denominator simply cannot explain that, so the numerator must have grown... Well, the only thing the diff added was `__shfl_down_sync`. To Google! From [Nvidia](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure), on what the LSU pipeline issues:
> **The LSU pipeline issues** load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory. It also issues special register reads (S2R), **shuffles**, and CTA-level arrive/wait barrier instructions to the L1TEX unit.

There it is. A shuffle *does ride* the LSU pipeline! The instruction burns a memory issue slot like any load would.

Shuffles apparently have no counter of their own, but L1TEX bills them as shared memory traffic that is neither a load nor a store, so they fall out as the residual:

| counter | what it counts | all lanes | warp reduction |
| --- | --- | ---: | ---: |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | all shared memory data path traffic | 88,995,987 | 233,491,143 |
| `..._op_ld.sum` | the load part | 88,029,436 | 85,420,773 |
| `..._op_st.sum` | the store part | 847,545 | 910,425 |
| **residual** | **shuffles, plus a little S2R and barrier traffic** | **119,006** | **147,159,945** |

```
Shuffles per gradient:  5
Gradients per gaussian: 29,409,057
Total `__shfl_down_sync` calls: 5 * 29,409,057 = 147,045,285
```
Basically dead on to the residual SMEM operations. If we add those to the memory instructions, the `89.49%` falls right out:

| | all lanes | warp reduction |
| --- | ---: | ---: |
| memory instructions (from above) | 83,092,870 | 83,092,870 |
| shuffles | 119,006 | 147,159,945 |
| **total LSU instructions** | **83,211,876** | **230,252,815** |
| `gpc__cycles_elapsed.max` | 26,214,542 | 8,944,626 |
| **instructions per cycle** | **3.174** | **25.742** |
| `sm__inst_executed_pipe_lsu` [%] | 11.03 | 89.49 |

```
New LSU inst. / Prev LSU inst. = 2.767
New cycles / Prev cycles =       0.341
Prev % if cyckes issuing an LSU instruction: 11.03%

11.03% * 2.767 / 0.341 = 89.501%
```

The `2.93x` from fewer cycles was only part of the picture. The `2.77x` numerator increase is the 147 million shuffles!

Okay, issue derived. How do we fix it? Well, clearly we are issuing too many shuffles, let's try doing fewer shuffles and letting *some* queueing happen in L2. How many shuffles to cut? We currently do 5 per gradient, let's try some other values:

| shuffles per gradient | shuffles per gaussian | atomic lanes per gaussian | atomic sectors | atomic work vs 5 | LSU instructions | LSU vs 5 | L2 atomic rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 45 | 1 | 29.4M | 1x | 230.3M | 1.000 | 13.6% |
| 4 | 36 | 2 | 58.8M | 2x | 200.8M | 0.872 | 31.3% |
| 3 | 27 | 4 | 117.6M | 4x | 171.4M | 0.745 | 73.3% |
| 2 | 18 | 8 | 235.3M | 8x | 142.0M | 0.617 | 177% |
| 1 | 9 | 16 | 470.5M | 16x | 112.6M | 0.489 | 447% |
| 0 | 0 | 32 | 941.1M | 32x | 83.2M | 0.361 | 1209% |

Of course, as we saw last time, there aren't *actually* 32 atomics per lane. In reality, all these numbers are ceilings. But, we don't want to increase the L2 atomic rate over 100%, even theoretically. Then we're just swapping the limiting factor and making the kernel slower! 3 shuffles per gradient is the best we can do, and it's what we'll use to theoretically get the best speedup.

After swapping so four threads perform the `atomicAdd` and only 3 shuffles are executed, we see see the benefits:
```
Runtime: 8.52ms
Compute (SM) Throughput [%]	83.50
Memory Throughput [%]	83.50
L1/TEX Cache Throughput [%]	87.55
L2 Cache Throughput [%]	47.11
DRAM Throughput [%]	7.72
Mem Pipes Busy [%]	83.50
```
Much better, though it doesn't match the theoretical. That is because, if you work it out, the atomic lanes per gaussian at 3 shuffles really works out to be 2.55, not the theoretical 4.
