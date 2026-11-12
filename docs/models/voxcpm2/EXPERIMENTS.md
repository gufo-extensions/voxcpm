# VoxCPM2 experiments

Kernel experiments on Strix Halo with `tools/bench/gemm_bench.cpp`, which
times the products at the model's shapes on rotating copies of the weights
(November 11 and 12, 2026). A plain read of the same matrices reaches
220–245 GB/s. Real-time factors are `voxcpm_synthesize` on the 37-word
benchmark paragraph (release build).

The bench used to rotate 8 copies per shape. That keeps the Q8_0 tiles and
the smaller bf16 matrices of the local stacks partly in the 32 MB MALL, so
some of those rates exceed the read limit; the comparisons within a row
share the setting. The bench now rotates at least 128 MB per shape, so
every product reads DRAM as in the model: the wave counts were checked
again that way, and the last row gives the kept products' times.

| Experiment | Result | Kept |
| --- | --- | --- |
| hipBLASLt for every product | Its tiles (96×96 and similar) waste most work on 1–22 rows: 96 % of GPU time, real-time factor 0.84 | Only for more than 32 rows |
| One wave per weight row, activations staged in LDS | 1 row: 95–175 GB/s; 22 rows: 80–130 GB/s; real-time factor 0.50 | For 9–32 rows with bf16 rows (the DiT moved to the matrix cores below) |
| Two or four weight rows per wave | Slower at every row count (register pressure, fewer blocks) | No |
| Two or four 16-byte pieces per lane | Slightly faster up to 16 rows, slower above | Replaced by the sweep |
| Activation vector in LDS for one row | At most 10 % faster: the shared vector was not the bottleneck | Replaced by the sweep |
| Sweep: each block reads a contiguous block of 4 weight rows like a plain read, activation slices in registers | 1 row: 209–240 GB/s; 5 rows: 184–225 GB/s; 22 rows: about 50 GB/s (the slices spill); real-time factor 0.42 | Up to 8 rows |
| Sweep with the activations in LDS for 22 rows | 20–26 GB/s: one shuffle reduction per activation row and the full LDS (63 KB) leave the block starved | No |
| Whole weight row in registers before the activations are staged (22 rows) | 36–87 GB/s: 48 KB of LDS per block and the registers cut occupancy | No |
| Next chunk's weights requested while the current one is used (22 rows) | 54–118 GB/s, slower: more open requests to scattered rows hurt the DRAM more than the overlap helps | No |
| `v_dot2_f32_bf16` on the packed operands instead of unpacking to float32 | 22 rows: 4.5 % faster products; 1–5 rows unchanged (already at the read limit) | Yes |
| AudioVAE on its own stream, decoding a patch while the next one is generated; latents stay on the device; no host syncs within a step | Real-time factor 0.42 → 0.40 | Yes |
| Split K over up to four blocks per weight row with partial sums added in a fixed order (22 rows, few weight rows) | 7–14 % slower: idle compute units were not the limit | No |
| Attention: four waves per (row, head) with an online softmax in registers instead of 256 threads with the scores in LDS and seven barriers | Real-time factor 0.403 → 0.395 | Yes |
| Warm-up at load (a reference prompt, one patch, one decoding) | The first request no longer times hipBLASLt candidates: 0.403 → 0.396, first audio 0.29 → 0.12 s on the long text; loading 2.0 → 2.4 s | Yes |
| Matrix cores (`v_wmma_f32_16x16x16_bf16`) on Q8_0 weights repacked into tiles of 16 rows × 32 columns (one wave per 16 weight rows and a range of the reduction, int8 values converted exactly to bf16, float32 sums scaled per block, split reductions added in a fixed order) | DiT layer with 22 rows: 106 µs against 311 µs for the bf16 wave per row; 1 row: 200–310 GB/s | Yes, for every Q8_0 product of up to 32 rows |
| Q8_0 sweep (the bf16 sweep with 8 bytes per thread, unsigned values and the activation sum as offset) | 1 row: 200–270 GB/s, a tie with the matrix cores except the LM's gate/up (123 against 134 µs); 5 rows: 115–160 GB/s against 200–300 GB/s | No: one layout and kernel for Q8_0 |
| Matrix cores with two weight tiles per wave (each activation fragment used twice) | 15–50 % slower at every shape | No |
| Next block's Q8_0 weights requested before this block's products | 5–10 % slower at 22 rows: loads complete in order, so waiting for this block's activations waited for the prefetch too | Replaced by the next row |
| Next block's weights and activations requested before this block's products; int8 values sign-extended instead of offset by 128 (2.5 instead of 3.5 instructions per value) | DiT layer with 22 rows: 106 → 95 µs | Yes |
| Matrix cores on the row-major bf16 weights (22 rows) | o and down faster than the wave per row, gate/up 2–5 times slower: each wave reads its 16 rows 64 bytes at a time | Replaced by the next row |
| Matrix cores on bf16 weights repacked into the same tiles (the DiT's matrices only) | DiT layer with 22 rows: 310 → 199 µs | Yes |
| Waves per product (160–2560 tried) | 22 rows: about 160 (fewer, longer waves); 1–16 rows: about 640; confirmed with the 128 MB rotation | Yes |
| End to end with the matrix-core products | Real-time factor with the Hugging Face weights 0.396 → 0.293, with the Q8_0 GGUF 0.408 → 0.187 | Yes |
| Products of 33–128 rows on Q8_0 or bf16 tiles in chunks of 32 on the matrix cores instead of expanding the tiles for hipBLASLt (the same bytes up to 128 rows: one 8-bit read per chunk against an 8-bit read, a 16-bit write and a 16-bit read) | Prompts of up to 128 rows keep the Q8_0 values exact like the decoding steps; real-time factor with the Q8_0 GGUF 0.182–0.185 | Yes |
| The kept products with 128 MB rotated per shape | DiT layer with 22 rows: bf16 wave per row 354 µs, bf16 tiles 214 µs, Q8_0 tiles 111 µs (72 % of the Q8_0 read limit); LM layer with 1 row: bf16 sweep 424 µs, Q8_0 tiles 248 µs | |
