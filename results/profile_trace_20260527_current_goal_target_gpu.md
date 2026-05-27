# Profile Trace Summary

## `/mountpoint/.exp/profiles/20260527-035343-2539001-gpu_matrix_long_prefill_512_2048_gpu_paged_default_r1_20260527_035342/plugins/profile/2026_05_27_03_54_07/INDCS0291.atrapa.deloitte.com.trace.json.gz`

- scope: `gpu`

### Pattern Totals

| pattern | total ms | count |
| --- | ---: | ---: |
| `MemcpyD2D` | 15.91 | 259 |
| `cutlass` | 64.54 | 42 |
| `gather` | 4.66 | 81 |
| `gemm_fusion` | 243.57 | 6424 |
| `input_reduce_fusion` | 34.66 | 797 |
| `loop_dynamic_slice` | 18.07 | 1152 |
| `loop_dynamic_update_slice` | 12.07 | 2286 |
| `transpose` | 45.08 | 312 |
| `wrapped_concatenate` | 17.39 | 288 |

### Top Events

| event | total ms | count |
| --- | ---: | ---: |
| `gemm_fusion_dot_general_744` | 57.46 | 48 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 36.91 | 30 |
| `gemm_fusion_dot_general_729` | 36.60 | 18 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 27.62 | 12 |
| `gemm_fusion_dot_2` | 26.04 | 24 |
| `loop_add_fusion_18` | 25.29 | 1152 |
| `loop_multiply_transpose_fusion_17` | 21.76 | 18 |
| `input_reduce_fusion` | 21.19 | 573 |
| `loop_dynamic_slice_multiply_subtract_fusion` | 18.07 | 1152 |
| `wrapped_concatenate` | 17.39 | 288 |
| `MemcpyD2D` | 15.91 | 259 |
| `gemm_fusion_dot_234` | 15.25 | 15 |
| `gemm_fusion_dot_210` | 14.13 | 1152 |
| `loop_divide_fusion_5` | 13.50 | 96 |
| `gemm_fusion_dot_general_731` | 12.06 | 18 |
| `gemm_fusion_dot_211` | 11.85 | 1152 |
| `gemm_fusion_dot_286` | 11.77 | 360 |
| `gemm_fusion_dot_285` | 11.26 | 270 |
| `gemm_fusion_dot_283` | 11.12 | 36 |
| `loop_multiply_fusion_47` | 10.86 | 24 |
| `input_concatenate_fusion` | 10.48 | 1242 |
| `input_scatter_fusion_59` | 9.33 | 18 |
| `loop_broadcast_fusion_2` | 7.95 | 1 |
| `fusion_2634` | 7.67 | 18 |
| `loop_transpose_fusion_50` | 7.52 | 17 |
| `input_reduce_fusion_19` | 6.48 | 6 |
| `input_reduce_fusion_25` | 6.36 | 6 |
| `gemm_fusion_dot_374` | 6.21 | 360 |
| `loop_add_fusion_74` | 5.62 | 21 |
| `triton_softmax_227` | 5.41 | 288 |

### Top HLO Ops

| HLO module | HLO op | event | total ms | count | kernel details |
| --- | --- | --- | ---: | ---: | --- |
| `jit_compiled` | `command_buffer_486` | `gemm_fusion_dot_234` | 15.25 | 15 | `regs:56 static_shared:0 dynamic_shared:36864 grid:1940,1,1 block:128,1,1 occ_pct:16.6667` |
| `jit_compiled` | `loop_broadcast_fusion.2` | `loop_broadcast_fusion_2` | 7.95 | 1 | `regs:24 static_shared:0 dynamic_shared:0 grid:98304,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_1` | `memcpy128` | 5.06 | 30 | `regs:16 static_shared:0 dynamic_shared:0 grid:1728,1,1 block:256,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_690` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.62 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_916` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.61 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_1141` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.61 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_464` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.60 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_236` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.59 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_1359` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.59 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_1` | `wrapped_concatenate` | 2.91 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_80` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_243` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_405` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_324` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_162` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_657` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_883` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_1334` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_1109` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_197` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |

## `/mountpoint/.exp/profiles/20260527-035410-2539077-gpu_matrix_long_prefill_512_2048_gpu_paged_default_r2_20260527_035342/plugins/profile/2026_05_27_03_54_35/INDCS0291.atrapa.deloitte.com.trace.json.gz`

- scope: `gpu`

### Pattern Totals

| pattern | total ms | count |
| --- | ---: | ---: |
| `MemcpyD2D` | 15.89 | 259 |
| `cutlass` | 64.48 | 42 |
| `gather` | 4.67 | 81 |
| `gemm_fusion` | 243.58 | 6424 |
| `input_reduce_fusion` | 34.66 | 797 |
| `loop_dynamic_slice` | 18.07 | 1152 |
| `loop_dynamic_update_slice` | 12.07 | 2286 |
| `transpose` | 45.06 | 312 |
| `wrapped_concatenate` | 17.39 | 288 |

### Top Events

| event | total ms | count |
| --- | ---: | ---: |
| `gemm_fusion_dot_general_744` | 57.45 | 48 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 36.90 | 30 |
| `gemm_fusion_dot_general_729` | 36.59 | 18 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 27.58 | 12 |
| `gemm_fusion_dot_2` | 26.05 | 24 |
| `loop_add_fusion_18` | 25.30 | 1152 |
| `loop_multiply_transpose_fusion_17` | 21.75 | 18 |
| `input_reduce_fusion` | 21.19 | 573 |
| `loop_dynamic_slice_multiply_subtract_fusion` | 18.07 | 1152 |
| `wrapped_concatenate` | 17.39 | 288 |
| `MemcpyD2D` | 15.89 | 259 |
| `gemm_fusion_dot_234` | 15.24 | 15 |
| `gemm_fusion_dot_210` | 14.13 | 1152 |
| `loop_divide_fusion_5` | 13.54 | 96 |
| `gemm_fusion_dot_general_731` | 12.06 | 18 |
| `gemm_fusion_dot_211` | 11.85 | 1152 |
| `gemm_fusion_dot_286` | 11.77 | 360 |
| `gemm_fusion_dot_285` | 11.27 | 270 |
| `gemm_fusion_dot_283` | 11.12 | 36 |
| `loop_multiply_fusion_47` | 10.85 | 24 |
| `input_concatenate_fusion` | 10.47 | 1242 |
| `input_scatter_fusion_59` | 9.33 | 18 |
| `loop_broadcast_fusion_2` | 7.96 | 1 |
| `fusion_2634` | 7.64 | 18 |
| `loop_transpose_fusion_50` | 7.51 | 17 |
| `input_reduce_fusion_19` | 6.48 | 6 |
| `input_reduce_fusion_25` | 6.37 | 6 |
| `gemm_fusion_dot_374` | 6.21 | 360 |
| `loop_add_fusion_74` | 5.62 | 21 |
| `triton_softmax_227` | 5.41 | 288 |

### Top HLO Ops

| HLO module | HLO op | event | total ms | count | kernel details |
| --- | --- | --- | ---: | ---: | --- |
| `jit_compiled` | `command_buffer_486` | `gemm_fusion_dot_234` | 15.24 | 15 | `regs:56 static_shared:0 dynamic_shared:36864 grid:1940,1,1 block:128,1,1 occ_pct:16.6667` |
| `jit_compiled` | `loop_broadcast_fusion.2` | `loop_broadcast_fusion_2` | 7.96 | 1 | `regs:24 static_shared:0 dynamic_shared:0 grid:98304,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_1` | `memcpy128` | 5.06 | 30 | `regs:16 static_shared:0 dynamic_shared:0 grid:1728,1,1 block:256,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_1141` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.61 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_690` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.60 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_236` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.60 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_464` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.60 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_916` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.59 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_1359` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4::Params)` | 4.59 | 2 | `regs:218 static_shared:0 dynamic_shared:81920 grid:264,4,8 block:128,1,1 occ_pct:8.33333` |
| `jit_compiled` | `command_buffer_1` | `wrapped_concatenate` | 2.91 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_80` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_162` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_324` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_405` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_243` | `wrapped_concatenate` | 2.89 | 45 | `regs:16 static_shared:0 dynamic_shared:0 grid:12288,1,1 block:128,1,1 occ_pct:100` |
| `jit_compiled` | `command_buffer_431` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.81 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_1109` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_197` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.80 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_883` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.79 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
| `jit_compiled` | `command_buffer_657` | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4::Params)` | 2.79 | 2 | `regs:224 static_shared:0 dynamic_shared:73728 grid:160,1,1 block:256,1,1 occ_pct:16.6667` |
