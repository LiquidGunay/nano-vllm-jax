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
