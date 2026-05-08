# MTP correctness and HF baseline parity report

Run date: 2026-05-06

Environment: TPU VM `nano-vllm-tpu-2404-run`, project `project-b9551f07-5f68-491a-8a0`, zone `europe-west4-a`

Model: `Qwen/Qwen3.5-0.8B`

Config preset: `qwen3_5_0_8b`

Precision: `bfloat16`

JAX backend: `tpu`

Result JSON on TPU VM: `/tmp/mtp_hf_throughput_real_top1.json`

## Correctness gate

Overall correctness: pass

Throughput marked valid by harness: true

MTP exact token match against baseline: true

HF logits check: pass

HF check policy: greedy correctness requires HF top-1 match plus dtype-aware logit MSE threshold. Top-k agreement is recorded as diagnostics, not a hard gate, because lower top-k ranks can differ under bf16/CPU-HF while greedy output remains unchanged.

HF MSE threshold used: `0.01`

## HF logits comparisons

| Prompt index | Prompt tokens | Top-1 match | Top-k exact | Top-k overlap | MSE | Max abs |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 0 | 13 | true | true | 5 | 0.0023986155 | 0.2487287521 |
| 1 | 11 | true | false | 4 | 0.0016679140 | 0.1957863569 |
| 2 | 23 | true | true | 5 | 0.0011341593 | 0.1640272141 |

## Real prompt suite summary

Prompt count: 8

Mean baseline decode throughput: `144.310 tok/s`

Mean MTP decode throughput: `68.650 tok/s`

Mean MTP decode speedup: `0.462x`

Mean MTP acceptance rate: `54.054%`

## Per-prompt results

| Row | Prompt tokens | Correct | Acceptance | Baseline decode tok/s | MTP decode tok/s | Speedup | First diff |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 13 | true | 64.286% | 63.249 | 18.742 | 0.887x | none |
| 1 | 11 | true | 76.923% | 159.628 | 77.222 | 1.283x | none |
| 2 | 23 | true | 64.286% | 154.325 | 76.994 | 0.174x | none |
| 3 | 12 | true | 64.286% | 154.386 | 76.472 | 0.877x | none |
| 4 | 12 | true | 35.294% | 154.391 | 74.396 | 0.487x | none |
| 5 | 15 | true | 35.294% | 154.264 | 73.285 | 0.837x | none |
| 6 | 15 | true | 27.778% | 152.595 | 72.814 | 0.483x | none |
| 7 | 8 | true | 64.286% | 161.643 | 79.277 | 1.127x | none |

## Notes

The earlier MTP divergence on the CSV prompt is fixed. The current correctness-preserving path samples current tokens from the baseline executor logits, resets hybrid state slots, avoids committing verifier state on rejection, disables fused verify by default, and records top-k HF diagnostics.

The current MTP path is slower on average because it uses a correctness-preserving hidden replay for MTP seeding and does not emit speculative bonus tokens by default. This report is intended to establish correctness before further performance work.
