# Serving workload benchmark report

Generated at: `1780138094.3303754`
Model: `Qwen/Qwen3.5-0.8B`

## Workloads

- `decode_steady_b1`: Single active row with warmup; emphasizes steady decode after one normal prefill.

## Results

| workload | mode | TPU | valid | exact | next-step | HF | prefill tok/s | decode tok/s | e2e tok/s | decode speedup | measured enable | scheduler reason | scheduler speedup | scheduler EWMA base/spec ms | predicted diag | predicted disagrees | e2e speedup | acceptance | drafts p/a/r | step a/r/f | fallback | acc p50/p95 ms | rej p50/p95 ms | fb p50/p95 ms | host ms | device ms | post ms | first_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode_steady_b1 | baseline | false | false | true |  | true | 16.97 | 106.8 | 2.085 | 1.003 |  | disabled | n/a (smoke) | n/a (smoke)/n/a (smoke) | 1.003 | false | 0.9399 | 0 | 0/0/0 | 0/0/7 | 7 | 0/0 | 0/0 | 8.935/11.38 | 0 | 0 | 0 |  |
