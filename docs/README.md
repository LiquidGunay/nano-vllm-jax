# Documentation

Current docs describe the cleaned serving path only.

- [Architecture](architecture.md): ownership boundaries and the engine loop.
- [Walkthrough](walkthrough.md): one request through packed prefill, decode, and a prefix-cache hit.
- [Benchmark](benchmark.md): the fixed B=1 contract, validity gates, and recorded result.
- [Speculative verification](speculative-verification.md): the optional
  drafter boundary and packed target-state commit.

Historical logs, optimization notes, and removed experimental-path writeups
belong outside this branch.
