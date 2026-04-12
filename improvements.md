This change reduces inference latency for the blur pipeline.
It should make batch and interactive runs noticeably faster without changing model outputs.
See benchmarks in the PR description for before/after numbers.
