# Serving the deep reviewer tier: DeepSeek-V4-Flash vs deepseek-v3:671b

**Measured 2026-09-04 → 2026-09-06 on Bridges-2 (H100-80 GB).** This answers
which candidate for the deep supervision tier can be *served* on this
allocation. It is not the reviewer comparison — which model audits better is
scored against the incident corpus and is reported separately.

## What was run

| job | model | backend | GPUs | ctx | outcome |
|---|---|---|---|---|---|
| 45198717 | deepseek-v3:671b | ollama 0.33.3 | 8 (whole node) | 32768 | **failed** — pulled, then `HTTP 500` on the first generation; job ended after 7 min 14 s |
| 45331031 | deepseek-v3:671b | ollama 0.33.3 | 4 (shared) | 65536 | **failed** — `TIMEOUT` at the 2 h wall, still loading tensors; never served a token |
| 45331073 | DeepSeek-V4-Flash | vLLM 0.28 | 4 (shared) | 32768 | **ok** |
| (45..., 2026-09-05) | DeepSeek-V4-Flash | vLLM 0.28 | 4 (shared) | 65536 | **ok** |

## DeepSeek-V4-Flash, measured

| | 32K context | 64K context |
|---|---|---|
| load to first token | 621 s | 1031 s |
| 16,113-token prefill | 0.75 s | 0.75 s |
| decode | 147.4 tok/s | 147.4 tok/s |
| KV cache | fp8 (required — the engine asserts it for this architecture) | fp8 |
| JSON mode | accepted | accepted |

Halving the context halved the load and changed nothing else that was measured.

## deepseek-v3:671b, why it did not serve

The Q4 checkpoint is ~376 GB against 4 × 80 GB of device memory, so ollama
placed part of the model on the host and loaded through `mmap` from the shared
filesystem. Its own log says so: *"tensor overrides to CPU are used with mmap
enabled"*. After 115 minutes it had still not answered a one-token request. On
eight GPUs — enough device memory on paper — the server returned `HTTP 500` on
the first generation instead.

**This is a serving result, not a judgement of the model.** A different backend
(vLLM with an appropriate quantisation), a different checkpoint, or a whole-node
allocation held long enough may well serve it. What is established is narrower
and is what the design needed to know: on the allocation this campaign actually
has, one of the two candidates is usable as a windowed deep tier and the other
is not.

## Consequence for the design

The plan already treats the deep tier as *windowed*, with nothing depending on
it. That is now measured rather than assumed: a tier whose model can take two
hours to load — or fail to — cannot sit on the path of a scheduler tick. The
fast tier and the deterministic checks carry the loop; the deep tier is
consulted inside a window when one is open.

## A defect this exposed in the measurement tool

Job 45331031 was killed at its wall before the result writer ran, so it left no
result file at all. From the artifacts alone it was indistinguishable from a job
that was never submitted — the same silence-instead-of-failure this campaign
exists to close, reproduced by the tool built to measure it.
`run_model_verify.sh` now writes a `status: loading` marker at job start and
overwrites it at the end, so a walltime kill leaves a record of what was
attempted (v3.32.4).
