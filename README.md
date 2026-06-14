# MI50 16 GB / gfx906 inference benchmarks

Local LLM inference benchmarks on an AMD Radeon **MI50 / Radeon Pro VII**
(gfx906 / Vega 20, 16 GB HBM2) — the cheap-but-awkward 16 GB compute card that
modern ROCm has officially dropped. Each subdirectory is one **model + engine**
benchmark set with raw data, charts, and the exact configs used.

## Hardware

| | |
|--|--|
| GPU | AMD Radeon Pro VII / MI50 (gfx906 / Vega 20), 16 GB HBM2, ~1 TB/s |
| Notes | no matrix cores (pre-CDNA); has int8/int4 dot-product (DLOPS); officially dropped in ROCm ≥ 7 |

## Benchmark sets

| set | models | engine / backends | highlights |
|--|--|--|--|
| [gemma-4-12B + bge-m3 on llama.cpp](gemma4%3A12b%2Bbge-m3_llama.cpp/) | gemma-4-12B Q4_K_M (chat) + bge-m3 (embed) | llama.cpp b9623 — **ROCm 7.2.3 vs ROCm 6.3.3 vs Vulkan/RADV** | ROCm ~2.2× on embeddings & ~20% on prefill; decode ~tie; ROCm6≈ROCm7; f16 KV is the free win |

*(more sets to come — each gets its own `model+engine` directory.)*

## Conventions

- **Same llama.cpp build across backends** within a set → differences are the
  backend, not the version.
- Throughput as **aggregate** (all concurrent requests) and **per-request**;
  prefill (PP) and decode (TG) reported separately.
- Charts are SVG (matplotlib); raw harness output kept alongside as `bench-*.log`.
- Configs (`docker-compose*.yml`, `models.ini`, Dockerfiles) are the exact ones
  benchmarked, copied in for reproducibility.
