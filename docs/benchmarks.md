# Single-GPU Benchmarks (Phase 3)

> **Multi-GPU / heterogeneous results (Phase 4):** see [`multi-gpu.md`](multi-gpu.md).
>
> **Captured:** 2026-10-14 (UTC), **all-NVIDIA config** (RX 580 removed → GTX 1060 3 GB added).
> Tool: `llama-bench` from both Phase-2 builds. Reproduce: `python scripts/bench_single_gpu.py --reps 3`.

## Method (fixed, equivalent across all GPUs)

| Setting | Value |
|---|---|
| Model / quant | Qwen2.5-0.5B-Instruct **Q4_K_M** (462.96 MiB, 630 M params) — fits every card + both backends |
| `-ngl` (layers offloaded) | **99** (full offload) |
| `-p` (prompt tokens) | **512** → prompt-processing throughput (pp512) |
| `-n` (gen tokens) | **128** → generation throughput (tg128) |
| `-sm` (split mode) | **none** (single-GPU: no layer/tensor split) |
| llama-bench `-r` (internal reps) | 5 per invocation |
| Harness invocations per device | **3**, reported as **median** |
| Peak VRAM | sampled via `nvidia-smi` @0.2s (unreliable for this fast tiny model — see note) |

Backends: **CUDA** (`builds\cuda`) and **Vulkan** (`builds\vulkan`), both from source `b11565`.
Device IDs are read live from `llama-bench --list-devices` (never assumed — see inventory §4.3).

## Results (median of 3 runs)

| Backend | Device | GPU | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|
| CUDA | CUDA0 | RTX 4070 Ti | **39765.56** | **519.52** |
| CUDA | CUDA1 | GTX 1060 6GB | 4787.56 | 119.85 |
| CUDA | CUDA2 | GTX 1060 3GB | 4227.51 | 97.28 |
| Vulkan | Vulkan0 | RTX 4070 Ti | 29339.55 | 506.90 |
| Vulkan | Vulkan1 | GTX 1060 6GB | 2104.31 | 146.42 |
| Vulkan | Vulkan2 | GTX 1060 3GB | 2049.98 | 133.20 |

Raw per-run data: `docs/benchmarks/single_gpu_raw_20261014T071321Z.csv`
Summary: `docs/benchmarks/single_gpu_summary_20261014T071321Z.csv`

## Observations (measured, not estimated)

- **RTX 4070 Ti leads by a wide margin** on both backends: tg128 ≈ 520 (CUDA) / 507 (Vulkan) t/s;
  pp512 ≈ 39.8k (CUDA) / 29.3k (Vulkan) t/s. CUDA edges Vulkan on prompt-processing; generation is
  close on this tiny model (the CUDA-vs-Vulkan gap tends to widen on larger models).
- **The two GTX 1060s track their VRAM/class.** 6 GB card: ~120 (CUDA) / ~146 (Vulkan) t/s gen.
  3 GB card: ~97 (CUDA) / ~133 (Vulkan) t/s gen. Both far slower than the 4070 Ti (~4–5×).
- **Vulkan beats CUDA on generation for the 1060s** (146 vs 120; 133 vs 97) but loses on
  prompt-processing — the same pattern seen pre-hardware-swap, so it's a real backend trait on
  Pascal, not noise.
- **All 18 runs completed cleanly** (exit_ok = true), no crashes, valid output. Variance across the
  3 runs was < ~3% on every device (tg128 spread tight).
- **Device parity achieved:** all three GPUs are now CUDA-capable, so any of them can serve as a
  training/inference target without a cross-vendor (AMD) special case. This matches the
  training-first plan; the RX 580 stays aside for a later Vulkan-only inference experiment.

## Startup latency (from run log, wall-clock per invocation incl. model load)

| GPU | CUDA build | Vulkan build |
|---|---:|---:|
| RTX 4070 Ti | ~2.4 s | ~2.4 s (first run 30.0 s — Vulkan pipeline warmup) |
| GTX 1060 6GB | ~6.9 s | ~7.1 s |
| GTX 1060 3GB | ~8.6 s | ~7.7 s |

> Vulkan's **first** run on the 4070 Ti was 30 s (shader/pipeline cache warmup), then ~2.4 s.
> Remember this one-time cost for server cold-starts.

## Limitations (honest, per principle #7)

- **Model is too small to meaningfully measure peak VRAM.** The 0.5B Q4_K_M loads/frees faster than
  the 0.2 s `nvidia-smi` poll on the fast cards, so the VRAM column is largely blank here. This run
  is a **backend-correctness + relative-ranking** probe, not a capacity test. VRAM *capacity* and
  layer/tensor splitting across cards need a larger model — that is Phase 4.
- Fixed single settings; no context-length sweep or batch-size sweep yet.

## Reproduce

```powershell
python <AETHER_ROOT>\scripts\bench_single_gpu.py --reps 3
# outputs timestamped raw + summary CSV under docs\benchmarks\
```

## Jadis on the 3 GB GTX 1060 — quant × context sweep (2026-10-15)

Answers the capacity question the 0.5 B probe above cannot: **does a real 2.7 B
Jadis build fit the GTX 1060 3 GB, at what context, and how fast?**

Harness `scripts/bench_3gb.py` drives **`llama-cli`** (because `llama-bench` has
no context-size flag), `-dev CUDA2`/`Vulkan2` (the 3 GB card), `-ngl 99`, greedy,
48 generated tokens, peak VRAM sampled via `nvidia-smi` @0.2 s. Raw CSVs:
`docs/benchmarks/three_gb_*.csv`. KV cost for this architecture is only
**16 KiB/token** (8 of 30 layers are full attention, GQA 8 KV heads × 64 head-dim
× 2 tensors × 2 B), so the weights dominate and context is cheap.

### CUDA (recommended)

| Quant | File | Peak VRAM 2 k | 4 k | 8 k | 16 k | tg t/s |
|---|---:|---:|---:|---:|---:|---:|
| `q3_k_s` | 1.18 GiB | 1387 MiB | 1421 | 1489 | 1625 | 30.7 |
| `q3_k_m` | 1.27 GiB | 1479 MiB | 1513 | 1581 | 1717 | 32.5 |
| **`iq4_xs`** (imatrix) | **1.41 GiB** | **1623 MiB** | **1657** | **1725** | **1861** | **44.4** |
| `q4_k_s` | 1.49 GiB | 1705 MiB | 1739 | 1807 | 1943 | 40.6 |
| `q4_k_m` | 1.56 GiB | 1773 MiB | 1807 | 1875 | 2011 | 40.6 |
| `q5_k_s` | 1.77 GiB | 1987 MiB | 2021 | 2089 | 2225 | 39.6 |
| `q8_0` | 2.68 GiB | **OOM** (all contexts) | — | — | — | — |
| `f16` | 5.03 GiB | not attempted — larger than the card | — | — | — | — |

Card total **3071 MiB** (~2965 MiB free with the desktop idle):

- **Everything up to `q5_k_s` fits, even at 16 k context** — `q4_k_m` @16 k peaks
  at 2011 MiB, leaving ~950 MiB headroom.
- **`iq4_xs` is the sweet spot**: 100 MiB smaller than `q4_k_m` *and* ~10 % faster
  (44.4 vs 40.6 t/s) — the importance matrix pays for itself.
- **`q8_0` is the measured hard boundary** — it OOMs during load at every context
  (2.68 GiB of weights + buffers exceed the card). Nothing else failed.

### Vulkan (faster generation, real hazard)

| Quant | Peak VRAM | pp t/s | tg t/s |
|---|---:|---:|---:|
| `q3_k_s` | 1359 MiB @4 k | 268 | 32.4 |
| `q3_k_m` | 1452 MiB @4 k | 271 | 37.4 |
| `iq4_xs` | 1595 MiB @4 k | 344 | 56.5 |
| `q4_k_s` | 1676 MiB @4 k | 304 | 59.3 |
| `q4_k_m` | 1812 MiB @8 k | 309 | 58.9–61.7 |
| `q5_k_s` | **device lost** @~1920 MiB | — | — |

Vulkan keeps the Pascal trait seen above: generation ~40–50 % faster than CUDA,
prompt processing ~35 % slower. **But `q5_k_s` under Vulkan fails with
`ggml_vulkan: device lost` at ~1920 MiB — well under the card's limit — and that
driver fault rebooted the machine** (05:55:15 UTC, `Kernel-Power 41`), taking a
concurrent QLoRA run with it. See the constraint in `plan.md`. These rows come
from `logs/bench-3gb-vulkan.log` rather than a summary CSV: the crash killed the
sweep before it wrote one. Treat Vulkan `q5_k_s`+ on this card as unsupported, and
never run Vulkan here while training.

### Recommendation

- **3 GB card → `iq4_xs` on CUDA** (`-c 4096…16384 -ngl 99`); `q4_k_m` for maximum
  compatibility, `q3_k_m`/`q3_k_s` when something else is holding VRAM.
- Keep `f16`/`q8_0` as CPU/reference artifacts.
- Reproduce:
  ```powershell
  python scripts\bench_3gb.py --dir cache\quants --ctx 2048,4096,8192,16384 --backends CUDA
  ```

