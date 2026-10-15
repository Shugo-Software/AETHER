# Multi-GPU / Heterogeneous Benchmarks (Phase 4)

> **Captured:** 2026-10-14 (UTC). Tool: custom harness `scripts/bench_multi_gpu.py`
> (llama-cli, CUDA build). Model: **Bonsai-27B Q1_0** (3.54 GB — large enough to stress VRAM).

## Why this model

> **Note (2026-10-14):** Bonsai-27B was **retired as the distillation target** (see
> [`plan.md`](plan.md)); the current picks are Qwen3.5-9B (teacher) / LFM2.5-2.6B
> (student). These Phase 4 measurements remain valid as model-agnostic **capacity-probe**
> data — only the "target model" framing below is historical.

- Chosen at the time as the actual SamurAI distillation target.
- It **did not fit** on the 1060 3 GB alone (~2.46 GB free) → forced real layer splitting.
- Q1_0 quant loads and generates cleanly on the CUDA build (de-risked before this run).

## Method

| Setting | Value |
|---|---|
| Model | `Bonsai-27B-Q1_0.gguf` (3.54 GB, 65 layers) |
| `-ngl` | **99** (offload as many layers as fit) |
| `-sm` | **layer** (default pipelined split) |
| `-dev` | per-combination device list (IDs read live from `--list-devices`) |
| `-p` / `-n` | 128 prompt / 64 gen tokens (harness uses a short fixed prompt) |
| Timed reps | 3 per combo → **median** |
| Split evidence | 1 verbose (`-v`) run per combo → parsed `layer N assigned to device …` + buffer report |

## Results (median of 3 runs)

| Combo | Devices | pp (t/s) | tg (t/s) | Model buffer split (MiB) | ok |
|---|---|---:|---:|---|---:|
| single_4070Ti | CUDA0 | **97.0** | **60.8** | 3446 (all on CUDA0) | 3/3 |
| single_1060_6GB | CUDA1 | 35.6 | 11.5 | 3446 (all on CUDA1) | 3/3 |
| single_1060_3GB | CUDA2 | — | — | **does not fit** (see below) | 0/3 |
| 4070Ti+1060_6GB | CUDA0,CUDA1 | 65.9 | 24.2 | 2304 + 1143 | 3/3 |
| 4070Ti+1060_3GB | CUDA0,CUDA2 | 73.9 | 30.4 | 2765 + 682 | 3/3 |
| 1060_6GB+1060_3GB | CUDA1,CUDA2 | 32.2 | 10.9 | 2304 + 1143 | 3/3 |
| all_three | CUDA0,CUDA1,CUDA2 | 58.0 | 19.2 | 1997 + 920 + 528 | 3/3 |

Raw: `docs/benchmarks/multi_gpu_raw_20261014T073444Z.csv`
Summary: `docs/benchmarks/multi_gpu_summary_20261014T073444Z.csv`
Verbose logs (per combo): `docs/benchmarks/multi_gpu_verbose_20261014T073444Z_*.log`

### Layer distribution (from verbose logs, all 65 layers on GPU)

| Combo | Distribution (tensor-group assignments) | Ratio ≈ free-VRAM share |
|---|---|---|
| 4070Ti+1060_6GB | CUDA0=135, CUDA1=60 | 68 : 32 (11.0 GB : 5.2 GB free) |
| 4070Ti+1060_3GB | CUDA0=162, CUDA2=33 | 83 : 17 (11.0 GB : 2.5 GB free) |
| 1060_6GB+1060_3GB | CUDA1=135, CUDA2=60 | 68 : 32 (5.2 GB : 2.5 GB free) |
| all_three | CUDA0=117, CUDA1=54, CUDA2=24 | ≈ 11.0 : 5.2 : 2.5 GB free |

## Findings (measured)

1. **Splitting is capacity-weighted, not speed-weighted.** llama.cpp divides layers/KV/model
   buffers proportional to each card's *free VRAM*. The 1060s are slow, but they pull a
   proportional share of the layers regardless — so **the fastest card gates every multi-GPU run**.
2. **Multi-GPU here is a capacity tool, not a speed tool.** Best throughput is the single
   4070 Ti (60.8 t/s tg). Every multi-GPU combo is slower than it: adding a 1060 halves or
   thirds it (24–30 t/s), all-three = 19.2 t/s. The pipeline is bounded by the slowest stage.
   (Theoretical exception: `--tensor-split` manual ratios or `row` split mode could rebalance
   compute — tested as a follow-up, not in this run.)
3. **The 1060 3 GB alone refuses the model gracefully.** llama.cpp's fit check: *"projected to
   use 20377 MiB of device memory vs. 2310 MiB of free device memory"* → clean exit, no crash.
   But paired with any larger card it accepts its capacity-proportional share (682/528 MiB of
   model buffer) and generates fine.
4. **Counterintuitive: the *smaller* second card paired better.** `4070Ti+1060_3GB` reached
   30.4 t/s tg while `4070Ti+1060_6GB` reached only 24.2 t/s — adding the 3 GB card cost less
   throughput than adding the 6 GB card. The KV-cache splits differ (5174+1194 vs 6050+2750
   MiB), suggesting context placement/allocator noise matters. With only 3 reps this is a
   **signal to verify**, not a law — worth a dedicated follow-up with `-ts` control.
5. **15 of 18 timed runs completed cleanly** (all except the 3 expected OOM-failures on bare
   CUDA2; its verbose run failed the same way → 4 failed runs total). Variance across reps
   < ~5% on every working combo.
6. **Bonsai-27B Q1_0 validated on CUDA.** Historical note: at the time it was the target,
   now retired — 60.8 t/s generation on the 4070 Ti, 11.5 t/s on a 1060 6 GB (current
   teacher/student picks are in [`plan.md`](plan.md)).

## Honest limitations

- Single split mode (`-sm layer`) with default (free-VRAM-proportional) tensor split; `row`
  and `tensor` split modes and manual `-ts` ratios untested.
- Short prompt/generation (128/64); long-context KV growth not measured (KV already dominates:
  e.g. 6016 MiB on the bare 4070 Ti at defaults).
- 3 reps; no error bars beyond observed spread.
- The earlier interactive `-v` probe hung llama-cli in REPL mode — harness now always passes
  `-st --simple-io` (fixed in `bench_multi_gpu.py`).

## Reproduce

```powershell
python <AETHER_ROOT>\scripts\bench_multi_gpu.py --reps 3
# writes multi_gpu_{raw,summary}_<UTC tag>.csv + multi_gpu_verbose_<tag>_<combo>.log
```
