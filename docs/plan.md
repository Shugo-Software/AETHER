# AETHER / SamurAI — Project Plan (canonical roadmap)

> **Persisted 2026-10-14.** This file is the source of truth for phase status and the
> teacher/student architecture. Update it as phases complete — future sessions should
> read this first. Detailed evidence lives in the sibling docs (see Artifact index).

## Architecture: teacher → student distillation on heterogeneous GPUs

```
┌─ RTX 4070 Ti (CUDA0, 12 GB) ────────┐      ┌─ GTX 1060 6GB (CUDA1) + GTX 1060 3GB (CUDA2) ─┐
│ TEACHER: Qwen/Qwen3.5-9B            │      │ STUDENT: LiquidAI/LFM2.5-2.6B                 │
│ Q4_K_M GGUF (~5.2 GiB), llama-server│ JSONL│ QLoRA training (fp16), transformers+peft+      │
│ · Phase 5 Cline endpoint            │─G:\─▶│ bitsandbytes, joint via accelerate ZeRO       │
│ · Phase 6 distillation data → G:\   │ disk │ · fallback: train on 6GB alone, 3GB = eval card│
└─────────────────────────────────────┘      │ · eval: llama.cpp --dev CUDA1,CUDA2 (layer)   │
                                             └────────────────────────────────────────────────┘
```

- Lanes are **separate processes on disjoint GPUs** → teacher generation and student
  training run concurrently (principle #4 applies: read device IDs from `--list-devices`).
- **Teacher pick:** `Qwen/Qwen3.5-9B` — GGUF: `lmstudio-community/Qwen3.5-9B-GGUF`
  (Q4_K_M / Q6_K / Q8_0).
- **Student pick:** `LiquidAI/LFM2.5-2.6B` — HF safetensors for training + official GGUF
  (Q4_K_M ~1.5 GB) for layer-split inference eval.
- Bonsai-27B was **retired as the target** (2026-10-14); it remains only as the Phase 4
  capacity-probe model.

## Phase table

| Phase | Goal | Status |
|---|---|---|
| 1 | Hardware/software inventory + operating principles | ✅ 2026-10-13 — `hardware-inventory.md` |
| 2 | Dual llama.cpp builds (Vulkan + CUDA) in separate dirs; existing CPU+RPC install untouched | ✅ `builds\vulkan`, `builds\cuda` — `build-guide.md` |
| 3 | Single-GPU benchmarks (backend × GPU matrix) | ✅ (×2 after RX 580 → GTX 1060 3GB swap) — `benchmarks.md` |
| 4 | Heterogeneous multi-GPU inference (layer split) | ✅ 2026-10-14 — `multi-gpu.md`, `scripts/bench_multi_gpu.py` |
| 5 | Cline local-provider profile → llama-server (teacher, CUDA0), clean rollback, cloud profiles untouched | ✅ 2026-10-14 — `cline-integration.md` (profile `qwen35-teacher`; 5 originals verified byte-identical; UI switch-out = rollback) |
| 6 | Teacher dataset generation: Qwen3.5-9B on 4070 Ti → JSONL on G: (starcoderdata transforms + teacher-gen) | ✅ Pilot 75/75 (tag `pilot1`, 2026-10-14) — `dataset-generation.md` |
| 7 | Student training: LFM2.5-2.6B QLoRA on the 1060 6GB (fp16, plain Trainer; TRL-fused path broken on Pascal) | ✅ pilot `pilot1` DONE + dashboard live :8091 with start/stop controls; **Jadis** (`adapters/lfm25-jadis`) trained 2026-10-15 on `phase7_combined_jadis` (250 rows, loss 2.4755 — corpus since purged); `adapters/lfm25-jadis-starcoder` = Jadis resumed (`--resume`) 6 epochs on the 75-row `phase6_pilot1` pilot (loss 1.2319). See the naming caveat below |
| 8 | Model export: merge LoRA → standalone fp16 model → GGUF (f16/q8_0/q4_k_m) → optional HF push, with independent verification | ✅ 2026-10-15 — `docs/export.md`; `exports/lfm25-jadis-starcoder/` (merged + f16 + q4_k_m), `scripts/verify_export.py` **2/2 passed** |
| 9 | DPO / preference optimization (`trl` on the pinned env) | ⬜ |
| 10 | Jadis 1.0 release: fix the `<think>` turn shape, retrain fresh, multi-quant export (+ license/card), 3 GB runbook, independent eval | ✅ 2026-10-15 — adapter `jadis-1.0` (75 rows, 10 epochs, loss 1.6664); `exports/jadis-1.0/` = merged + 7 GGUFs; `verify_export` 2/2; eval `logs/eval-jadis-1.0-nothink.md`; see `student-training.md`, `export.md`, `benchmarks.md` |

> **Names vs lineage (updated 2026-10-15):** `adapters/lfm25-jadis-starcoder` is
> **misnamed** — its run record says `dataset: phase6_pilot1.jsonl`, `rows: 75`,
> `epochs: 6.0`, `resumed_from: adapters/lfm25-jadis` (not StarCoder data). It is
> the **pre-fix** build (trained before the `<think>` turn-shape fix) and is kept
> only as a historical record — see `logs/eval-lfm25-jadis-starcoder.md`. The
> release artifact is **`jadis-1.0`** (fresh from base, corrected turn shape).
> Ancestor corpus `phase7_combined_jadis.jsonl` was purged, so `lfm25-jadis` is
> not reproducible from scratch (the adapter itself survives).
> **Release naming:** the model ships as **Jadis 1.0** (`exports/jadis-1.0/`,
> repo `ShugoSoftware/Jadis-1.0`); upstream lineage stays disclosed in the card
> (`base_model: LiquidAI/LFM2.5-2.6B`, LFM Open License v1.0) because the license
> requires attribution — brand name and lineage tag are different things.

## Hard-won constraints (2026-10-15) — pushing to GitHub hangs on this rig

`git push` stalls indefinitely (reads/fetches are fine). Trace
(`GIT_TRACE=1 GIT_CURL_VERBOSE=1`) shows git invoking
`git-credential-manager get`, which blocks in a non-interactive shell — the
**system** gitconfig (`C:\Program Files\Git\etc\gitconfig`) registers
`credential.helper = manager` *in addition to* the user's `store` helper, and
GCM is tried first. The stored credentials themselves are valid (verified
against the API: both entries have `push: true`).

Fix — reset the accumulated helper chain for the invocation:

```powershell
git -c credential.helper= -c credential.helper=store push origin main
```

Use that form for every push from this machine (and from any non-interactive
process). Symptom to recognise: `git ls-remote` works, `git push` never returns.

## Hard-won constraints (2026-10-15) — Vulkan on the 3 GB card can take the box down

`llama-cli` **Vulkan** on the GTX 1060 3 GB with `q5_k_s` died with
`ggml_vulkan: device lost` (exit `3221226505` = `STATUS_STACK_BUFFER_OVERRUN`) at
a peak of only ~1920 MiB (so this is not a plain out-of-VRAM error — the driver
faults). **The host then rebooted** (`Kernel-Power 41`, "rebooted without cleanly
shutting down") at 05:55:15 UTC, killing a concurrently running QLoRA training run
at step 13/50 (`logs/aborted-jadis-1.0-driver-crash.*` — kept as evidence) and the
dashboard process.

Rules that follow:

- **Never run a Vulkan workload concurrently with a training run** on this rig.
  A driver fault on one GPU takes every other GPU process with it.
- Treat Vulkan `q5_k_s`+ on the 3 GB card as **unsupported**: CUDA runs `q5_k_s`
  at 16 k context fine (2225 MiB). If Vulkan is wanted for the faster Pascal
  generation, stay at `q4_k_m`/`q4_k_s`/`iq4_xs` (all measured clean).
- CUDA sweeps (`bench_3gb.py`) were run alongside training without incident;
  only the Vulkan device-lost event was fatal.

## Verified constraints (facts — don't re-derive, but do re-verify if upgrading)

1. **PyTorch must be pinned `2.14.0+cu126`** (last prebuilt wheels with Pascal sm_61;
   2.15 drops Maxwell/Pascal/Volta from published CUDA 12.6 wheels — PyTorch dev notice
   2026-09-04). Any upgrade path past 2.14 = build-from-source on CUDA 12.x.
2. **Pascal training specifics:** fp16 only (no bf16), no flash-attention; **Unsloth
   excluded** (official minimum compute capability 7.0, Triton risk on Windows). Training
   stack = vanilla `transformers` + `peft` + `bitsandbytes` QLoRA + `accelerate` ZeRO.
3. **All Python/venv/model artifacts go to G:** — C: is critically small and **volatile**:
   observed 2026-10-14 spiking 2.8 GB → 39 MB → 2.5 GB within minutes (Windows Update /
   VSS transient allocation — `wuauserv` running, 4 GB in SoftwareDistribution). The prior
   winget-CUDA incident filled C: to 0 bytes. If C: runs low, safe reclaim (ask first,
   they belong to other projects): **uv cache ≈ 9.4 GB**, user Temp ≈ 1.9 GB,
   npm-cache ≈ 1.1 GB, pip cache ≈ 77 MB (all re-downloadable). Never install to C:.
4. **llama.cpp source pinned at `f9a95ae`** (2026-10-12): arch support verified in-tree —
   `qwen35`/`qwen35moe` ✅ and `lfm2`/`lfm2moe` ✅ (LFM2.5 uses the `lfm2` arch tag).
   Builds: Vulkan `builds\vulkan` (VS generator), CUDA `builds\cuda` (Ninja — the VS
   generator fails with "No CUDA toolset found"; runtime DLLs must sit next to the exe).
5. **Phase 4 findings (model-agnostic):** llama.cpp splits layers proportional to *free
   VRAM* (capacity-weighted, not speed-weighted) → multi-GPU = capacity tool; slowest
   card gates throughput. Bare 4070 Ti beats every multi-combo for speed.

## Open gates (must pass before Phase 6 dataset generation — principle #6)

- [x] **Qwen3.5-9B license/terms** confirmed from the official `Qwen/Qwen3.5-9B` model card.
      → **Apache License 2.0** (`LICENSE` raw file, Copyright 2026 Alibaba Cloud).
      Derivatives + commercial use permitted without conditions. ✅ 2026-10-14.
- [x] **LFM2.5 license** (HF tag `license:other`) confirmed to permit training derivatives
      + redistribution of the fine-tuned student.
      → **LFM Open License v1.0** (Liquid AI): derivative works explicitly granted;
      commercial use licensed up to a **$10M annual-revenue Threshold** (§5) — not
      exceeded here; derivatives may carry their own modification license provided the
      base terms are met (§4). ✅ 2026-10-14.
- [x] **GGUF smoke tests:** teacher Q4_K_M loads+generates on CUDA0; student Q4_K_M
      loads+generates layer-split on CUDA1+CUDA2 (proves `lfm2` arch end-to-end).
      → 2026-10-14, CUDA build, exit 0 both: **Qwen3.5-9B Q4_K_M on CUDA0**
      (pp 82.5 / tg 50.1 t/s); **LFM2.5-2.6B Q4_K_M on CUDA1+CUDA2** (`-sm layer`)
      (pp 218.7 / tg 43.9 t/s). Models stored at `models\` (G:, 5.37 GB + 1.60 GB).

## Operating principles

Canonical list lives in `hardware-inventory.md`. Key ones for this phase:
- **#2** existing CPU+RPC install stays untouched (separate dirs only).
- **#4** device index ≠ physical GPU across tools — read `--list-devices` live.
- **#6** verify claims from primary sources; never assume (this is what the Open gates are).
- **#7** measure and report honestly, including failures and limitations.
- **#8** model sources (lmstudio-community GGUF repos) are read-only origins.
- **#9** local path stays cloud-free.

## Artifact index

| Path | Content |
|---|---|
| `docs/hardware-inventory.md` | Phase 1 inventory, principles, ecosystem context |
| `docs/build-guide.md` | Phase 2 Vulkan + CUDA build procedures |
| `docs/benchmarks.md` | Phase 3 single-GPU results (0.5B probe model) |
| `docs/multi-gpu.md` | Phase 4 split results (Bonsai-27B as capacity probe) |
| `docs/cline-integration.md` | Phase 5 Cline storage map, local profile, rollback |
| `scripts/start-teacher-server.ps1` / `stop-teacher-server.ps1` | Phase 5 teacher endpoint lifecycle (CUDA0, :8080) |
| `docs/benchmarks/*.csv` | Raw + summary benchmark evidence (timestamped) |
| `scripts/bench_single_gpu.py` / `bench_multi_gpu.py` | Phase 3/4 harnesses |
| `scripts/build-*.ps1/.bat`, `assemble-cuda-toolkit.ps1` | Build tooling |
| `builds/`, `toolchains/` | Build outputs + portable CUDA 12.8.1 toolkit |
| `models/` | Teacher/student GGUFs (Qwen3.5-9B Q4_K_M, LFM2.5-2.6B Q4_K_M) — G:, never C: |
| `docs/dataset-generation.md` | Phase 6 harness, pilot results, quality gates |
| `scripts/generate_dataset.py` | Phase 6 harness (stdlib-only client, seeded, manifests) |
| `datasets/phase6_pilot1.jsonl` + `.manifest.json` | Phase 6 pilot: 75/75 samples, 0 failures |
| `corpus/starcoderdata/python/` | Shard `train-00000-of-00059.parquet` (~758 MB, 3.1 m rows) + sha256 |
| `venvs/ml/` | Phase 6/7 Python env (torch 2.14.0+cu126, transformers 5.19, peft, bnb, accelerate, trl) — G:, never C: |
| `docs/dashboard.md` | Dashboard usage (http://127.0.0.1:8091/, tailnet via Tailscale bind) |
| `docs/student-training.md` | Phase 7 harness, Windows quoting trap, TRL-on-Pascal gotchas, smoke/pilot |
| `scripts/train_student.py` / `run-train.cmd` | Phase 7 QLoRA trainer + quoting-safe launcher (TEMP/HF/PIP caches forced to G:) |
| `scripts/sample_student.py` | Generate student samples → `logs/train-<tag>.samples.json` (dashboard "Live generations") |
| `scripts/train_control.py` | Dashboard training control: `start_training`/`stop_training`/`active_run`/`list_adapters` (safe-launch, PID-guarded stop, resume-from-adapter) |
| `scripts/export_model.py` | Export pipeline: merge LoRA → standalone fp16 model (`exports/<name>/`), comma-separated `--gguf f16,q4_k_m,q3_k_m…` (f16/q8_0/q6_k/q5_k*/q4_k*/q3_k*/iq4_xs via `--imatrix`), `--name`/`--title` release identity, brand card + shipped `LICENSE` + manifest **before** the push, optional Hugging Face push (private default) — `docs/export.md` |
| `scripts/export_control.py` | Dashboard export control: `start_export`/`stop_export`/`active_export` (same safe-launch + PID-guarded pattern as training; quant list read from `export_model.QUANT_TYPES`) |
| `scripts/verify_export.py` | Independent export smoke test: merged model via plain `transformers` (asserts all-`F16` header) + GGUF via `llama-cli` (`-st -ngl 0`) → `logs/verify-<name>.json` |
| `scripts/eval_jadis.py` | Behaviour eval: base vs base+adapter, two prompt conventions (`template` prefill vs `raw`), auto metrics (`</think>` close/lead, preamble, per-recipe format, memory overlap) → `logs/eval-<tag>.md`/`.json`; `--samples` feeds the dashboard |
| `scripts/bench_3gb.py` | 3 GB deployment sweep: llama-cli per (quant × ctx × backend) with peak-VRAM sampling → `docs/benchmarks/three_gb_<stamp>.csv` |
| `exports/<name>/` | Export output: merged fp16 safetensors + tokenizer + `chat_template.jinja`, `<name>-<quant>.gguf` set, generated `README.md` model card, `LICENSE`, `export-manifest.json` |
| `adapters/jadis-1.0` | **Release adapter** — fresh QLoRA on `phase6_pilot1.jsonl` (75 rows, 10 epochs) with the corrected `<think>` turn shape |
| `adapters/lfm25-jadis`, `lfm25-jadis-starcoder` | Earlier adapters (pre-fix turn shape); the ancestor corpus was purged — see the naming note above |

