# AETHER

<img src="assets/aether-logo.png" alt="AETHER" width="180" align="right">

**A local teacher → student distillation pipeline** for small language models,
built and measured on a single heterogeneous Windows rig (one Ada GPU as the
teacher, two Pascal GPUs as students). The trains of interest are the **Jadis**
models: compact instruction models fine-tuned from `LiquidAI/LFM2.5-2.6B`.

Everything here is reproducible from source: dataset generation, QLoRA training,
merging, GGUF quantization, independent verification, behavioural evaluation and
the dashboard that drives it all.

> **Release 1.0.0** — 2026-10-15. The first packaged release: the pipeline runs
> end to end from the dashboard, on external datasets as well as generated ones.
> See [What's in 1.0.0](#whats-in-100) below.

```
┌──────────────────────────────┐        ┌──────────────────────────────────────┐
│ TEACHER  Qwen3.5-9B (Q4_K_M) │        │ STUDENT  LFM2.5-2.6B  (QLoRA fp16)   │
│ RTX 4070 Ti · CUDA0 · :8080  │──data──▶ GTX 1060 6GB · trained on-card       │
│ llama-server, OpenAI-compat. │        │ merged → fp16 / GGUF (f16 → q3_k_s)  │
└──────────────────────────────┘        └──────────────────────────────────────┘
                ▲                                        │
                │        teacher-as-judge review         │ quantized copies served
                └────────────────────────────────────────┘ on the 3 GB GTX 1060
                     (recursive loop: reviewers gate each iteration)
```

## What's in 1.0.0

The whole path from raw data to a served, quantized student, driven from one page:

| Stage | What it does | Where |
|---|---|---|
| **Dataset generation** | Teacher-written A/B pairs, manifests, dedupe, anchor holdout | `scripts/generate_dataset.py` · `docs/dataset-generation.md` |
| **External ingestion** | Link any Hugging Face dataset → pull to G: → convert → verify → fit. Whole dataset by default (`--limit 0`), every shard of the chosen split read, and agentic traces mined for their plain answers instead of refused | `scripts/hf_datasets.py` · `scripts/ingest_hf_dataset.py` |
| **Queue** | Ordered and self-healing: a dataset whose runner died is *stalled*, not locked, and resumes at `train`. `--queue-plan` prints what would run — and how big a pull is — before anything downloads | `docs/dashboard.md` |
| **Student training** | QLoRA fp16 on the 6 GB Pascal card, loss masked to the answer | `scripts/train_student.py` · `docs/student-training.md` |
| **Recursive loop** | train → export → serve → teacher review → gate; failures become the next corpus | `scripts/recursive_train.py` · `docs/recursive-loop.md` |
| **Export** | merge LoRA → standalone fp16 → GGUF (f16 / q8_0 / q4_k_m / iq4_xs …) → optional Hub push, then independent verification | `scripts/export_model.py` · `scripts/verify_export.py` · `docs/export.md` |
| **Dashboard** | 12 panels, 5 s live polling, controls for training / export / the queue, plan + preflight, loopback or tailnet bind | `scripts/dashboard.py` · `docs/dashboard.md` |

Measured while packaging this release: converting `openai/gsm8k` went from
**2,000 rows to 7,473** once the caps were removed (`--limit N` still caps on
purpose, and `--limit 2000` was verified to still yield exactly 2,000); an agentic
code-SFT set (`11-47/MiniMax-M2.1-Code-SFT`) now yields **654 records from 1,306
rows** where it previously yielded none, each tagged `provenance.agentic` with the
caveat recorded; and a queue item whose runner was killed mid-iteration is reported
**stalled** and resumes at `train` instead of locking its row forever.

## What is in this repository

| Path | What it is |
|---|---|
| `scripts/` | The pipeline: dataset generation **and external dataset ingestion** (pull to G: → convert → verify → fit), the queue and its preflight plan, QLoRA training, the recursive loop, export (merge → GGUF → Hub), verification, evaluation, benchmarks, and the dashboard that drives it all |
| `docs/` | The engineering record: hardware inventory, build guide, benchmarks, dataset generation, training, export, dashboard, and `plan.md` — **the canonical roadmap** |
| `docs/benchmarks/` | Raw benchmark evidence (CSV) and the summaries quoted in the docs |
| `datasets/` | Dataset *manifests* only. The JSONL corpora are regenerable and intentionally not committed |

**Not in the repo (by design):** model weights, LoRA adapters, GGUF files, Python
virtualenvs, CUDA toolchains, llama.cpp builds, caches and logs. Those live on the
machine — and the *released* weights live on Hugging Face.

## The rig it was built on

| Device | Role |
|---|---|
| RTX 4070 Ti (12 GB, Ada, cc 8.9) | Teacher serving (llama-server, :8080) |
| GTX 1060 6 GB (Pascal, cc 6.1) | Student training (QLoRA, fp16) |
| GTX 1060 3 GB (Pascal, cc 6.1) | Quantized student serving / eval |

Two Pascal-specific constraints are baked into the code: fp16 only (no bf16), no
flash-attention, and a pinned PyTorch build — see `docs/hardware-inventory.md`
and `docs/student-training.md`. llama.cpp device ids (CUDA0/1/2) do **not** match
`nvidia-smi` indices; every script takes the device explicitly.

## Quickstart

```powershell
# 1. Python env (pinned torch for Pascal) — see docs/plan.md "Verified constraints"
python -m venv venvs\ml
venvs\ml\Scripts\pip install -r requirements.txt

# 2. llama.cpp builds (CUDA + Vulkan) — see docs/build-guide.md
#    Point these two env vars at your own trees:
set AETHER_LLAMA_CPP_SRC=<path to a llama.cpp checkout>   # HF -> GGUF conversion
set AETHER_BENCH_MODEL=<path to a small GGUF>             # benchmark probe model

# 3. Teacher endpoint (CUDA0, loopback only)
powershell -ExecutionPolicy Bypass -File scripts\start-teacher-server.ps1

# 4. Generate a dataset, then train / export / verify
python scripts\generate_dataset.py --n-a 50 --n-b 25
scripts\run-train.cmd --tag my-pilot --epochs 3
python scripts\export_model.py --adapter my-pilot --name my-pilot --gguf f16,q4_k_m
python scripts\verify_export.py --model exports\my-pilot

# 4b. Or bring in an external dataset — plan it before you pay for it
python scripts\hf_datasets.py --queue-plan --remote        # what would run, how big
python scripts\hf_datasets.py --advance org/name --split full   # pull to G: -> convert -> verify -> fit
python scripts\hf_datasets.py --queue-run                  # then train each queued dataset

# 5. Dashboard (tailnet or loopback bind)
powershell -ExecutionPolicy Bypass -File scripts\start-dashboard.ps1 -Port 8091
```

## Verification

Three checks run offline, and they are what "verified" means here — all three pass
on this tree:

```powershell
python scripts\hf_datasets.py --selftest   # 24 checks: URL parsing, the registry,
                                           # the drop gate, the queue lock (stale
                                           # vs live), the stage ladder, the plan
python scripts\dashboard.py --selftest     # 7 checks: the inline script parses,
                                           # every id exists, every fetch hits a
                                           # route `Handler` actually serves
python scripts\check_no_personal_data.py   # scans what git would publish
```

`dashboard.py --selftest` exists because of a one-character bug that shipped in the
first commit: a `\n` inside a JS string literal became a real newline when served,
the whole `<script>` block died with `Invalid or unexpected token`, and **every
button on the page was dead** while every panel still rendered. A server-rendered
page cannot be trusted by eye, so it is asserted. The same reasoning runs through
the pipeline: state files name the process that wrote them, the queue refuses to
touch what is in flight, and a dataset whose runner died is reported *stalled*
rather than assumed alive.

## Privacy guard

`scripts/check_no_personal_data.py` scans exactly what git would publish and fails
on overlay/private IPs, OS user paths, GPU UUIDs and any term you pass with
`--forbid` (or list in `check_files.txt`). Run it before every push — it is the
regression test for "nothing personal ships":

```powershell
python scripts\check_no_personal_data.py --forbid myhandle
```

## Models

Released weights are published on Hugging Face under **ShugoSoftware** (e.g.
`ShugoSoftware/Jadis-1.0`), with the GGUF set in a sibling `-GGUF` repository.
Model cards state the lineage (`base_model: LiquidAI/LFM2.5-2.6B`) and ship
**both** license texts — ours and the upstream one.

## Licensing

**Proprietary — All Rights Reserved, non-commercial.** See `LICENSE`.

- **Code** in this repository: © 2026 Shugo-Software. No commercial use, no
  redistribution, no derivative distribution without written permission.
- **Model weights** (Jadis line: adapters, merged checkpoints, GGUF
  quantisations): the same terms. Because they are **derivatives of
  `LiquidAI/LFM2.5-2.6B`**, the **LFM Open License v1.0** also applies and is
  shipped with every export as `LICENSE-UPSTREAM-LFM.md`; its attribution and
  revenue-threshold conditions must be met in addition to ours. Our license adds
  a non-commercial restriction to our contribution — it does not relicense
  anything upstream.
- **Datasets** keep their own licenses. `THIRD-PARTY.md` lists every upstream
  component, dataset and toolchain with its license and how it is used; each
  corpus manifest records its source repo, revision and parameters.

Commercial rights are available on request.
