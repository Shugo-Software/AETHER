# AETHER

**A local teacher → student distillation pipeline** for small language models,
built and measured on a single heterogeneous Windows rig (one Ada GPU as the
teacher, two Pascal GPUs as students). The trains of interest are the **Jadis**
models: compact instruction models fine-tuned from `LiquidAI/LFM2.5-2.6B`.

Everything here is reproducible from source: dataset generation, QLoRA training,
merging, GGUF quantization, independent verification, behavioural evaluation and
the dashboard that drives it all.

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

## What is in this repository

| Path | What it is |
|---|---|
| `scripts/` | The pipeline: dataset generation, QLoRA training, dashboard + controls, export (merge → GGUF → Hub), verification, evaluation, benchmarks |
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

# 5. Dashboard (tailnet or loopback bind)
powershell -ExecutionPolicy Bypass -File scripts\start-dashboard.ps1 -Port 8091
```

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
Model cards state the lineage (`base_model: LiquidAI/LFM2.5-2.6B`) and ship the
upstream license text.

## Licensing

- **Code** in this repository: Apache License 2.0 (`LICENSE`).
- **Model weights**: derivatives of `LiquidAI/LFM2.5-2.6B` → **LFM Open License
  v1.0**, which requires attribution and ships with the weights.
- **Datasets**: each keeps its own license; see the manifest and
  `docs/dataset-generation.md`.
