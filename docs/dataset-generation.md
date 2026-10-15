# Phase 6 — dataset generation

Teacher: `qwen3.5-9b-teacher` (Qwen3.5-9B Q4_K_M on CUDA0, loopback `:8080`).
Student target: LFM2.5-2.6B QLoRA. Two sources, per requirement:

- **A. starcoderdata transforms** (`bigcode/starcoderdata`, ToU-gated,
  PASSED via our HF account): python shard
  `corpus/starcoderdata/python/train-00000-of-00059.parquet` (~758 MB,
  3.1 m rows) → rotating recipes `explain_code` / `qa_pairs` / `code_review`.
- **B. pure teacher generation** from 10 built-in seed prompts (recipe `seed_gen`).

## Harness

`scripts/generate_dataset.py` (stdlib-only client): deterministic seeded row
order, append/resume via `--tag`, `enable_thinking=false` (~30x cheaper),
JSONL + `manifest.json` (params, counts, sha256, teacher identity).
Endpoint must be up: `powershell -File scripts/start-teacher-server.ps1`.
Always launch pilot/background runs with `python -u` (unbuffered) so the
redirected log streams live. Throughput measured: **≈ 0.14 samples/s**
(1024-token cap) — plan ≈ 8.5 min per 75 samples.

## Pilot run (2026-10-14, tag `pilot1`) — 75/75, 0 failed, 541.5 s (~9 min)

`generate_dataset.py --n-a 50 --n-b 25` → `datasets/phase6_pilot1.jsonl`
(336.8 KB) + manifest. Recipe split: 17 explain_code / 17 qa_pairs /
16 code_review / 25 seed_gen. Completion chars: min 480 / median 1337 /
max 4729. Spot-checks (a-000000 explain on a TCP-server file, b-000000
threading-vs-asyncio) read concrete and correct. Rate ≈ 0.14 samples/s.

## ~~Prose recipes (EleutherAI/pile)~~ — removed 2026-10-15

The Pile track (`--source pile`) was removed to keep the tree small: datasets,
HF cache stubs, progress tracking, and the `load_source_pile` /
`PROSE_RECIPES` code are gone from `generate_dataset.py`. Jadis was trained on
that data (adapter `adapters/lfm25-jadis` survives); the source JSONLs were
deleted. Only the two StarCoder-backed sources (A/B below the title) remain.

## Quality gates (before any full-scale run)

1. **Spot-read ≥ 5 records per recipe** — concrete, grounded in the code,
   no hallucinated APIs.
2. **Truncation audit** — fraction of completions hitting `max_tokens`
   (pilot: several at exactly 1024 tokens → consider `--max-tokens 1536`
   for review-heavy batches).
3. **De-dupe check** — same `(repo, path)` shouldn't appear twice in one
   batch (seeded order makes this auditable).
4. Record any recipe wording changes in the manifest's `system_prompt`
   (edit + re-run from a new tag — never silently mutate).

## External dataset ingest (Phase 11, 2026-10-15)

`scripts/ingest_hf_dataset.py` imports third-party datasets into the same record
schema (`{id, source, recipe, system, prompt, completion, reasoning?, provenance,
meta}`) plus a manifest that pins the **source revision sha** and counts every
drop reason. Two destinations, deliberately different:

| Source | Output | Purpose |
|---|---|---|
| `FineEnvs/openai-math` (apache-2.0) | `datasets/eval/openai-math.jsonl` | **eval only** — 369 rows, recipe `proof_approach`, prompts trimmed to 4,000 chars (333/369 were longer) |
| `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` (apache-2.0) | *nothing written* | **rejected as training data** — see below |

### Why Fable-5.1 was rejected (measured, not assumed)

The dataset card reads like reasoning data (10k<n<100k rows, `format:json`,
apache-2.0). The rows are **agentic coding-agent transcripts**:

| Measurement | `train` (5k) | `full` (10k) |
|---|---|---|
| rows sampled | 300 | 400 |
| rows containing `tool_calls` | **300 / 300** | **400 / 400** |
| messages per row | 45–85, assistant/tool alternating | 45–85 |
| rows with a repo-scale authoring preamble | 294 / 300 | same shape |
| user prompt chars (median / max) | 4,908 / 9,002 | 4,947 / 9,003 |
| final assistant chars (median) | 132 (a status report) | **0** — usually empty |

There is no self-contained prompt→answer pair to select. Training on them would
teach a 2.6B student to *report on* repo-scale work it cannot possibly do — the
opposite of the direct, no-fabrication behaviour Jadis 1.0 was engineered for.
The ingester therefore writes nothing for this source and exits non-zero **with
the evidence printed**, so a future run cannot silently "succeed" with junk.

For training data at this scale use the teacher recipes over the StarCoder shard
(`explain_code` / `qa_pairs` / `code_review` / `seed_gen`): self-contained,
grounded in supplied code, and already proven to shape the student — see
`student-training.md`.

### Operator notes (Windows)

- `openai-math` ships a custom `harbor` loader inside `datasets` that reads its
  files with the *platform* codec (cp1252 here) → `UnicodeDecodeError` on valid
  UTF-8. The ingester re-executes itself with `PYTHONUTF8=1`; it uses a
  **list-args `subprocess`**, because a spaced install path (`Program Prototype`)
  gets re-split by `os.execve` — the same Windows quoting trap documented in
  `student-training.md`.
- Datasets under `datasets/eval/` are unreachable from the training control
  plane by construction: datasets launch by bare filename from `datasets/`, so
  `eval/openai-math.jsonl` cannot be selected (dashboard or API).
