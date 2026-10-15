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

## Verifying and fitting a corpus before training (2026-10-15)

Asking "is this corpus good?" is the wrong question; ask **"what does the trainer
actually feed the model?"** Two tools answer that, and both import the trainer's
own renderer (`train_student.format_record` / `format_head`) so they cannot drift
from training:

- `scripts/verify_dataset.py --jsonl … --max-len 1536 [--overlap …]` — read-only.
  Reports composition, token lengths, and a per-row truncation class:
  **complete** / **partial** (answer partly cut) / **answer_cut** (answer gone),
  whether `<|im_end|>` survives, the share of loss-relevant tokens that are
  answers, duplicate prompts, and prompt overlap with another file (leakage).
  Exits non-zero if any row loses its answer.
- `scripts/fit_corpus.py --in … --out … --max-len 1536 --holdout …` — rewrites a
  corpus so every row is learnable: drops anchor-overlapping prompts, trims
  answers back to the last natural boundary (paragraph → line → sentence → word)
  that still fits, drops rows whose prompt alone exceeds the budget, and caps
  duplicates per prompt. Writes `<out>.filter.json` provenance (input sha256,
  policy, per-reason counts).

Measured on `phase6_pro1.jsonl` (500 rows, the first corpus built for the loop):

| | before fit | after fit (`phase6_pro1-fit.jsonl`) |
|---|---|---|
| rows | 500 | 330 |
| complete / partial / **answer_cut** | 352 / 81 / **67** | 330 / 0 / **0** |
| terminator `<|im_end|>` survives | 0 (verifier bug — see below) | 330 (100%) |
| answer tokens kept | 72.5% | 100% |
| unique prompts | 410 (one prompt repeated **x14**) | 330 (no duplicates) |
| prompts also in the anchor | 11 (in 101 rows) | **0** |
| answers as share of loss tokens | 35.8% | 35.8% (the fix is masking; see `student-training.md`) |

Dropped: `prompt_too_long=67`, `anchor_holdout=101`, `answer_too_short_after_trim=2`.

### The anchor set is also the seed source — a structural conflict

`phase6_pilot1.jsonl` (60 prompts) is the anchor for every comparison, but the
`seed_gen` recipe draws its seeds from that same list. So 101 rows of `pro1`
carried anchor prompts: **you cannot both train on them and measure against
them.** Options, in order of preference:

1. hold them out (what `fit_corpus.py --holdout` does; the anchor stays a real
   holdout, at the cost of the teacher-gen rows — which is what happened here:
   the fitted corpus is 330 teacher-*answered* StarCoder rows),
2. generate a **fresh** seed list for future corpora (Phase 6 change) so seeds
   and anchor stop being the same 60 prompts,
3. accept a contaminated anchor and stop calling it an anchor.

Verifier bug found the same day: the terminator check compared only the *final*
token, but a rendered row ends `…answer<|im_end|>\n`, so it reported 0/500 for
corpora that were 100% correct. Fixed (`_has_subseq` searches a small trailing
window). Lesson: a measurement tool needs the same suspicion as the thing it
measures — an implausible 0% is a bug report, not a finding.

