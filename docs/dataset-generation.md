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
  Reports composition, token lengths, and a per-row truncation class —
  **complete** / **partial** (answer partly cut) / **answer_cut** (answer gone) —
  whether `<|im_end|>` survives, the share of loss-relevant tokens that are answers,
  duplicate prompts, and prompt overlap with another file (leakage). Exits non-zero if
  any row loses its answer. Measured with the same tokenizer call the trainer
  truncates with (`fit_corpus.py` uses it too — one accounting, no drift), and it
  reports separately how many rows the fitter had already rewritten (`rewrites`, with
  the cut point and boundary of the first few) — those are rows that would otherwise
  have been dropped, so they are reported beside the classes, not inside them.
- **Prompt trimming — the row is kept, not dropped** (2026-10-16). `fit_corpus.py
  --prompt-trim tail` (the default) keeps the *end* of an over-long prompt, cuts it at
  a natural boundary (paragraph → line → sentence → clause → word) and inserts a
  marker where the cut happened: `[... 3,412 earlier characters omitted ...]`. The
  answer stays intact whenever that is possible; when it is not, the answer is
  trimmed to a boundary as well — never marked, because a marker inside a completion
  is something the model would learn to emit — and the cut is recorded instead.
  `head` keeps the start of the prompt (question-first corpora), `none` restores the
  old behaviour of dropping the row.

  Every rewritten row carries its own attribution in `meta.fit`:

  ```json
  {"policy": "prompt-tail@word", "max_len": 1536,
   "prompt": {"trimmed": true, "chars_before": 9812, "kept_chars": 4210,
              "dropped_chars": 5602, "cut_at_char": 5602, "boundary": "word",
              "marker": "[... 5602 earlier characters omitted ...]"},
   "answer": {"trimmed": true, "cut_at_char": 1180, "boundary": "line",
              "answer_tokens": 96, "marker": null}}
  ```

  and the corpus's `.filter.json` lists `prompt_trimmed_rows`, `answer_trimmed_rows`
  and `rewritten_rows` (row id + cut point + boundary, one entry per rewrite), so
  "this output changed" can always be traced to *which* truncation caused it.
  Measured on this rig: `phase6_pro1` 330 → 397 rows (67 prompt-trimmed),
  `starcoderdata-code` 2384 → 2386 (3 prompt-trimmed), and corpora with no over-long
  prompts come out byte-identical — the policy cannot touch a row that already fits.

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

## The Hugging Face dataset pipeline (link → pull to G: → learn → release)

`scripts/hf_datasets.py` runs one dataset at a time, whole, on disk, then removes
it. State per dataset lives in `logs/hf-datasets.json`:

```
added → pulled → converted → verified → fitted → training → learned_verified → dropped
                                        ↘ not_trainable / error
```

- **pull** — `snapshot_download` into `cache/datasets/<org>--<name>/` (G:, gitignored).
  Card file patterns are globs (`main/train-*`) and many datasets ship Parquet, so
  patterns are expanded and both `.jsonl` and `.parquet` are read. When a card
  offers several splits of the same data (`5k/` and `10k/`), only the largest is
  mapped, and when it lists several configs under one split name (`main/` +
  `socratic/` for gsm8k) the **first** wins — pulling both would silently double
  the corpus with a different recipe.
- **convert** — maps the *local* copy (never re-streams the hub) into the training
  schema. `--map chat` turns each assistant turn of a `messages` transcript into a
  context → answer pair; `--map flat` auto-detects prompt/answer column pairs
  (`prompt/completion`, `instruction/response`, `question/answer`, …), and a row's
  `reasoning`/`thinking` field is rendered into the `<think>` block.
- **verify / fit** — `verify_dataset.py` and `fit_corpus.py`, mandatory before
  training (see above).
- **train** — launches a `recursive_train.py` iteration on the fitted corpus.
- **learned_verified** — set only when that iteration's **gate passes** (the
  teacher graded the trained student on held-out rows). This is the licence to
  delete: `--drop` refuses while the state is anything else, because keeping the
  local copy is what makes another attempt possible.
- **drop** — deletes the copy, records how many bytes were freed.

**Full ingestion is the default (2026-10-15).** The pipeline pulls the dataset to
G: and then parses *the whole local copy*: `--limit 0` (the default) converts
every row, the scan is never capped, and every shard of the chosen split is read.
Four caps used to sit in this path, and each one made a full pull look like a
partial dataset:

| Cap | What it did |
|---|---|
| `--limit 2000` | wrote 2,000 rows and stopped |
| `--scan-limit 20000` | only examined the first 20,000 source rows — `convert` never even passed the flag |
| "keep the biggest file" | turned a 20-shard split into one file |
| `full` guessed as `full/train.jsonl` | downloaded **nothing** on a card with no `full` branch, then reported `pulled rows=0` |

Behind those, `--map chat` on a flat dataset dropped every row
(`no_user_or_assistant: 7473`) — which reads as "unusable data" when the data is
fine. `--map` now defaults to `auto` and detects chat vs flat from the first row.

Measured on the local gsm8k copy (7,473 rows in `main/train-*.parquet`):
`scanned=7473 written=7473 dropped={}` — against exactly **2,000 rows** with the
old default. Caps are now opt-in and stay honest: `--limit 2000` still gives
`scanned=2001 written=2000`.

**Failures that say what they are (2026-10-15).** `convert failed rc=1` covered
three different things. The ingester's manifest now decides the state:

| Outcome | Recorded as |
|---|---|
| rows written | `converted` |
| 0 rows, drops counted | `not_trainable`, with the reason in words — *"mapped 0 of 1306 rows — tools_present (1306/1306 — every row carries a tool schema)"* |
| no manifest at all | `error` — "no manifest written", with the log tail |

`advance` stops after such a verdict instead of failing at `verify`, and the queue
moves on to the next dataset instead of dying with the run.

**Agentic traces are mined, not refused (decision, 2026-10-15).** A row carrying a
tool schema used to be dropped whole; its plain assistant answers are exactly the
self-contained pairs a student can learn from, so they are kept now
(`--agentic keep`, the default) while every tool-call turn is still skipped.
Measured on `11-47/MiniMax-M2.1-Code-SFT`: **1306 agentic rows → 654 records**
(median answer 1695 chars) where all 1306 were previously refused. Each record
carries `provenance.agentic`, the manifest records `agentic_rows`, and the caveat
is written into the registry note and the job log — *their prompts contain tool
results the student cannot produce itself, so weigh the gate accordingly*.
`--agentic drop` restores the refusal. The other drops are visible and tunable:
`too_many_code_fences 312` (over `--max-fences 3`), `agentic_marker 198`,
`prompt_too_long 106`.

**Where the caches go (2026-10-15).** The ingester always pinned the HF caches to
G:, but `pull()`/`count_rows()` (in-process) and the `verify`/`fit`/`rewrite`
subprocesses did not — they inherited the venv default on **C:, which had 0.5 GB
free**. That is how counting one parquet shard died with
`OSError: [Errno 28] No space left on device` while G: held 1.6 TB. Now
`cache_env()`/`pin_caches()` put every ingestion process (the dashboard included)
on the G: caches, and a parquet row count reads the **footer**
(`pyarrow.parquet.ParquetFile(p).metadata.num_rows`) instead of materialising the
file into an arrow cache.

**A pull says how big it is first (2026-10-15).** `pull()` logs the file count and
bytes before fetching, says so when the card reports no sizes (the space guard
cannot run on an unknown size), and **refuses more than 50 files or 50 GB** unless
`--allow-big-pull` (or the panel's *allow a huge pull* box) is given.
`--queue-plan --remote` asks the hub up front: *"863 file(s), size not reported by
the card — a corpus, not a training set"*.

**One job per dataset.** `spawn()` refuses a second job on a repo a live job
already owns (409, with the pid), and a dashboard-spawned job no longer records
itself twice in the jobs feed — two rows for one click read like two downloads.

**Check before you pay.** `--queue-plan` (and the panel's **Plan queue** button,
`GET /api/hf/plan`) prints, per pending item, the stages the runner would execute,
what is on disk for it — pulled size, corpus and fitted rows, adapter/GGUF
artifacts, gate so far — and the preconditions that would fail. Preconditions come
in two kinds, and the split matters: a **BLOCKER** stops the item, while **BLOCKS:
training/gate only** (a down teacher on `:8080`) stops just the training tail.
Ingestion is local work on a copy already on G:, so it is never held up by a
service the gate needs hours later: the run does `pull → convert → verify → fit`,
then refuses to start the training loop (exit 3; `--force` overrides) and leaves
the corpus fitted on disk, so a later run resumes at `train`. `queue_run` also
never starts a second runner for an item that already has one.
For an end-to-end check of the machinery, bound it — `--limit 8 --max-steps 2`
still runs every stage (train → export → quantize → serve → review → gate) in
minutes; on eight rows the gate result is meaningless, and a fail keeps the local
copy, so nothing is deleted.

```powershell
python scripts\hf_datasets.py --add https://huggingface.co/datasets/org/name
python scripts\hf_datasets.py --advance org/name --split full --map auto --tag myrun
python scripts\hf_datasets.py --train org/name
python scripts\hf_datasets.py --learned-verified org/name
python scripts\hf_datasets.py --drop org/name --yes
python scripts\hf_datasets.py --queue-plan          # what the queue would do (dry run)
python scripts\hf_datasets.py --queue-run          # one dataset after another
python scripts\hf_datasets.py --unstick            # fix a runner that is alive but doing nothing
python scripts\hf_datasets.py --unstick org/name --dry-run   # just say what it would fix
python scripts\hf_datasets.py --selftest           # URL parsing, registry, drop gate, queue lock, resume decision
```

## When something wedges

Two different failures used to look the same in the panel — "working on X" forever:

| | what it looks like | how it is handled |
|---|---|---|
| **stalled** | the state says in-flight, the owner pid is gone (machine slept, console closed, OOM) | the queue itself: `stalled()` resumes the row at the stage its corpus reached |
| **wedged** | the runner is *still a process*, but its log has not grown for `--stall-min` (15) minutes **and** it burns no CPU (a file lock, a socket that will never answer) | `--unstick`: kill the tree, rewind the row to the stage its files on disk actually reached (`_resume_stage`), keep every file |

Silence alone is deliberately not enough to condemn a runner: a large shard converts
for a long time between lines. Two things keep that honest — the ingester now prints
`[ingest] working: N scanned, M written, ... every ~20s` (`AETHER_PROGRESS_SECS`
overrides it), and the CPU reading decides. `--unstick` samples CPU for
`--sample-secs` (5) before acting, and `--queue-run` does the same check while it
waits, so a wedged conversion is killed and retried on the queue's next pass instead
of holding everything behind it. Nothing is ever deleted by the fixer, and a row it
touched carries the reason in `wedge` + `note`.

## Training from the panel vs the queue (one story, 2026-10-16)

Both drive the same dataset, and they no longer disagree about it. **Start** in the
Training controls panel records its run in the dataset registry exactly as the queue
does (`state=training` plus the owner pid), so the queue panel shows the row as
active, `--queue-run` waits for it instead of starting a second trainer on the same
corpus, and the dataset picker is annotated with the queue's own view (position,
NEXT in queue, fitted vs converted, state, locked, and how many rows lose their
answer — with `health_fitted` for the fitted file and `health` for the converted one,
because those are two different files).

Start runs the *trainer*: no export, serve, review or gate. When that pid is gone the
row is reconciled to **`trained`** — the adapter exists, so what it owes is the gate —
and the ladder for that state is `("gate",)`: the queue's next pass runs
`recursive_train.py --skip-train` (export → quantize → serve → review → gate) on the
adapter that already exists, instead of retraining and throwing away the run it just
paid for. If the run left no adapter (it failed, or was pinned to zero steps), the row
falls back to the stage its *files on disk* justify — `fitted` when the fitted corpus
is there — and is trained again on the next pass.

The same actions are in the dashboard ("Hugging Face datasets" panel: link, pull +
prepare, train, run queue, per-row verify/fit/train/gate/delete), where a slow
multi-GB pull runs detached and its log tail shows in "Pipeline jobs". Every
detached runner is recorded there — the web-spawned jobs *and* `train()` /
`--queue-run` started from a console.

**The queue's lock is self-healing (2026-10-15).** An item in an in-flight state is
locked (reorder/skip refused with 409) only while the process that claimed that
state is **alive and still ours**: `train()` records the loop pid, and the pid is
re-checked against its own command line, because Windows recycles pids (the gsm8k
loop pid had come back as `Code.exe`, which made a dead run look alive). A dataset
whose runner is gone shows as **stalled**, keeps its ▲ ▼ skip buttons, and
`--queue-run` resumes it at **train** — the fitted corpus is still on disk, so it
does not re-pull and re-derive it. An item with no recorded owner keeps its lock.

### Two measured verdicts (2026-10-15)

| Dataset | Result |
|---|---|
| **`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`** (2.36 GB pulled, 15,000 rows) | **Not trainable as SFT data.** Rows measure as 1 system + **1 user turn + 65–117 tool exchanges**; there is **no `reasoning_content` field** — the "max reasoning" *is* the tool usage. Single-turn extraction kept **0 of 10,000 rows** (`assistant_tool_calls: 8169`, `agentic_marker: 1811`); harvesting every prose turn found **383,087 tool-call turns**, **71,357 marker drops** and left **25 records** whose completions were ~96 chars — "I'll inspect the repo" bridges, which would teach a student to narrate work it cannot do. Recorded as `not_trainable` with the audit attached. Using it would need a **teacher rewrite** step: read each trace, distil it into a self-contained Q&A lesson. |
| **`openai/gsm8k`** (4.75 MB pulled, 7,473 rows) | **Clean.** 2,000 rows converted: **100% complete, 0 answer-cut, terminator 100%, answers 53.3% of loss tokens, 0 duplicates, 0 anchor overlap**, median row 185 tokens (fits 1536 with room to spare). Ready to train. |

The contrast is the point: the pipeline does not judge by likes or downloads
(5.6k vs 39k), it measures what a 2.6B student could actually learn. Fable is
excellent data for a frontier model that can act; it is not SFT data here.

### Teacher rewrite: turning traces into lessons (2026-10-15)

`scripts/rewrite_traces.py` is for datasets that hold **no prompt→answer pairs**.
It reads each agentic trace, builds a **digest** (the task + the agent's own
turns + `TOOL name(args)` per call + `RESULT[n chars]` heads, with a tools
histogram up front and the middle elided to fit ~7k chars), asks the teacher for
one **strict-JSON lesson** (`question` / `answer` / `reasoning`), validates it,
and writes it in the training schema. The lessons then go through the normal
verify → fit → train → gate path.

What makes it usable at 10k-trace scale:

- **filter before paying.** Traces whose task is a preamble *are skipped before
  any teacher call* (`task_agentic`, `task_too_short`, `task_too_long`,
  `no_work_in_trace`). This is not an optimisation — on the first Fable row a
  19.5 s teacher call returned a lesson about one PowerShell sentence that
  happened to sit inside the authoring preamble.
- **resumable.** Progress is a state file plus an append-only JSONL: kill it,
  restart it, it continues where it stopped (`--limit` is a target, not a cap).
- **honest about refusal.** A lesson that leaks agentic scaffolding
  (`tool_call`, "the transcript", repo paths), or is too short/long, is dropped
  and *counted by reason*; zero lessons exits non-zero and the pipeline records
  the dataset as `not_trainable` instead of shipping junk.
- `--census N` counts what a rewrite could use, without spending teacher time;
  `--dry-run` shows one real digest and the teacher's raw reply.

**Measured verdict on `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`:
0 of 10,000 traces can yield a lesson.** The census over the whole `full` split:

| Measurement | Value |
|---|---|
| Traces kept by the filter | **0 / 10,000 (0%)** — all `task_agentic` (9,970 seed-spec, 30 autonomous-agent preambles) |
| Task size | median 5,184 chars — every trace's "user turn" is the same wave-4 authoring preamble |
| Agent narration per trace | **median 94 chars** across 25–55 assistant turns (no reasoning prose to distil) |
| Tool calls | 816,686 — **94% have a placeholder name (`unknown_tool`, 384,395) or empty arguments (`{}`)**; the named ones are Bash ×183,990, Write ×138,020, WebFetch ×25,725, Edit ×18,295 |
| Result | rewrite refused; dataset stays `not_trainable` |

So this dataset fails *both* paths for the same underlying reason: it records
that an agent worked, not what it was asked or why. To be rewritable a trace
dataset needs a real task per row, named tool calls, and enough narration for a
teacher to see the technique — which is a different kind of dataset, and the
same command will handle it (`--rewrite org/name`).

**What good looks like** (validated on `datasets/rewrite-fixture.jsonl`, three
hand-built traces with real tasks, named tools and narration): 3/3 lessons, 0
drops, **2.7 s per trace**, and the lessons teach exactly what the traces did —
"why does a test that passes in isolation fail in a full suite?" (global state
leaking; pass it locally), "how can I speed up a slow pandas job without changing
its output?" (profile, then vectorise `apply`), "how do I make a config loader
fail loudly on unknown keys without breaking optional ones?" (validate against a
known-key set and raise). At that rate 10,000 *usable* traces would be ~7.5 h of
teacher time, resumable.


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

