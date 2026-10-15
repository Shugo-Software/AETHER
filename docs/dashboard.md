# AETHER Training Dashboard

> Browser progress view for the project. Stdlib-only Python, bind-selectable HTTP
> (loopback or tailnet), 5 s auto-refresh, plus a JSON API for scripting.

## Open it

1. Start (or reuse) the server:
   `powershell -ExecutionPolicy Bypass -File scripts\start-dashboard.ps1 -Port 8091`
   (refuses to double-bind; logs to `logs/dashboard-8091.log`).
   `-Bind` selects the interface: **`tailscale`** (default — auto-detects the
   Tailscale IPv4, falls back to loopback if Tailscale is down), **`loopback`**
   (127.0.0.1 only), or an explicit IP (`0.0.0.0` = every interface — pair with a
   scoped firewall rule).
2. Browse to the printed URL:
   - tailnet devices: **`http://<tailscale-ip>:8091/`** (e.g. `http://100.x.y.z:8091/`)
   - on this machine with a loopback bind: **`http://127.0.0.1:8091/`**
   JSON snapshot: `/api/snapshot`.

### Remote access (Tailscale) firewall

A tailnet bind only helps if Windows Firewall lets tailnet peers in. **Verified
2026-10-15:** on this machine tailnet peers reach the dashboard with **no
dedicated rule** (`Get-NetFirewallRule -DisplayName "AETHER dashboard 8091
(tailnet)"` returns nothing) — the existing per-app/profile rules already cover
it. Keep the command below as the fix if a client is ever blocked (needs
elevation):

```powershell
New-NetFirewallRule -DisplayName "AETHER dashboard 8091 (tailnet)" `
  -Direction Inbound -Protocol TCP -LocalPort 8091 `
  -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any
```

`100.64.0.0/10` is Tailscale's CGNAT range, so plain-LAN/Yggdrasil clients stay
blocked. Note the **control plane (start/stop) is reachable from every tailnet
device** — acceptable for a personal tailnet; keep binds loopback-only on
networks you don't trust.

## Training controls (start / stop / select dataset)

The **Training controls** card (near the top of the page, right under the Training
stat cards) lets you run the student without touching a shell:

- **Dataset** `<select>` — every `datasets/*.jsonl`.
- **Resume from** `<select>` — continue training an existing adapter (loads its
  weights with `PeftModel.from_pretrained(..., is_trainable=True)` instead of a
  fresh LoRA). Empty = fresh. Pick e.g. `lfm25-jadis` to keep teaching Jadis.
- **tag** — run/state/logs name (`logs/train-<tag>.*`); dots are allowed, so release
  names like `jadis-1.0` work.
- **adapter dir** — *optional*; writes the adapter to `adapters/<dir>`. Leave empty
  for the legacy `adapters/lfm25-<tag>`, or set `jadis-1.0` for a release name.
- **epochs / max-len / lr** — the common knobs (defaults 3 / 1024 / 2e-4).
- **Start training** → `POST /api/train/start`; **Stop training** → `POST /api/train/stop`.

The run is launched detached (`CREATE_NEW_PROCESS_GROUP`) so it survives a
dashboard restart; the live panels (loss curve, model health, GPU, log tail)
update as it trains. Status is tracked in `logs/train-active.json` and self-heals
if the process dies.

**Safety rails (bind is loopback by default; stdlib-only):** dataset must be a bare filename
under `datasets/` (no traversal), tag is `[A-Za-z0-9_-]+`, a single concurrent
run (double-start refused), and **stop only kills a PID whose command line still
names `train_student.py`** — it never kills a foreign process. Env mirrors
`run-train.cmd`: all caches on G:, `CUDA_VISIBLE_DEVICES=1` (the 1060 6GB).

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/snapshot` | — | full snapshot (incl. `control`) |
| POST | `/api/train/start` | `{dataset, tag, out?, epochs?, lr?, max_len?, lora_r?, lora_alpha?, batch?, grad_accum?, max_steps?, resume?}` | `{ok, pid, tag, …}` (409 if already active) |
| POST | `/api/train/stop` | — | `{ok, stopped, pid, tag}` |
| POST | `/api/export/start` | `{adapter, gguf?, hub?, repo?, public?}` | `{ok, pid, adapter, …}` (409 if already exporting) |
| POST | `/api/export/stop` | — | `{ok, stopped, pid, adapter}` |

Export details (merge → GGUF → Hugging Face): see `docs/export.md`.

## What it shows

- **Training (very top)** — stat cards **step `current / total`** · last loss · step time ·
  time remaining (ETA), plus status, run identity, and a progress bar; updated live every 5 s.
  The step total comes from the trainer's metrics `total` field, with a fallback parse of the
  log tail (`N/M [`), so ETA is a real number (`h m s`) once training starts.
- **Loss curve** — inline SVG line chart with range filter **Last 10 / 25 / 50 / All**.
- **Model health** — per-step `grad_norm` ("how far each update moves the policy") and
  `learning_rate` charts.
- **Live generations** — pick a prompt → read the student's full output (written by
  `scripts/sample_student.py`, or `scripts/eval_jadis.py --samples`).
- **Run configuration** — dataset, student, epochs, lr, max-len, batch, grad-accum,
  LoRA r/α, optim, precision (from `logs/train-<tag>.json` `config`).
- **Machine (GPUs, live `nvidia-smi`)** — per-card util + VRAM bars, so you can watch the 1060
  6GB load while training and confirm the 4070 Ti stays free (or busy on teacher work).
- **Phases** — 1–9 status table (static list in `dashdata.py`, update as phases land).
- **Teacher endpoint (:8080)** — UP/DOWN via `/health` + served model IDs.
- **Datasets** — every `datasets/*.jsonl` with sample count, size, and manifest tag.
- **Training log tail** — tail of the newest `logs/train-*.log`.

## How data flows

1. `scripts/train_student.py` writes `logs/train-<tag>.json` (state + `config`) at each stage
   and appends `{step, total, epoch, loss, grad_norm, learning_rate, ts}` to
   `logs/train-<tag>.metrics.jsonl` via a `TrainerCallback` (NaN/inf → `null`, valid JSON;
   `total` = the trainer's `max_steps`).
2. `scripts/sample_student.py` writes `logs/train-<tag>.samples.json` (prompt → completion).
3. `scripts/dashdata.py` `snapshot()` collects phases, GPUs, teacher, datasets, training
   (parses metrics.jsonl → step/total/loss/grad_norm/lr series + step-time + ETA; falls back
   to parsing `N/M [` from the log tail for the total), samples.
4. `scripts/dashboard.py` renders the HTML (stdlib-only) and serves `/` + `/api/snapshot`.
   The page polls `/api/snapshot` every 5 s and redraws the live panels with inline vanilla
   JS (no external libs); a 60 s meta-refresh is the no-JS fallback.

## Files

| Path | Role |
|---|---|
| `scripts/dashdata.py` | collectors + `snapshot()` (phases, GPUs, teacher, datasets, training) |
| `scripts/dashboard.py` | HTML render + `HTTPServer` on 127.0.0.1 (`/` and `/api/snapshot`) |
| `scripts/start-dashboard.ps1` | launcher (port guard, log files) |

## Dataset queue (its own section, 2026-10-15)

The queue has its own panel near the top and is edited from the page:

| Control | What it does |
|---|---|
| ▲ / ▼ | Move an item one slot, skipping past finished items |
| skip / unskip | Takes it out of the runner's path (or back in) — **files are kept**, this is not deletion |
| add | Appends a dataset by URL/`org/name` |

**Locked while active.** Anything the pipeline is working on *right now* — a
running job for that repo, or a state of `pulling/converting/verifying/fitting/
rewriting/training` — cannot be moved or skipped. The row's buttons are disabled
with a tooltip, and the API returns 409 with the reason
(`refusing to change openai/gsm8k: the pipeline is working on it right now
(state=training)`), so a click can never desynchronise the runner from what is
actually running. Verified at both levels: 8/8 over HTTP plus direct
function checks (`scripts/hf_datasets.py --selftest` covers the drop gate; the
queue lock is covered by `docs/` + the API probes).

Each row shows position, state, **what happens next** (`pull` → `convert` →
`verify` → `fit` → `train` → `review + gate` → `delete local copy`), rows, and the
bytes it holds on G:. Deletion stays a separate, gated action.

## Machine panel: temperature and host load

The Machine table now carries **GPU temperature and power draw** per card, with
colour thresholds (amber ≥70 °C, red ≥80 °C), plus a host line for **CPU load and
RAM**. Temperatures come from `nvidia-smi`; RAM is read through the Win32
`GlobalMemoryStatusEx` struct (instant) and only CPU load needs a subprocess.

## Model endpoints panel

All three servers at once — teacher plus both students — with the **model file
each was actually launched with** (read from its own command line, so a stale
endpoint is visible rather than assumed), its GPU, its VRAM, ctx and live state.
When training takes the cards back, the students correctly show as down.

## Recursive loop panel

Every iteration with its steps (including per-student `serve.a`/`serve.b`), the
**gate verdict** (`pass`/`fail`, the mean pass rate, the floor from
`logs/loop-<tag>.json`) and the student-by-student pass rates — the loop's
behavioural record, next to the loss curve rather than instead of it.

## Performance note

`snapshot()` runs on every page poll (5 s). It is cached by component — the
endpoint process scan for 30 s, host/CPU for 30 s, endpoints for 6 s, the HF
registry for 8 s, the dataset inventory for 15 s — because the naive version cost
6.7 s per request and the browser ended up with overlapping requests in flight.
Two specific fixes came out of measuring it: endpoint liveness is now a **raw TCP
connect** (0.25 s timeout) instead of three HTTP calls (4 s), and RAM no longer
shells out to PowerShell. Steady state is 0.7–2.9 s per snapshot.

## Branding and the logo (2026-10-15)


The page carries the AETHER emblem in the browser tab and as a header lockup.
Assets live in `assets/` and are generated by `scripts/make-logo.ps1` from the
source image (black line art on near-white), which derives the variants the UI
needs — black-on-transparent for light browser chrome, white-on-transparent for
the dark page and for dark browser chrome:

| File | Used for |
|---|---|
| `assets/aether-logo.png` | The source, untouched (README) |
| `assets/aether-icon.png` | Tab icon (light) and `/favicon.ico` |
| `assets/aether-icon-dark.png` | Tab icon under `prefers-color-scheme: dark` |
| `assets/aether-logo-dark-256.png` | Header lockup |

Served by the dashboard itself from `GET /assets/<name>` (basename only — path
traversal is refused, and missing assets 404 rather than erroring). Regenerate
with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make-logo.ps1 -Source "<source png>"
```

## What the model is learning (panel, 2026-10-15)

Everything in this panel is read back from what a pipeline stage actually wrote,
never from intent — `logs/train-<tag>.json` (the trainer), `logs/verify-dataset-*.json`
(`verify_dataset.py`) and `datasets/*.filter.json` (`fit_corpus.py`). It shows:

- **label masking** — `prompt_loss` (`masked` = only answer tokens carry gradient,
  `full` = the loss covers the prompt too), `rows_used` / `rows`,
  `dropped_no_answer` (rows left with no answer at `max_len`), and the **answer
  share of loss tokens**;
- **corpus health** — rows, complete / partial / **answer cut away**, answers kept %,
  terminator survives %, duplicate groups, anchor overlap, row-token percentiles;
- **fit provenance** — rows in/out, how many answers were trimmed, why rows were
  dropped;
- **warnings** in amber, e.g. `prompt_loss=full` (gradient diluted), rows that lose
  their answer entirely, anchor prompts inside the corpus, or a corpus that has
  never been measured. These are the failure modes that are otherwise invisible
  until the model misbehaves, so they are stated in words rather than implied by a
  chart.

## Hugging Face datasets (panel, 2026-10-15)

Link a dataset by pasting **any** huggingface.co URL (or `org/name`), then drive
the pipeline from the dashboard: **Link → Pull + prepare → Train → Run queue**, plus
per-row `verify` / `fit` / `train` / `gate` / `delete`. The panel shows each
dataset's state, rows, bytes held on G:, the corpus or fitted file, why a dataset
was rejected (with its audit numbers), the queue (`pending`, `next`), the dataset
drive's held/free space, and a **Pipeline jobs** table with each job's last log
line. Long jobs (a multi-GB pull, a training iteration) run detached, so the page
never blocks. Deletion is refused until an iteration has passed the gate on that
dataset — the UI asks for confirmation and the API enforces the same rule.
Implementation: `scripts/hf_datasets.py` (`--selftest` covers URL parsing, the
registry and the drop gate). See `docs/dataset-generation.md`.

POST `/api/hf/run` with `{"action": "...", "repo": "...", "split": "...",
"map": "chat|flat", "limit": N, "max_len": N}` drives the panel; actions are the
`hf_datasets.py` CLI flags (`add`, `pull`, `convert`, `verify`, `fit`, `train`,
`learned_verified`, `drop`, `advance`, `queue_run`).

## Notes

- **Trainers must copy their stdout to `logs/train-<tag>.log`** for the log tail to
  show: relaunch background runs with output teed to that path (see `run-train.cmd`
  usage in `student-training.md`). State JSON (`logs/train-<tag>.json`) is written by
  the trainer itself.
- The phase table is hand-maintained — bump `PHASES` in `dashdata.py` when a phase
  completes (7 = student training, 8 = model export, both `done`; 9 = DPO still `todo`).
- **C: is critically small/volatile** (see `plan.md`). `run-train.cmd` redirects
  `TEMP`/`TMP`/`TMPDIR` + the HF/PIP caches to `<AETHER_ROOT>\cache\`
  so training never fills C:. Always launch training via `run-train.cmd` (never raw
  `Start-Process` with a spaced path — the Windows quoting trap).
