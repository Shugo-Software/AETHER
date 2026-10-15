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
