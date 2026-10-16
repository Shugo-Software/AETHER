# Dashboard review — the control plane vs. what is actually running

> Reviewed 2026-10-15 against a live run, because that is the only way this class of
> bug shows up: every panel rendered fine, the numbers were right, and the buttons
> were wrong. Symptom as reported: **"it is training but there is no way to stop
> it."**

## The measured evidence

At 06:19 UTC, three processes were training `11-47/MiniMax-M2.1-Code-SFT`, and the
dashboard's own control payload said:

```
control.active    : (null)      <- nothing to stop, so Stop stayed disabled
control.can_start : True        <- Start stayed ENABLED
training.state    : training    <- the page knew training was happening
queue.active      : 11-47/MiniMax-M2.1-Code-SFT

processes: 23656 hf_datasets.py --queue-run  -> 20128 (worker)
              -> 25480 recursive_train.py -> 9444 (worker)
                 -> 25048 train_student.py -> 23204 (worker)
```

Start was not merely cosmetic: `start_training()` refused only when *its own* record
was live, so a click would have spawned a second `train_student.py` onto the card
the loop's trainer already occupied (both use `CUDA_VISIBLE_DEVICES=1`).

## Root cause: two launchers, four records, nothing read across

| Launcher | Records the run in | Read by |
|---|---|---|
| `train_control.start_training` (UI button) | `logs/train-active.json` | `train_control.active_run` |
| `hf_datasets.train` (queue) | `logs/hf-train-active.json` + registry `loop_pid` | **nothing** — write-only |
| `export_control.start_export` (UI button) | `logs/export-active.json` | `export_control.active_export` |
| the workers themselves | `logs/train-<tag>.json`, `logs/loop-<tag>.json` | `dashdata` (display only) |

## Findings

| # | Sev | Finding | Status |
|---|---|---|---|
| 1 | HIGH | Stop disabled and Start enabled while a queue-started run trained; Start would have double-launched onto a busy card | fixed |
| 2 | HIGH | The queue's active-run record was written by nothing and read by nothing — dead state that was clearly meant to be this bridge | fixed |
| 3 | MED | Export Stop could not stop the loop's export step (same blindness, `recursive_train.py` calls `export_model.py` directly) | fixed |
| 4 | MED | No way to end a **queue job** at all: no `stop`/`cancel` verb among add/pull/convert/verify/fit/rewrite/train/drop/advance/queue-run, so a stuck multi-GB pull could only be killed by hand | fixed |
| 5 | MED | The page never said *which* run the controls referred to: panels are tag-scoped, the queue is repo-scoped, nothing joined them | fixed (control panel names role/tag/repo/source) |
| 6 | — | ~~Launcher has no single-instance guard~~ | **not a bug — my error**: `start-dashboard.ps1:33-36` already refuses when the port is in use (it is why two launcher invocations did not produce two servers). The parent/worker process pair I saw is the normal launch pattern (finding 7), not a duplicate instance. I briefly added a second guard and removed it. |
| 7 | INFO | Every stage runs as a parent + worker pair with identical command lines, so the pid in a record is not the whole job: a stop must kill the **subtree** and verify the marker on the root pid | designed for (see below) |
| 8 | LOW | Stray 0-row artifacts (`datasets/aether-state-probe-*`, `starcoderdata-lessons.jsonl`) | cleaned / noted |

Related, fixed earlier the same day: the loop only wrote `logs/loop-<tag>.json` on a
failure or at completion, so a first iteration spending ~3 h in its train step left
the panel showing another tag's failed iteration (`docs/plan.md`).

## What changed

* **`scripts/run_state.py` (new)** — one resolver over *every* source: the three
  active-run records, the loop/queue registry (`loop_pid`), the loop's own state, and
  the trainer's own state file. It verifies liveness against the **command line**
  (Windows recycles pids, so a live pid proves nothing), derives each stage's role
  (`sweep` > `loop` > `trainer` > `export` > `imatrix`/`quantize`), walks the process
  tree, and exposes one marker-checked `taskkill /F /T` kill. It also reports the
  `--queue-run` runner separately, so "stop the job" never kills the thing that moves
  on to the next dataset. `--selftest` (11 checks) proves the judgement: a pid with no
  marker is never ours, a stop on one is refused, a dry run kills nothing, the tree
  walk reaches grandchildren, and endpoints (`llama-server`) are never jobs.
* **`train_control`** — `active_run()` now delegates to the resolver, so Stop works on
  runs this module never launched; `start_training()` refuses with the live run named
  ("already running — loop pid=25480 tag=… repo=… (queue training). Stop it first.");
  `stop_training(dry_run=)` kills the subtree.
* **`export_control`** — same treatment, plus `busy_run()`: an export is refused while
  a trainer holds the card (a merge is CPU-only, imatrix/quantize are not), instead of
  failing later with an OOM.
* **Dashboard** — the control panel prints what is running (role, tag, repo, source)
  and when a run may be missing one; `/api/train/stop` and `/api/export/stop` accept
  `{"dry_run": true}`; the queue panel gained **Stop running job** and a per-row
  `stop` for the active row.
* **`hf_datasets`** — `stop` joined `JOBS` (handled inline, not spawned: the caller
  must be told *what* stopped), with `--stop [REF] [--dry-run]` and `hf_datasets.stop()`
  which resolves the repo's job, or the primary run, or the newest live job from the
  feed (so stuck pulls/converts are stoppable too). The registry row keeps its state
  and gains `note="stopped by operator (was …, pid …) — files kept"`; the queue's own
  self-heal then re-plans the row, so nothing has to be un-stuck by hand.

## Verification (against the live run, nothing killed)

```
run_state --selftest                  ALL OK (11 checks)
dashboard --selftest                  ALL OK (7 checks, div balance 110/110)
hf_datasets --selftest                ALL OK (24 checks)
hf_datasets --stop --dry-run          {"role":"loop","pid":25480,
                                       "tree":[25480,9444,25048,23204],
                                       "note":"dry run — would kill 4 process(es)"}
POST /api/train/stop {"dry_run":true} http 200, same payload
POST /api/hf/run {"action":"stop"}    http 200, same payload
snapshot.control                      can_start=False,
                                      describe="loop pid=25480 tag=minimax-m2-1-code-sft
                                      repo=11-47/MiniMax-M2.1-Code-SFT (queue training)"
page                                  "Stop running job" present, 1 per-row stop
```

The training kept running through all of it, which is the point: the control plane can
now answer "what would you stop?" without stopping it, and the same click that used to
do nothing now ends the run.

## Still open (needs its own work, tracked in the plan)

* The **multi-dataset sweep** (one weight load, per-corpus checkpoint + teacher-graded
  serve) so several corpora train back to back.
* `FineEnvs/openai-math` conversion (its columns are not in the ingest `FLAT_PAIRS`)
  and the `bigcode/starcoderdata` 289.5 GB decision — see the queue's `state`/`note`.
* The Jadis Pro export/push to Hugging Face.

