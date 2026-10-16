#!/usr/bin/env python3
"""Train several corpora back to back on **one** base-model load, grading each one.

`recursive_train.py` runs one corpus per process: every dataset pays the model load
again, and nothing tells you whether phase 3 helped or hurt. The queue's whole point
is to work through a list of datasets, so this drives that list inside a single
process instead — the trainer's own `train_corpus()` per phase, a checkpoint after
each one, and the teacher's verdict on each checkpoint before moving on.

Per phase:
  train    the corpus (fresh optimizer + fresh schedule, same loaded weights)
  checkpoint  a full PEFT adapter under adapters/<tag>/phase-NN-<corpus>
  export   merge that checkpoint on the CPU (export_model.py --gguf <quant>)
  serve    the quant on the idle 1060 (CUDA2) — training holds CUDA1, teacher CUDA0
  grade    review_students.py: the teacher scores a sample of that corpus
  history  optionally re-grade the *earlier* corpora with this checkpoint, which is
           how forgetting shows up while it is still cheap to react to

Everything is written to `logs/sweep-<tag>.json` in the same shape the loop's state
uses (iterations, per-step ok, a gate), so the dashboard's loop panel renders it
without a second UI. A phase that fails to grade is recorded as such and the sweep
continues; a phase whose training fails stops the sweep (a crash is not a data point).

    python scripts/train_sweep.py --tag jadis-pro --phase-max-steps 60 ^
        --corpus datasets/minimax-m2-1-code-sft-fit.jsonl ^
        --corpus datasets/gsm8k-fit.jsonl --corpus datasets/starcoderdata-code-fit.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

import recursive_train as rt  # noqa: E402 - run() + pass_rate(), one implementation
import train_student as ts    # noqa: E402 - the trainer itself: one code path
import aether_models as am    # noqa: E402 - which base model this sweep trains on

ROOT = rt.ROOT
LOGS = rt.LOGS
ADAPTERS = os.path.join(ROOT, "adapters")
EXPORTS = os.path.join(ROOT, "exports")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def phase_slug(corpus):
    """A short, filesystem-safe name for a corpus ('gsm8k-fit.jsonl' -> 'gsm8k')."""
    base = os.path.basename(corpus)
    for suf in ("-fit.jsonl", ".jsonl"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    return base[:28]


def sweep_path(tag):
    return os.path.join(LOGS, "sweep-%s.json" % tag)


def save(state, path, record):
    """Write the phase record into the sweep state (loop-panel-compatible shape)."""
    iters = [it for it in state.get("iterations", [])
             if it.get("iteration") != record["iteration"]]
    iters.append(record)
    state["iterations"] = sorted(iters, key=lambda x: x.get("iteration") or 0)
    state["updated_utc"] = now()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


def review_rate(tag):
    path = os.path.join(LOGS, "review-%s.jsonl" % tag)
    return rt.pass_rate(path)


def serve_start(model, port, alias):
    """Serve one checkpoint on the idle 3 GB card (training holds CUDA1)."""
    serve = os.path.join(ROOT, "scripts", "serve_student.ps1")
    return rt.run([rt.POWERSHELL, "-ExecutionPolicy", "Bypass", "-File", serve,
                   "-Model", model, "-Device", "CUDA2", "-Port", str(port),
                   "-Alias", alias, "-Ctx", "4096"],
                  "sweep-%s-serve.log" % alias)


def serve_stop(port, alias):
    serve = os.path.join(ROOT, "scripts", "serve_student.ps1")
    return rt.run([rt.POWERSHELL, "-ExecutionPolicy", "Bypass", "-File", serve,
                   "-Stop", "-Port", str(port)],
                  "sweep-%s-serve-stop.log" % alias)


def review(corpus, port, tag, args, sample=None):
    """Teacher-grades a sample of `corpus` against whatever is being served.

    `args` supplies the default sample size (and the teacher endpoint, via
    review_students.py's own defaults); `sample` overrides it for a retention check.
    """
    ok, tail = rt.run([rt.PY, "-u",
                       os.path.join(ROOT, "scripts", "review_students.py"),
                       "--student", tag, "--port", str(port),
                       "--dataset", corpus,
                       "--sample", str(sample if sample is not None else args.sample),
                       "--tag", tag],
                      "sweep-%s-review.log" % tag)
    rate, scored = review_rate(tag)
    return bool(ok), {"ok": bool(ok), "pass_rate": rate, "scored": scored,
                      "tail": tail, "corpus": os.path.basename(corpus)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="train several corpora back to back on one base-model load")
    ap.add_argument("--tag", default="jadis-pro")
    ap.add_argument("--model", default=None,
                    help="base model to train (default: %s); every phase keeps this "
                         "one load, so the whole sweep is one base" % am.DEFAULT)
    ap.add_argument("--corpus", action="append", default=[],
                    help="fitted corpus; repeatable, and the order is the phase order")
    ap.add_argument("--phase-max-steps", type=int, default=60,
                    help="optimizer steps per phase (0 = use --phase-epochs)")
    ap.add_argument("--phase-epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--prompt-loss", choices=["masked", "full"], default="masked")
    ap.add_argument("--two-card", action="store_true",
                    help="let accelerate place the model (train_student.py reads it)")
    ap.add_argument("--sample", type=int, default=40,
                    help="rows the teacher grades per phase")
    ap.add_argument("--min-pass", type=float, default=0.6)
    ap.add_argument("--history-sample", type=int, default=10,
                    help="rows per earlier corpus, re-graded by each new checkpoint "
                         "(0 = off) — this is how forgetting shows up early")
    ap.add_argument("--quant", default="q4_k_m",
                    help="quant served for grading (fits the idle 3 GB card)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-grade", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="stop after a phase that fails its gate (default: continue "
                         "and record it)")
    ap.add_argument("--resume", default=None,
                    help="adapter to continue from (phase 1 starts here)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, do nothing")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.selftest:
        return selftest()
    corpora = [os.path.abspath(c) for c in a.corpus]
    if not corpora:
        raise SystemExit("give at least one --corpus (a fitted jsonl)")
    for c in corpora:
        if not os.path.isfile(c):
            raise SystemExit("no such corpus: %s — fit it first "
                             "(hf_datasets.py --fit <repo>)" % c)
    plan = [{"phase": i + 1, "corpus": os.path.basename(c), "slug": phase_slug(c),
             "max_steps": a.phase_max_steps, "epochs": a.phase_epochs}
            for i, c in enumerate(corpora)]
    if a.dry_run:
        print(json.dumps({"tag": a.tag, "phases": plan,
                          "export": not a.skip_export, "quant": a.quant,
                          "grade": not a.skip_grade, "sample": a.sample,
                          "history_sample": a.history_sample,
                          "checkpoints": os.path.relpath(
                              os.path.join(ADAPTERS, a.tag), ROOT)}, indent=2))
        return 0

    state_path = sweep_path(a.tag)
    state = {"tag": a.tag, "kind": "sweep", "started_utc": now(),
             "updated_utc": now(), "iterations": []}
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            pass
    out_dir = am.adapter_dir(a.tag)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(LOGS, "train-%s.metrics.jsonl" % a.tag)
    tstate_path = os.path.join(LOGS, "train-%s.json" % a.tag)

    def state_write(corpus=None, **kw):
        """Keep logs/train-<tag>.json truthful while the sweep runs.

        It is what the dashboard's training panel renders, what the liveness probe
        checks, and what Stop kills — the pid here is the sweep's own, so the button
        ends the whole sweep, not just the phase in flight.
        """
        prev = {}
        try:
            with open(tstate_path, encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = {}
        prev.update(kw)
        prev["tag"] = a.tag
        prev["kind"] = "sweep"
        prev["dataset"] = os.path.basename(corpus or corpora[0])
        prev["out"] = os.path.relpath(out_dir, ROOT)
        prev["pid"] = os.getpid()
        with open(tstate_path, "w", encoding="utf-8") as fh:
            json.dump(prev, fh, indent=2)

    print("[sweep] %d phase(s) on one model load -> adapters/%s"
          % (len(corpora), a.tag), flush=True)
    # The mask matters more than it looks. With several GPUs visible, bitsandbytes'
    # 4-bit load and its transient allocations anchor on cuda:0 — and on this rig
    # cuda:0 is the 4070 Ti the teacher lives on. Phase 1 died at step 6 on
    # 2026-10-16 with 'CUDA out of memory ... GPU 0 has a total capacity of 11.99 GiB
    # of which 0 bytes is free', while the 6 GB 1060 sat idle: the sweep had been
    # started by hand without the mask the control plane always sets.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is not set, and this rig has three GPUs.\n"
            "  Without the mask the trainer's allocations anchor on cuda:0 — the card\n"
            "  the teacher runs on — so a long sweep dies with a CUDA OOM instead of\n"
            "  using the 6 GB 1060. Set it the way the control plane does:\n"
            "    $env:CUDA_VISIBLE_DEVICES='1'    # the 6 GB 1060, then run the sweep\n"
            "  or launch through the dashboard / scripts\\run-train.cmd.")
    state_write(status="loading_tokenizer", base_model=am.resolve(a.model),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[sweep] base model: %s" % am.describe(a.model), flush=True)
    tok = ts.load_tokenizer(a.model)
    dev = ts.resolve_device(a)
    state_write(status="loading_model", device=dev, base_model=am.resolve(a.model))
    resume_dir = None
    if a.resume:
        resume_dir = os.path.abspath(a.resume)
        if not os.path.isfile(os.path.join(resume_dir, "adapter_config.json")):
            raise SystemExit("--resume %r has no adapter_config.json" % a.resume)
    model = ts.load_qlora_model(a.lora_r, a.lora_alpha, dev, resume=resume_dir,
                                model=a.model)
    model.print_trainable_parameters()

    offset, results = 0, []
    for i, (corpus, ph) in enumerate(zip(corpora, plan), start=1):
        phase_dir = os.path.join(out_dir, "phase-%02d-%s" % (i, ph["slug"]))
        rec = {"iteration": i, "phase": i, "corpus": os.path.basename(corpus),
               "started_utc": now(), "steps": {}}
        print("[sweep] phase %d/%d: %s (%s steps)"
              % (i, len(corpora), ph["corpus"], ph["max_steps"] or ph["epochs"]),
              flush=True)
        state_write(status="training", phase=i, corpus=corpus, epochs=ph["epochs"],
                    batch=a.batch, grad_accum=a.grad_accum, lr=a.lr,
                    max_len=a.max_len, prompt_loss=a.prompt_loss)
        try:
            summary = ts.train_corpus(
                a, tok, model, corpus, out_dir, state_write, phase=i,
                max_steps=(ph["max_steps"] or None), epochs=ph["epochs"],
                metrics_path=metrics_path, metrics_fresh=(i == 1), offset=offset,
                phase_dir=phase_dir, resume_dir=(resume_dir if i == 1 else None))
        except Exception as e:  # noqa: BLE001 — a crash is data too, so record it
            rec["status"] = "failed:train"
            rec["steps"]["train"] = {"ok": False, "error": str(e)}
            save(state, state_path, rec)
            print("[sweep] phase %d training failed: %s" % (i, e),
                  file=sys.stderr, flush=True)
            return 1
        offset += summary["steps"]
        rec["steps"]["train"] = {"ok": True, "loss": summary["train_loss"],
                                 "steps": summary["steps"], "rows": summary["rows"],
                                 "checkpoint": os.path.relpath(phase_dir, ROOT)}
        save(state, state_path, rec)

        model_path = None
        if not a.skip_export:
            name = "%s-p%02d" % (a.tag, i)
            ok, tail = rt.run([rt.PY, "-u",
                               os.path.join(ROOT, "scripts", "export_model.py"),
                               "--adapter", phase_dir, "--name", name,
                               "--title", "%s phase %d" % (a.tag, i),
                               "--gguf", a.quant],
                              "sweep-%s-p%02d-export.log" % (a.tag, i))
            model_path = os.path.join(EXPORTS, name, "%s-%s.gguf" % (name, a.quant))
            have = os.path.isfile(model_path)
            rec["steps"]["export"] = {"ok": bool(ok and have), "tail": tail,
                                      "gguf": (os.path.relpath(model_path, ROOT)
                                               if have else None)}
            save(state, state_path, rec)
            if not have:
                rec["status"] = "failed:export"
                save(state, state_path, rec)
                if a.stop_on_fail:
                    return 1
                continue

        rates = []
        if not a.skip_grade and model_path:
            server = "%s-p%02d" % (a.tag, i)
            ok, tail = serve_start(model_path, a.port, server)
            rec["steps"]["serve"] = {"ok": ok, "tail": tail}
            if ok:
                _ok_r, payload = review(corpus, a.port, server, a)
                rec["steps"]["review"] = payload
                if payload.get("pass_rate") is not None:
                    rates.append(payload["pass_rate"])
                if i > 1 and a.history_sample:
                    hist = {}
                    for j, earlier in enumerate(corpora[:i - 1], start=1):
                        htag = "%s-p%02d-ret%02d" % (a.tag, i, j)
                        _ok_h, ph_payload = review(earlier, a.port, htag, a,
                                                   sample=a.history_sample)
                        hist[os.path.basename(earlier)] = ph_payload.get("pass_rate")
                    rec["retention"] = hist
                    rec["steps"]["retention"] = {"ok": True, "targets": hist}
            serve_stop(a.port, server)
            save(state, state_path, rec)

        rec["mean_pass_rate"] = (sum(rates) / len(rates)) if rates else None
        prev_rate = next((it.get("mean_pass_rate")
                          for it in reversed(state.get("iterations", []))
                          if it.get("iteration") != i
                          and it.get("mean_pass_rate") is not None), None)
        rec["gate"] = {"floor": a.min_pass, "previous": prev_rate,
                       "clears_floor": bool(rates)
                       and rec["mean_pass_rate"] >= a.min_pass,
                       "no_regression": (prev_rate is None
                                         or rec["mean_pass_rate"] is None
                                         or rec["mean_pass_rate"] >= prev_rate)}
        rec["gate"]["passed"] = bool(rec["gate"]["clears_floor"]
                                     and rec["gate"]["no_regression"])
        rec["status"] = "done"
        rec["finished_utc"] = now()
        save(state, state_path, rec)
        results.append(rec)
        print("[sweep] phase %d done: loss=%.4f grade=%s gate=%s"
              % (i, rec["steps"]["train"]["loss"],
                 ("%.0f%%" % (100 * rec["mean_pass_rate"]))
                 if rec["mean_pass_rate"] is not None else "n/a",
                 "PASSED" if rec["gate"]["passed"] else "NOT PASSED"), flush=True)
        if a.stop_on_fail and not rec["gate"]["passed"]:
            print("[sweep] stopping: phase %d failed its gate" % i, flush=True)
            break

    state_write(status="done", phase=len(results))
    print("[sweep] %d/%d phases finished — adapter %s, state %s"
          % (len(results), len(corpora), os.path.relpath(out_dir, ROOT),
             os.path.relpath(state_path, ROOT)), flush=True)
    return 0


def selftest():
    """Offline checks on the sweep's bookkeeping (no GPU, no training)."""
    ok = True

    def check(good, label, extra=""):
        nonlocal ok
        ok = ok and bool(good)
        print("[selftest] %s %s%s" % ("ok  " if good else "FAIL", label,
                                      (" — %s" % extra) if extra else ""))

    check(phase_slug("datasets/gsm8k-fit.jsonl") == "gsm8k",
          "phase_slug strips the path and the -fit suffix")
    check(phase_slug("datasets/starcoderdata-code-fit.jsonl") == "starcoderdata-code",
          "phase_slug keeps a multi-word name intact")
    check(sweep_path("jadis-pro").replace("\\", "/").endswith("logs/sweep-jadis-pro.json"),
          "the state lives in logs/sweep-<tag>.json")

    probe = os.path.join(LOGS, "sweep-selftest-probe.json")
    st = {"tag": "selftest-probe", "iterations": []}
    save(st, probe, {"iteration": 2, "status": "done", "mean_pass_rate": 0.5,
                     "steps": {"train": {"ok": True}}, "gate": {"passed": False}})
    save(st, probe, {"iteration": 1, "status": "done", "mean_pass_rate": 0.75,
                     "steps": {"train": {"ok": True}}, "gate": {"passed": True}})
    back = json.load(open(probe, encoding="utf-8"))
    check([it["iteration"] for it in back["iterations"]] == [1, 2],
          "phase records are kept in order")
    check(len(back["iterations"]) == 2 and back.get("updated_utc"),
          "a second phase updates the record instead of replacing it")
    check(back["iterations"][1]["gate"]["passed"] is False,
          "the gate verdict is part of the record")
    os.remove(probe)

    check(review_rate("definitely-not-a-review") == (None, 0),
          "a missing review file reports 0 scored rows instead of crashing")

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["--dry-run", "--tag", "selftest-probe",
                   "--corpus", "datasets/gsm8k-fit.jsonl",
                   "--corpus", "datasets/minimax-m2-1-code-sft-fit.jsonl"])
    try:
        plan = json.loads(buf.getvalue())
    except ValueError:
        plan = {}
    check(rc == 0 and [p["slug"] for p in plan.get("phases", [])]
          == ["gsm8k", "minimax-m2-1-code-sft"],
          "a dry run plans the phases in the order given")
    check(plan.get("phases", [{}])[0].get("max_steps") == 60,
          "the per-phase step budget reaches the plan")
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

