#!/usr/bin/env python3
"""Phase 11 - the recursive loop: train -> export -> serve -> review -> gate -> fix.

One iteration, each step a subprocess call to the same script a human would run
(so the loop is inspectable and any step can be re-run by hand):

  1. train   - QLoRA a student on the corpus (pinned to the 6 GB card),
  2. export  - merge -> GGUF (iq4_xs for the 3 GB card, f16 for the 6 GB card),
  3. serve   - restart both student endpoints on the new weights,
  4. review  - teacher-as-judge over a corpus sample, per student,
  5. gate    - pass rate must clear a floor AND the previous iteration,
  6. fix     - teacher-written answers for the failed rows become the next
               iteration's corpus (that is what makes it recursive).

State lives in logs/loop-<tag>.json so the gate is a recorded decision.

Usage:
  python scripts/recursive_train.py --iteration 1 --tag pro1 --epochs 3 --sample 40
Resume/partial:
  python scripts/recursive_train.py --iteration 1 --tag pro1 --skip-train --skip-export
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
PY = os.path.join(ROOT, "venvs", "ml", "Scripts", "python.exe")
POWERSHELL = "powershell"

# The trainer must run with the SAME environment the dashboard/run-train.cmd use:
# train_control.TRAIN_ENV pins CUDA_VISIBLE_DEVICES=1 (the 1060 6GB) and moves the
# caches to G:. Without the mask, accelerate shards the dataloader across every
# visible GPU (500 rows became 167 batches => 3x fewer optimizer steps) and the
# model can land on the 4070 Ti, where the teacher already lives (OOM).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import train_control as _tc
    BASE_ENV = dict(_tc.TRAIN_ENV)
except Exception:  # noqa: BLE001 — keep the loop usable if that import breaks
    BASE_ENV = {}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args():
    ap = argparse.ArgumentParser(description="AETHER recursive train/review loop")
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--tag", required=True, help="model/run tag, e.g. pro1")
    ap.add_argument("--corpus",
                    default=os.path.join(ROOT, "datasets", "phase6_pro1.jsonl"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="pin the optimizer-step budget explicitly (-1 = use "
                         "epochs). Use this for reproducible runs: HF derives "
                         "total steps from the dataloader, which has been "
                         "observed to disagree with rows/batch/accum.")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--sample", type=int, default=40,
                    help="rows reviewed per student")
    ap.add_argument("--min-pass", type=float, default=0.60,
                    help="gate floor (mean pass rate)")
    ap.add_argument("--student-a-port", type=int, default=8082)
    ap.add_argument("--student-b-port", type=int, default=8083)
    ap.add_argument("--quant-a", default="iq4_xs", help="quant for the 3 GB card")
    ap.add_argument("--quant-b", default="f16", help="quant for the 6 GB card")
    ap.add_argument("--imatrix", default=os.path.join(ROOT, "cache", "quants",
                                                     "imatrix.gguf"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-serve", action="store_true")
    ap.add_argument("--skip-review", action="store_true")
    ap.add_argument("--no-correct", action="store_true",
                    help="do not build the fix corpus from failures")
    return ap.parse_args()


def run(cmd, log_name):
    """Run one step, capturing output to logs/<log_name>. Returns (ok, tail)."""
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, log_name)
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update({"PYTHONIOENCODING": "utf-8",
                "HF_HOME": os.path.join(ROOT, "cache", "hf"),
                "HF_HUB_CACHE": os.path.join(ROOT, "cache", "hf"),
                "HF_DATASETS_CACHE": os.path.join(ROOT, "cache", "hf"),
                "TEMP": os.path.join(ROOT, "cache", "temp"),
                "TMP": os.path.join(ROOT, "cache", "temp")})
    print("[loop] $ %s" % " ".join(cmd), flush=True)
    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=ROOT, env=env)
    tail = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tail = "\n".join(fh.read().splitlines()[-6:])
    except OSError:
        pass
    print("[loop]   exit=%d  log=%s" % (proc.returncode, path), flush=True)
    return proc.returncode == 0, tail


def pass_rate(jsonl_path):
    """(pass_rate, scored) from a review JSONL, or (None, 0)."""
    if not os.path.isfile(jsonl_path):
        return None, 0
    scored = passed = 0
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec.get("score"), int):
                scored += 1
                if rec.get("pass"):
                    passed += 1
    return ((passed / scored) if scored else None), scored


def save(state, path, record):
    iters = [it for it in state.get("iterations", [])
             if it.get("iteration") != record["iteration"]]
    iters.append(record)
    state["iterations"] = sorted(iters, key=lambda x: x.get("iteration", 0))
    state["updated_utc"] = now()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    state_path = os.path.join(LOGS, "loop-%s.json" % args.tag)
    state = {"tag": args.tag, "updated_utc": now(), "iterations": []}
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            pass
    prev = next((it for it in state.get("iterations", [])
                 if it.get("iteration") == args.iteration - 1), None)
    record = {"iteration": args.iteration, "started_utc": now(),
              "corpus": os.path.relpath(args.corpus, ROOT),
              "epochs": args.epochs, "steps": {}}
    export_dir = os.path.join(ROOT, "exports", args.tag)

    # 0. free the 6 GB card: training and student B cannot share it. This also
    #    guarantees the later serve step starts from a clean slate.
    if not args.skip_train or not args.skip_serve:
        serve = os.path.join(ROOT, "scripts", "serve_student.ps1")
        ok, tail = run([POWERSHELL, "-ExecutionPolicy", "Bypass", "-File", serve,
                        "-StopAll"], "loop-%s-it%d-pre-stop.log"
                       % (args.tag, args.iteration))
        record["steps"]["pre_stop"] = {"ok": ok, "tail": tail}

    # 1. train
    if not args.skip_train:
        ok, tail = run([PY, "-u", os.path.join(ROOT, "scripts", "train_student.py"),
                        "--dataset", args.corpus, "--tag", args.tag,
                        "--out", os.path.join("adapters", args.tag),
                        "--epochs", str(args.epochs),
                        "--max-len", str(args.max_len),
                        "--grad-accum", str(args.grad_accum)]
                       + (["--max-steps", str(args.max_steps)]
                          if args.max_steps and args.max_steps > 0 else []),
                       "loop-%s-it%d-train.log" % (args.tag, args.iteration))
        record["steps"]["train"] = {"ok": ok, "tail": tail,
                                    "max_steps": args.max_steps,
                                    "max_len": args.max_len}
        if not ok:
            record["status"] = "failed:train"
            save(state, state_path, record)
            return 1
    else:
        record["steps"]["train"] = {"ok": True, "skipped": True}

    # 2. export: merge -> f16 GGUF (f16 is the source for every quant).
    #    Old quant artifacts are removed first, so a failed quantize can never
    #    leave the previous iteration's weights in place to be served silently.
    if not args.skip_export:
        import glob as _glob
        stale = [p for p in _glob.glob(os.path.join(export_dir, "*.gguf"))]
        for p in stale:
            try:
                os.remove(p)
                print("[loop]   removed stale artifact %s" % os.path.basename(p),
                      flush=True)
            except OSError as e:
                print("[loop]   could not remove %s: %s" % (p, e), flush=True)
        cmd = [PY, "-u", os.path.join(ROOT, "scripts", "export_model.py"),
               "--adapter", args.tag, "--name", args.tag,
               "--title", args.tag.upper(), "--gguf", "f16"]
        ok, tail = run(cmd, "loop-%s-it%d-export.log" % (args.tag, args.iteration))
        record["steps"]["export"] = {"ok": ok, "tail": tail, "quants": ["f16"]}
        if not ok:
            record["status"] = "failed:export"
            save(state, state_path, record)
            return 1
    else:
        record["steps"]["export"] = {"ok": True, "skipped": True}

    # 2b. quantize for the cards. iq4_xs gets a FRESH importance matrix computed
    # from this iteration's weights (an imatrix from older weights is a
    # measurable quality regression), then f16 for the 6 GB card.
    if not args.skip_export:
        f16 = os.path.join(export_dir, "%s-f16.gguf" % args.tag)
        imatrix = os.path.join(ROOT, "cache", "quants",
                               "imatrix-%s.gguf" % args.tag)
        calib = os.path.join(ROOT, "cache", "quants", "calib.txt")
        bin_dir = os.path.join(ROOT, "builds", "cuda", "bin")
        if "iq4_xs" in (args.quant_a, args.quant_b):
            if os.path.isfile(calib):
                ok, tail = run([os.path.join(bin_dir, "llama-imatrix.exe"),
                                "-m", f16, "-f", calib, "-o", imatrix,
                                "-dev", "CUDA1", "-ngl", "99", "-c", "512",
                                "--chunks", "48"],
                               "loop-%s-it%d-imatrix.log" % (args.tag, args.iteration))
                record["steps"]["imatrix"] = {"ok": ok, "tail": tail,
                                              "path": imatrix}
                if ok:
                    ok2, tail2 = run([os.path.join(bin_dir, "llama-quantize.exe"),
                                      "--imatrix", imatrix, f16,
                                      os.path.join(export_dir,
                                                   "%s-iq4_xs.gguf" % args.tag),
                                      "IQ4_XS"],
                                     "loop-%s-it%d-requant.log"
                                     % (args.tag, args.iteration))
                    record["steps"]["imatrix"]["quantized"] = ok2
                    record["steps"]["imatrix"]["tail2"] = tail2
            else:
                record["steps"]["imatrix"] = {"ok": False,
                                              "error": "calibration text missing"}
        record["steps"]["export"]["quants"] = ["f16", args.quant_a, args.quant_b]

    # 3. restart both student endpoints on the new weights
    if not args.skip_serve:
        a = os.path.join(export_dir, "%s-%s.gguf" % (args.tag, args.quant_a))
        b = os.path.join(export_dir, "%s-%s.gguf" % (args.tag, args.quant_b))
        missing = [p for p in (a, b) if not os.path.isfile(p)]
        if missing:
            record["status"] = "failed:missing-artifacts"
            record["missing"] = missing
            save(state, state_path, record)
            print("[loop] refusing to serve: missing %s" % missing, file=sys.stderr)
            return 1
        serve = os.path.join(ROOT, "scripts", "serve_student.ps1")
        steps = [("serve-stop", [POWERSHELL, "-ExecutionPolicy", "Bypass", "-File",
                                 serve, "-StopAll"]),
                 ("serve-a", [POWERSHELL, "-ExecutionPolicy", "Bypass", "-File",
                              serve, "-Model", a, "-Device", "CUDA2", "-Port",
                              str(args.student_a_port), "-Alias", "jadis-a",
                              "-Ctx", "4096"]),
                 ("serve-b", [POWERSHELL, "-ExecutionPolicy", "Bypass", "-File",
                              serve, "-Model", b, "-Device", "CUDA1", "-Port",
                              str(args.student_b_port), "-Alias", "jadis-b",
                              "-Ctx", "4096"])]
        record["steps"]["serve"] = {}
        for name, cmd in steps:
            ok, tail = run(cmd, "loop-%s-it%d-%s.log"
                           % (args.tag, args.iteration, name))
            record["steps"]["serve"][name] = {"ok": ok, "tail": tail}
            if not ok:
                record["status"] = "failed:%s" % name
                save(state, state_path, record)
                return 1
    else:
        record["steps"]["serve"] = {"skipped": True}

    # 4. review both students (failures also become the next corpus)
    fix_path = os.path.join(ROOT, "datasets",
                            "%s-fix%d.jsonl" % (args.tag, args.iteration))
    reviews = {}
    if not args.skip_review:
        for label, port in (("a", args.student_a_port), ("b", args.student_b_port)):
            tag = "%s-it%d-%s" % (args.tag, args.iteration, label)
            cmd = [PY, "-u", os.path.join(ROOT, "scripts", "review_students.py"),
                   "--student", label, "--port", str(port),
                   "--dataset", args.corpus, "--sample", str(args.sample),
                   "--tag", tag]
            if not args.no_correct:
                cmd += ["--correct-failures", fix_path]
            ok, tail = run(cmd, "loop-%s-review-%s.log" % (args.tag, tag))
            rate, scored = pass_rate(os.path.join(LOGS, "review-%s.jsonl" % tag))
            reviews[label] = {"ok": ok, "pass_rate": rate, "scored": scored,
                              "tail": tail}
        record["steps"]["review"] = reviews

    # 5. gate
    rates = [v["pass_rate"] for v in reviews.values()
             if isinstance(v.get("pass_rate"), (int, float))]
    record["mean_pass_rate"] = (sum(rates) / len(rates)) if rates else None
    record["gate"] = {
        "floor": args.min_pass,
        "previous": (prev or {}).get("mean_pass_rate"),
        "clears_floor": bool(rates) and record["mean_pass_rate"] >= args.min_pass,
        "no_regression": (prev is None or record["mean_pass_rate"] is None
                          or record["mean_pass_rate"]
                          >= (prev.get("mean_pass_rate") or 0)),
    }
    record["gate"]["passed"] = (record["gate"]["clears_floor"]
                                and record["gate"]["no_regression"])
    record["fix_corpus"] = (os.path.relpath(fix_path, ROOT)
                            if os.path.isfile(fix_path) else None)
    record["status"] = "done"
    record["finished_utc"] = now()
    save(state, state_path, record)

    print("\n[loop] iteration %d: mean pass rate %s | gate %s"
          % (args.iteration,
             "%.0f%%" % (100 * record["mean_pass_rate"])
             if record["mean_pass_rate"] is not None else "n/a",
             "PASSED" if record["gate"]["passed"] else "NOT PASSED"))
    for label, v in reviews.items():
        rate = v["pass_rate"]
        print("[loop]   student %s: %s over %d rows"
              % (label, "%.0f%%" % (100 * rate) if isinstance(rate, float)
                 else "n/a", v["scored"]))
    if record["fix_corpus"]:
        print("[loop]   next-iteration corpus: %s" % record["fix_corpus"])
    print("[loop] state -> %s" % state_path)
    return 0 if record["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())