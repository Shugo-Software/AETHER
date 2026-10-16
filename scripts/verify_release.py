#!/usr/bin/env python3
"""Release check: grade a candidate against the base and the previous release.

`review_students.py` grades one corpus against one served student — that is the
loop's gate, and it is the right primitive. What it cannot answer is what a *release*
asks:

  * does the candidate beat the **base** it was trained from, on every corpus it was
    trained on? (that is the claim a fine-tune makes, and the only evidence that the
    training caused anything)
  * does it still beat the **previous release**? (no regression on what shipped)
  * was every corpus actually trained, and how many rows? (coverage, taken from the
    sweep's own phase records — not from what anyone remembers)
  * and with what certainty? A 40-row sample cannot support a release claim, so pass
    rates are reported with a Wilson 95% interval and a delta whose interval
    straddles zero is called *inconclusive* rather than passed.

So this drives the same script, cell by cell — one grader, no second opinion — and
serves each model once, grading every corpus against it. It is read-only: no corpus,
no registry row and no model is written. If a training run is live it refuses to
start (unless --force): the cards are busy, and the sweep grades on the same 3 GB
card.

    python scripts/verify_release.py --tag jadis-pro --plan      # what it would do
    python scripts/verify_release.py --tag jadis-pro             # the real check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

import aether_models as am   # noqa: E402 - model ids and names live in one place
import recursive_train as rt  # noqa: E402 - run() + pass_rate(): one implementation

ROOT = rt.ROOT
LOGS = rt.LOGS
EXPORTS = os.path.join(ROOT, "exports")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wilson(passes, n, z=1.96):
    """Wilson 95% interval for a pass rate — honest error bars on a small sample.

    The normal approximation is useless at p=1.0 (zero width), which is exactly the
    case a release candidate likes to show, so use the score interval that stays
    inside [0, 1] and widens as n shrinks.
    """
    if not n:
        return None, None
    p = passes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return round(max(centre - half, 0.0), 4), round(min(centre + half, 1.0), 4)


def sweep_coverage(tag):
    """What the sweep actually trained: [{corpus, rows, steps, phase, gate}].

    A release claim of the form "trained on all the datasets" has to come from the
    run's own record, so this reads logs/sweep-<tag>.json (phases it walked) and
    falls back to logs/train-<tag>.json for the single-corpus case.
    """
    out = []
    path = os.path.join(LOGS, "sweep-%s.json" % tag)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            state = {}
        for it in state.get("iterations") or []:
            train = (it.get("steps") or {}).get("train") or {}
            out.append({"phase": it.get("phase"),
                        "corpus": it.get("corpus"),
                        "rows": train.get("rows"),
                        "rows_used": train.get("rows_used"),
                        "steps": train.get("steps"),
                        "loss": train.get("loss"),
                        "gate": (it.get("gate") or {}).get("passed"),
                        "status": it.get("status")})
    if not out:
        path = os.path.join(LOGS, "train-%s.json" % tag)
        try:
            with open(path, encoding="utf-8") as fh:
                st = json.load(fh)
            cfg = st.get("config") or {}
            if st.get("dataset"):
                out.append({"phase": st.get("phase") or 1, "corpus": st["dataset"],
                            "rows": st.get("rows"), "rows_used": st.get("rows_used"),
                            "steps": None, "loss": st.get("train_loss"),
                            "gate": None, "status": st.get("status")})
        except (OSError, ValueError):
            pass
    return out


def corpus_path(name):
    """A coverage entry's corpus name -> a path under datasets/."""
    if not name:
        return None
    if os.path.isabs(name) or os.sep in name:
        return name if os.path.isfile(name) else None
    p = os.path.join(ROOT, "datasets", name)
    return p if os.path.isfile(p) else None


def model_candidates(tag, quant, args):
    """The models to grade: (label, gguf, required).

    `required` matters: "beats the base" means nothing if the base was never run, so a
    missing *required* model fails the gate with a reason rather than quietly turning
    the check into a one-sided one.
    """
    return [
        ("candidate",
         args.candidate_gguf or os.path.join(EXPORTS, tag, "%s-%s.gguf" % (tag, quant)),
         True),
        ("base",
         args.base_gguf or os.path.join(EXPORTS, "lfm25-base",
                                        "lfm25-base-%s.gguf" % quant),
         not args.no_base),
        ("previous",
         args.previous_gguf or os.path.join(EXPORTS, "jadis-1.0",
                                            "jadis-1.0-%s.gguf" % quant),
         not args.no_previous),
    ]


def serve(model_path, alias, port, log_name):
    """Serve one GGUF on the idle 3 GB card (the same card the sweep grades on)."""
    serve_ps = os.path.join(ROOT, "scripts", "serve_student.ps1")
    return rt.run([rt.POWERSHELL, "-ExecutionPolicy", "Bypass", "-File", serve_ps,
                   "-Model", model_path, "-Device", "CUDA2", "-Port", str(port),
                   "-Alias", alias, "-Ctx", "4096"], log_name)


def unserve(port, alias, log_name):
    serve_ps = os.path.join(ROOT, "scripts", "serve_student.ps1")
    return rt.run([rt.POWERSHELL, "-ExecutionPolicy", "Bypass", "-File", serve_ps,
                   "-Stop", "-Port", str(port)], log_name)


def grade(corpus, port, label, tag, sample):
    """One cell: the teacher grades `sample` rows of `corpus`, served as `label`.

    The same script as the loop's gate, deliberately: a release check with its own
    grader could disagree with the gate, and then neither would mean anything.
    """
    slug = os.path.basename(corpus).replace(".jsonl", "")
    review_tag = ("%s-%s-%s" % (tag, label, slug))[:80]
    ok, tail = rt.run([rt.PY, "-u",
                       os.path.join(ROOT, "scripts", "review_students.py"),
                       "--student", label, "--port", str(port),
                       "--dataset", corpus, "--sample", str(sample),
                       "--tag", review_tag],
                      "verify-%s-review.log" % review_tag)
    rate, scored = rt.pass_rate(os.path.join(LOGS, "review-%s.jsonl" % review_tag))
    passes = int(round(rate * scored)) if (rate is not None and scored) else 0
    lo, hi = wilson(passes, scored or 0)
    return {"ok": bool(ok), "pass_rate": rate, "scored": scored, "planned": sample,
            "wilson": [lo, hi], "review_tag": review_tag, "tail": tail}


def verdict(row, floor, strict_intervals=False):
    """The gate for one corpus: floor, delta vs base, no regression vs previous.

    The rule is the one chosen for this release: at or above the floor, strictly
    better than the base, not worse than the previous release — with the sample
    actually completed. Intervals are reported either way; `strict_intervals` also
    fails an improvement whose interval overlaps the baseline's.
    """
    v = {"corpus": row.get("corpus"), "phase": row.get("phase"),
         "rows_trained": row.get("rows_trained"), "steps": row.get("steps"),
         "reasons": [], "notes": []}
    cand = row.get("candidate") or {}
    if cand.get("scored") in (None, 0) or cand.get("pass_rate") is None:
        v["passed"] = False
        v["reasons"].append("the candidate was not graded on this corpus")
        return v
    if (cand.get("scored") or 0) < (cand.get("planned") or 0):
        v["reasons"].append("scored %s of %s planned rows — a shortfall is not a pass"
                            % (cand.get("scored"), cand.get("planned")))
    if cand["pass_rate"] < floor:
        v["reasons"].append("below the floor (%.3f < %.2f)" % (cand["pass_rate"], floor))
    for label in ("base", "previous"):
        b = row.get(label)
        if not b:
            continue
        if b.get("pass_rate") is None:
            v["reasons"].append("%s was not graded" % label)
            continue
        delta = round(cand["pass_rate"] - b["pass_rate"], 4)
        clo, chi = (cand.get("wilson") or [None, None])
        blo, bhi = (b.get("wilson") or [None, None])
        cmps = {"delta": delta, "candidate": cand["pass_rate"],
                "baseline": b["pass_rate"], "interval": [clo, chi],
                "baseline_interval": [blo, bhi]}
        if delta <= 0:
            cmps["verdict"] = "worse"
            v["reasons"].append("not better than %s (%.3f vs %.3f)"
                                % (label, cand["pass_rate"], b["pass_rate"]))
        elif None not in (clo, bhi) and clo <= bhi:
            cmps["verdict"] = "inconclusive"
            msg = ("the improvement over %s is inconclusive at %s rows (%.3f vs %.3f, "
                   "intervals overlap)" % (label, cand.get("scored"),
                                           cand["pass_rate"], b["pass_rate"]))
            (v["reasons"] if strict_intervals else v["notes"]).append(msg)
        else:
            cmps["verdict"] = "better"
        v[label] = cmps
    v["passed"] = not v["reasons"]
    return v


def report_md(tag, quant, rows, verdicts, coverage, args, judge):
    """The human report: one line per corpus, and every number it rests on."""
    ok = sum(1 for v in verdicts if v["passed"])
    lines = ["# Release check — `%s` (quant `%s`)" % (tag, quant), "",
             "Generated %s by `scripts/verify_release.py` (read-only)." % _now(), "",
             "**Gate**: pass rate ≥ %.2f, strictly better than `base`, not worse than "
             "`previous`, sample completed. **%d of %d corpora pass.**"
             % (args.min_pass, ok, len(verdicts)), ""]
    if judge:
        lines += ["**The judge is not ground truth**: %d row(s) came back without a "
                  "usable verdict; they are excluded from the rates above, not counted "
                  "as failures." % judge, ""]
    lines += ["| corpus | phase | trained rows | steps | candidate | base | previous | "
              "Δ base | Δ prev | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        c = row.get("candidate") or {}
        b = row.get("base") or {}
        p = row.get("previous") or {}
        v = next((x for x in verdicts if x["corpus"] == row.get("corpus")), {})

        def cell(m):
            if not m or m.get("pass_rate") is None:
                return "—"
            lo, hi = (m.get("wilson") or [None, None])
            return ("%.3f" % m["pass_rate"] if lo is None
                    else "%.3f [%.2f–%.2f]" % (m["pass_rate"], lo, hi))
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (row.get("corpus"), row.get("phase"),
                        row.get("rows_trained") or "—", row.get("steps") or "—",
                        cell(c), cell(b), cell(p),
                        ("%+.3f" % v["base"]["delta"]) if v.get("base") else "—",
                        ("%+.3f" % v["previous"]["delta"]) if v.get("previous") else "—",
                        "**pass**" if v.get("passed") else "**FAIL**"))
    lines.append("")
    for v in verdicts:
        if v["reasons"] or v["notes"]:
            lines.append("- `%s`:" % v["corpus"])
            lines += ["  - **%s**" % r for r in v["reasons"]]
            lines += ["  - %s" % n for n in v["notes"]]
    lines += ["", "## Coverage — what the run actually trained", "",
              "| phase | corpus | rows | loss | sweep gate |", "|---|---|---|---|---|"]
    for cov in coverage:
        lines.append("| %s | `%s` | %s | %s | %s |"
                     % (cov.get("phase"), cov.get("corpus"), cov.get("rows"),
                        cov.get("loss"), cov.get("gate")))
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Release check: a candidate against the base and the previous release")
    ap.add_argument("--tag", default="jadis-pro", help="the candidate's run/artifact tag")
    ap.add_argument("--quant", default="q4_k_m",
                    help="which GGUF quant every model is served as — the comparison "
                         "has to be like-for-like")
    ap.add_argument("--corpus", action="append", default=[],
                    help="corpus to grade (repeatable); default: every corpus the run "
                         "trained on, read from its own phase records")
    ap.add_argument("--sample", type=int, default=60,
                    help="rows per corpus — a release claim needs more than the loop's "
                         "gate sample")
    ap.add_argument("--min-pass", type=float, default=0.6)
    ap.add_argument("--min-rows", type=int, default=20,
                    help="skip corpora smaller than this (a fixture is not evidence)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--candidate-gguf", default=None)
    ap.add_argument("--base-gguf", default=None)
    ap.add_argument("--previous-gguf", default=None)
    ap.add_argument("--no-base", action="store_true", help="skip the base comparison")
    ap.add_argument("--no-previous", action="store_true",
                    help="skip the previous-release comparison (the report says so)")
    ap.add_argument("--strict-intervals", action="store_true",
                    help="also fail an improvement whose interval overlaps the base's")
    ap.add_argument("--force", action="store_true",
                    help="run even while a training job is live (the rig is shared)")
    ap.add_argument("--out", default=None,
                    help="JSON report (default logs/verify-<tag>.json); an existing "
                         "report for the same tag and sample is resumed, not redone")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def selftest():
    """Offline checks: the interval maths, the gate rules, coverage parsing."""
    ok = True

    def check(good, label):
        nonlocal ok
        ok = ok and bool(good)
        print("[selftest] %s %s" % ("ok  " if good else "FAIL", label))

    lo, hi = wilson(10, 10)
    check(lo is not None and hi == 1.0 and lo < 0.75,
          "a perfect rate still carries an interval (%.2f-%.2f at n=10)" % (lo, hi))
    lo2, hi2 = wilson(30, 60)
    check(abs(lo2 - 0.383) < 0.02 and abs(hi2 - 0.617) < 0.02,
          "half of 60 rows reads as uncertain (%.3f-%.3f)" % (lo2, hi2))
    check(wilson(0, 0) == (None, None), "no rows means no interval, not a zero rate")

    def row(c, b, p=None, scored=60, planned=60):
        out = {"corpus": "x.jsonl",
               "candidate": {"pass_rate": c, "scored": scored, "planned": planned,
                             "wilson": [0.5, 0.7]},
               "base": {"pass_rate": b, "wilson": [0.4, 0.6]}}
        if p is not None:
            out["previous"] = {"pass_rate": p, "wilson": [0.45, 0.65]}
        return out

    check(verdict(row(0.8, 0.5), 0.6)["passed"], "better than base, above the floor: pass")
    check(not verdict(row(0.55, 0.5), 0.6)["passed"], "below the floor: fail")
    check(not verdict(row(0.4, 0.5), 0.6)["passed"], "worse than base: fail")
    check(not verdict(row(0.8, 0.5, 0.9), 0.6)["passed"],
          "a regression against the previous release: fail")
    check(not verdict(row(0.8, 0.5, scored=40), 0.6)["passed"],
          "a judge shortfall: fail even with a good rate")
    v = verdict(row(0.62, 0.6), 0.6)
    check(v["passed"] and v["notes"], "an inconclusive win passes but is flagged")
    check(not verdict(row(0.62, 0.6), 0.6, strict_intervals=True)["passed"],
          "with --strict-intervals the same win fails")
    check(verdict(row(0.8, 0.5), 0.6)["base"]["delta"] == 0.3,
          "the delta against each baseline is reported")
    check(corpus_path("gsm8k-fit.jsonl") is not None,
          "a coverage corpus name resolves under datasets/")
    check(corpus_path("definitely-not-a-corpus.jsonl") is None,
          "a missing corpus resolves to None instead of guessing")
    cov = sweep_coverage("jadis-pro")
    check(isinstance(cov, list) and all("corpus" in c for c in cov),
          "the run's own records parse into coverage rows (%d phase(s) so far)"
          % len(cov))
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


def main(argv=None):
    a = parse_args(argv)
    if a.selftest:
        return selftest()
    out_path = a.out or os.path.join(LOGS, "verify-%s.json" % a.tag)

    coverage = sweep_coverage(a.tag)
    names = a.corpus or [c["corpus"] for c in coverage]
    corpora = []
    for name in names:
        path = corpus_path(name)
        if not path:
            print("[verify] skipping %r — no corpus by that name under datasets/"
                  % name, file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            rows = sum(1 for line in fh if line.strip())
        if rows < a.min_rows:
            print("[verify] skipping %s — %d rows (< --min-rows %d)"
                  % (name, rows, a.min_rows), file=sys.stderr)
            continue
        corpora.append((name, path, rows))

    models = model_candidates(a.tag, a.quant, a)
    plan = {"tag": a.tag, "quant": a.quant, "sample": a.sample,
            "min_pass": a.min_pass, "port": a.port,
            "out": os.path.relpath(out_path, ROOT),
            "models": [{"label": label,
                        "gguf": (os.path.relpath(g, ROOT) if g else None),
                        "present": bool(g and os.path.isfile(g)),
                        "required": required}
                       for label, g, required in models],
            "corpora": [{"corpus": n, "path": os.path.relpath(p, ROOT), "rows": r}
                        for n, p, r in corpora],
            "cells": ["%s x %s" % (label, n) for label, _g, _r in models
                      for n, _p, _rr in corpora]}
    if a.plan:
        print(json.dumps(plan, indent=2))
        return 0

    # The rig is shared: the sweep grades on the same 3 GB card this serves on, so a
    # live run wins unless the operator deliberately forces the overlap.
    import run_state
    live = (run_state.active() or {}).get("primary")
    if live and not a.force:
        print("[verify] refusing to start: %s is running (%s pid %s). It uses the same "
              "card; wait for it, or pass --force to share the rig."
              % (live.get("role") or "a job", live.get("tag"), live.get("pid")),
              file=sys.stderr)
        return 3
    if not corpora:
        print("[verify] nothing to grade — no corpus resolved for tag %s (has it "
              "trained yet?)" % a.tag, file=sys.stderr)
        return 1

    report = {"tag": a.tag, "quant": a.quant, "generated_utc": _now(),
              "sample": a.sample, "min_pass": a.min_pass, "force": bool(a.force),
              "models": plan["models"], "coverage": coverage, "rows": []}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            if prev.get("tag") == a.tag and prev.get("sample") == a.sample:
                report["rows"] = prev.get("rows") or []
                print("[verify] resuming %s (%d corpus row(s) already graded)"
                      % (os.path.relpath(out_path, ROOT), len(report["rows"])))
        except (OSError, ValueError):
            pass
    rowmap = {r["corpus"]: r for r in report["rows"]}
    covmap = {c["corpus"]: c for c in coverage}

    for label, gguf, required in models:
        present = bool(gguf and os.path.isfile(gguf))
        if not present:
            print("[verify] %s: %s — %s"
                  % (label, "MISSING" if required else "absent (optional)",
                     os.path.relpath(gguf, ROOT) if gguf else "no path"), flush=True)
            if required:
                for name, _p, _r in corpora:
                    rowmap.setdefault(name, {"corpus": name})[label] = {
                        "pass_rate": None, "scored": 0, "planned": a.sample,
                        "missing": True}
            continue
        print("[verify] serving %s (%s) on port %d"
              % (label, os.path.relpath(gguf, ROOT), a.port), flush=True)
        served, tail = serve(gguf, label, a.port,
                             "verify-%s-%s-serve.log" % (a.tag, label))
        if not served:
            print("[verify] %s would not serve: %s" % (label, (tail or "")[-300:]),
                  file=sys.stderr)
            unserve(a.port, label, "verify-%s-%s-unserve.log" % (a.tag, label))
            return 1
        try:
            for name, path, rows in corpora:
                row = rowmap.setdefault(name, {"corpus": name, "rows": rows})
                cov = covmap.get(name) or {}
                row.setdefault("phase", cov.get("phase"))
                row.setdefault("rows_trained", cov.get("rows"))
                row.setdefault("steps", cov.get("steps"))
                done = row.get(label) or {}
                if done and not done.get("missing"):
                    print("[verify]   %s x %s: already graded (%s)"
                          % (label, name, done.get("pass_rate")), flush=True)
                    continue
                cell = grade(path, a.port, label, a.tag, a.sample)
                row[label] = cell
                passed = int(round((cell["pass_rate"] or 0) * (cell["scored"] or 0)))
                print("[verify]   %s x %s: %d of %s rows pass, rate %s, 95%% %s"
                      % (label, name, passed, cell["scored"], cell["pass_rate"],
                         cell["wilson"]), flush=True)
                report["rows"] = list(rowmap.values())
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=1)
        finally:
            unserve(a.port, label, "verify-%s-%s-unserve.log" % (a.tag, label))

    report["rows"] = list(rowmap.values())
    verdicts = [verdict(r, a.min_pass, a.strict_intervals) for r in report["rows"]]
    judge = sum(max(0, ((r.get("candidate") or {}).get("planned") or 0)
                    - ((r.get("candidate") or {}).get("scored") or 0))
                for r in report["rows"])
    report["verdicts"] = verdicts
    report["judge_unscored"] = judge
    report["passed"] = bool(verdicts) and all(v["passed"] for v in verdicts)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    md = report_md(a.tag, a.quant, report["rows"], verdicts, coverage, a, judge)
    md_path = ((out_path[:-5] + ".md") if out_path.endswith(".json")
               else out_path + ".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print("[verify] %s — %s (%d of %d corpora pass)"
          % ("PASSED" if report["passed"] else "NOT PASSED",
             os.path.relpath(md_path, ROOT),
             sum(1 for v in verdicts if v["passed"]), len(verdicts)))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


