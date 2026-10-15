#!/usr/bin/env python3
"""Phase 11 — review a student with the teacher (teacher-as-judge).

Protocol, per dataset row:

  1. ask the student (llama-server, temperature 0) for an answer,
  2. ask the teacher to grade that answer against the corpus reference, with a
     strict JSON verdict {score, pass, missing, wrong, critique},
  3. write one JSONL verdict per row + a Markdown report.

The teacher is a *thinking* model: reasoning tokens are billed against
`max_tokens`, so judging disables thinking (`enable_thinking: false`) and asks for
a JSON object — otherwise the verdict never arrives (measured: 160 tokens spent
on reasoning, empty content).

Students are addressed by endpoint, so the same command reviews the "before"
(Jadis 1.0) and "after" (Jadis Pro) models — that delta is the gate.

Usage:
  python scripts/review_students.py --student a --port 8082 --dataset datasets/phase6_pro1.jsonl --sample 50
  python scripts/review_students.py --student b --port 8083 --dataset datasets/phase6_pro1.jsonl --sample 50
  python scripts/review_students.py --student a --port 8082 --eval-math --sample 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
DATASETS = os.path.join(ROOT, "datasets")

JUDGE_SYSTEM = (
    "You are a strict grader of short technical answers. You receive the ORIGINAL "
    "TASK, a REFERENCE ANSWER written by an expert, and a STUDENT ANSWER. Grade "
    "the student answer against the reference and the task's own constraints.\n"
    "Rules:\n"
    "- Factual correctness and coverage of what the task explicitly asked for.\n"
    "- Deductions: preamble or meta-commentary (\"Sure,\", \"Here's how\", describing "
    "what it is about to do), restating the question, invented APIs or facts not "
    "supported by the reference, padding, or ignoring a stated word/length limit.\n"
    "- Do NOT reward verbosity. A correct, tight answer scores higher than a long "
    "hedging one.\n"
    "- Output ONLY a JSON object, no prose, with exactly these keys:\n"
    '  {"score": <1-5 integer>, "pass": <true|false>, "missing": ["..."], '
    '"wrong": ["..."], "critique": "<one sentence, max 25 words>"}\n'
    "Be terse: at most 3 items in each array, each item under 15 words. Do not "
    "quote long passages.\n"
    "score 5 = fully correct and complete; 4 = correct with a minor gap; "
    "3 = partially correct; 2 = mostly wrong or evasive; 1 = useless or fabricated. "
    "pass = score >= 4."
)

MATH_JUDGE_SYSTEM = (
    "You are a strict mathematics reviewer. You receive a formal proof problem and "
    "a proposed APPROACH from a small model (it is NOT expected to prove the "
    "theorem). Judge only whether the approach is mathematically sensible and "
    "relevant: does it name the right objects, cite a plausible theorem or "
    "invariant, and take a coherent first step?\n"
    "Deductions: restating the problem, vague hand-waving, mentioning unrelated "
    "theory, or claiming a complete proof it does not give.\n"
    "- Output ONLY a JSON object with exactly these keys:\n"
    '  {"score": <1-5 integer>, "pass": <true|false>, "missing": ["..."], '
    '"wrong": ["..."], "critique": "<one sentence, max 25 words>"}\n'
    "Be terse: at most 3 items per array, each under 15 words.\n"
    "score 5 = a plausible, well-targeted plan; 3 = generic but not wrong; "
    "1 = unrelated or fabricated. pass = score >= 3."
)


def parse_args():
    ap = argparse.ArgumentParser(description="Teacher-as-judge student review")
    ap.add_argument("--student", required=True, help="label, e.g. a / b / pro")
    ap.add_argument("--port", type=int, default=8082, help="student llama-server port")
    ap.add_argument("--dataset", default=os.path.join(DATASETS, "phase6_pro1.jsonl"))
    ap.add_argument("--eval-math", action="store_true",
                    help="review the eval-only openai-math set instead")
    ap.add_argument("--sample", type=int, default=50, help="rows to review")
    ap.add_argument("--offset", type=int, default=0, help="skip this many rows first")
    ap.add_argument("--teacher-port", type=int, default=8080)
    ap.add_argument("--teacher-model", default="qwen3.5-9b-teacher")
    ap.add_argument("--student-max-tokens", type=int, default=512)
    ap.add_argument("--judge-max-tokens", type=int, default=1200)
    ap.add_argument("--correct-failures", default=None,
                    help="path to append teacher model answers for FAILED rows "
                         "(the next iteration's training data)")
    ap.add_argument("--correct-max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--tag", default=None, help="report name (default: <dataset>-<student>)")
    return ap.parse_args()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rows(path, sample, offset):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[offset:offset + sample]


def post(port, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask_student(port, row, args):
    """One student call at temperature 0. Returns (answer, seconds)."""
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["prompt"]})
    t0 = time.time()
    data = post(port, {"messages": msgs, "max_tokens": args.student_max_tokens,
                       "temperature": 0}, args.timeout)
    dt = time.time() - t0
    return (data["choices"][0]["message"].get("content") or "").strip(), dt


def parse_json_loose(text):
    """First balanced {...} object in the text, or None (judges sometimes add prose)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


_SCORE_RE = re.compile(r'"score"\s*:\s*([1-5])')
_PASS_RE = re.compile(r'"pass"\s*:\s*(true|false)')

CORRECT_SYSTEM = (
    "You are an expert answering a technical task. Write the answer a strong "
    "model should give: correct, complete, concrete and tight. No preamble, no "
    "meta-commentary, no restating the task. Obey every stated constraint "
    "(length limits, required sections, output formats)."
)


def correct(row, rec, args):
    """Teacher-written answer for a failed row -> next iteration's supervision."""
    user = ("TASK:\n%s\n\nREFERENCE ANSWER (expert, for grounding):\n%s\n\n"
            "A student model answered this task and was graded %s/5. Problems "
            "found: %s\n\nWrite the answer it should have given."
            % (row["prompt"][:3000], (row.get("completion") or "")[:2000],
               rec.get("score"), rec.get("critique") or "-"))
    payload = {"messages": [{"role": "system", "content": CORRECT_SYSTEM},
                            {"role": "user", "content": user}],
               "max_tokens": args.correct_max_tokens,
               "temperature": 0.2,
               "chat_template_kwargs": {"enable_thinking": False}}
    data = post(args.teacher_port, payload, args.timeout)
    return (data["choices"][0]["message"].get("content") or "").strip()


def extract_verdict(raw):
    """Verdict from the judge text. Returns (verdict|None, repaired: bool).

    The judge can still hit the token ceiling mid-JSON, so if the object will not
    parse we recover the two fields the gate actually needs (score/pass) with a
    regex rather than silently dropping the row.
    """
    verdict = parse_json_loose(raw)
    if verdict is not None and isinstance(verdict.get("score"), (int, str)):
        try:
            verdict["score"] = int(verdict["score"])
            return verdict, False
        except (TypeError, ValueError):
            pass
    if '"score"' in raw:
        m = _SCORE_RE.search(raw)
        if m:
            score = int(m.group(1))
            p = _PASS_RE.search(raw)
            return ({"score": score,
                     "pass": (p.group(1) == "true") if p else score >= 4,
                     "missing": [], "wrong": [],
                     "critique": "(verdict truncated by the token budget - "
                                 "score/pass recovered by regex)"}, True)
    return None, False


def judge(row, answer, args, math_mode):
    """Teacher verdict for one answer. Returns (verdict|None, raw, repaired)."""
    system = MATH_JUDGE_SYSTEM if math_mode else JUDGE_SYSTEM
    user = ("ORIGINAL TASK:\n%s\n\nREFERENCE ANSWER:\n%s\n\nSTUDENT ANSWER:\n%s"
            % (row["prompt"][:3000], (row.get("completion") or "")[:2500],
               answer[:4000]))
    payload = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": args.judge_max_tokens,
        "temperature": 0,
        # thinking tokens would eat the budget before any JSON arrives (measured)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = post(args.teacher_port, payload, args.timeout)
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    verdict, repaired = extract_verdict(raw)
    return verdict, raw, repaired


def main():
    args = parse_args()
    ds_path = os.path.join(DATASETS, "eval", "openai-math.jsonl") \
        if args.eval_math else args.dataset
    if not os.path.isfile(ds_path):
        raise SystemExit("dataset not found: %s" % ds_path)
    rows = load_rows(ds_path, args.sample, args.offset)
    if not rows:
        raise SystemExit("no rows selected from %s" % ds_path)

    os.makedirs(LOGS, exist_ok=True)
    stem = os.path.splitext(os.path.basename(ds_path))[0]
    tag = args.tag or "%s-%s" % (stem, args.student)
    jsonl_path = os.path.join(LOGS, "review-%s.jsonl" % tag)
    md_path = os.path.join(LOGS, "review-%s.md" % tag)
    print("[review] student=%s port=%d rows=%d dataset=%s math=%s"
          % (args.student, args.port, len(rows), os.path.basename(ds_path),
             args.eval_math), flush=True)

    results = []
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            try:
                answer, dt = ask_student(args.port, row, args)
            except Exception as e:  # noqa: BLE001 — keep going, record it
                rec = {"id": row.get("id"), "recipe": row.get("recipe"),
                       "error": "student: %s" % e}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.append(rec)
                print("[review] %d/%d STUDENT_ERROR %s" % (i, len(rows), e),
                      flush=True)
                continue
            verdict, raw, repaired = None, "", False
            try:
                verdict, raw, repaired = judge(row, answer, args, args.eval_math)
            except Exception as e:  # noqa: BLE001
                raw = "judge error: %s" % e
            v = verdict or {}
            rec = {"id": row.get("id"), "recipe": row.get("recipe"),
                   "prompt": row.get("prompt"), "answer": answer,
                   "reference": (row.get("completion") or "")[:1500],
                   "answer_chars": len(answer), "answer_seconds": round(dt, 1),
                   "score": v.get("score"), "pass": v.get("pass"),
                   "missing": v.get("missing"), "wrong": v.get("wrong"),
                   "critique": v.get("critique"), "judge_repaired": repaired,
                   "judge_raw": None if verdict else raw[:800]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            print("[review] %2d/%d %-14s score=%-4s pass=%-5s %5.1fs"
                  % (i, len(rows), row.get("recipe"), rec["score"], rec["pass"],
                     dt), flush=True)

    scored = [r for r in results if isinstance(r.get("score"), int)]
    passed = [r for r in scored if r.get("pass")]
    by_recipe = {}
    for r in scored:
        by_recipe.setdefault(r["recipe"], []).append(r)
    mean = statistics.mean([r["score"] for r in scored]) if scored else None
    pass_rate = (len(passed) / len(scored)) if scored else None
    unparsed = len(results) - len(scored)

    lines = ["# Review `%s`" % tag, "",
             "- student: `%s` (port %d), temperature 0, max %d tokens"
             % (args.student, args.port, args.student_max_tokens),
             "- judge: `%s` on :%d, thinking disabled, JSON verdict"
             % (args.teacher_model, args.teacher_port),
             "- dataset: `%s` — %d rows"
             % (os.path.relpath(ds_path, ROOT), len(rows)),
             "- generated: %s" % now(), "",
             "**pass rate: %s | mean score: %s** (%d scored, %d unanswered/unparsed)"
             % ("%.0f%%" % (100 * pass_rate) if pass_rate is not None else "n/a",
                "%.2f" % mean if mean is not None else "n/a", len(scored),
                unparsed), "",
             "## By recipe", "", "| recipe | rows | pass | mean score |",
             "|---|---:|---:|---:|"]
    for recipe, rs in sorted(by_recipe.items()):
        p = sum(1 for r in rs if r.get("pass"))
        m = statistics.mean([r["score"] for r in rs])
        lines.append("| %s | %d | %.0f%% | %.2f |"
                     % (recipe, len(rs), 100 * p / len(rs), m))
    lines += ["", "## Worst verdicts", ""]
    for r in sorted(scored, key=lambda x: x["score"])[:8]:
        lines += ["### `%s` — score %s — %s"
                  % (r["id"], r["score"], "PASS" if r["pass"] else "FAIL"), "",
                  "**critique:** %s" % (r.get("critique") or "-"), ""]
        if r.get("missing"):
            lines.append("**missing:** %s" % "; ".join(map(str, r["missing"])))
        if r.get("wrong"):
            lines.append("**wrong:** %s" % "; ".join(map(str, r["wrong"])))
        lines += ["", "<details><summary>answer</summary>", "", "```",
                  (r.get("answer") or "")[:1500], "```", "</details>", ""]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n[review] pass rate %s | mean score %s | %d rows"
          % ("%.0f%%" % (100 * pass_rate) if pass_rate is not None else "n/a",
             "%.2f" % mean if mean is not None else "n/a", len(rows)))
    for recipe, rs in sorted(by_recipe.items()):
        p = sum(1 for r in rs if r.get("pass"))
        print("[review]   %-14s %2d rows  pass %3.0f%%  mean %.2f"
              % (recipe, len(rs), 100 * p / len(rs),
                 statistics.mean([r["score"] for r in rs])))
    print("[review] verdicts -> %s" % jsonl_path)
    print("[review] report   -> %s" % md_path)

    if args.correct_failures:
        row_by_id = {r.get("id"): r for r in rows}
        failed = [r for r in scored if not r.get("pass")]
        written = 0
        os.makedirs(os.path.dirname(os.path.abspath(args.correct_failures)),
                    exist_ok=True)
        with open(args.correct_failures, "a", encoding="utf-8") as fh:
            for rec in failed:
                row = row_by_id.get(rec.get("id"))
                if not row:
                    continue
                try:
                    fix = correct(row, rec, args)
                except Exception as e:  # noqa: BLE001
                    print("[review] correction failed for %s: %s"
                          % (rec.get("id"), e), flush=True)
                    continue
                if not fix:
                    continue
                fh.write(json.dumps({
                    "id": "%s-fix" % row.get("id"),
                    "source": "teacher-correction",
                    "recipe": row.get("recipe"),
                    "system": row.get("system"),
                    "prompt": row.get("prompt"),
                    "completion": fix,
                    "provenance": {"from_row": row.get("id"),
                                   "student": args.student,
                                   "review_tag": tag},
                    "meta": {"from_score": rec.get("score"),
                             "critique": rec.get("critique"),
                             "generated_utc": now()},
                }, ensure_ascii=False) + "\n")
                written += 1
                print("[review] correction for %s (was %s/5, %d chars)"
                      % (row.get("id"), rec.get("score"), len(fix)), flush=True)
        print("[review] wrote %d teacher correction(s) -> %s"
              % (written, args.correct_failures), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
