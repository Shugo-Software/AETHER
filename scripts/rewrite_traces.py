#!/usr/bin/env python3
"""Phase 11 - teacher rewrite: turn agentic traces into self-contained lessons.

Some datasets (measured: `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`)
contain no prompt->answer pairs at all: a row is one user task plus 65-117 tool
exchanges, and the "reasoning" *is* the tool usage. Such a row cannot supervise a
2.6B student - there is nothing self-contained in it to learn.

This stage reads those traces and has the **teacher** distil each one into a
lesson the student could actually learn:

    trace -> digest (task + compressed activity) -> teacher -> strict JSON
          -> {question, answer, reasoning?} -> validated -> our training schema

What the student learns is the *technique*, not the agentic behaviour: the
question is required to be self-contained (no repo paths, tool names, file names
or transcript references) and any lesson that leaks agentic scaffolding is
dropped with a counted reason, so the corpus stays auditable.

Resumable by design: 10k traces is hours of teacher time, so progress is a state
file plus an append-only JSONL. Kill it, restart it, and it continues.

Usage:
  python scripts/rewrite_traces.py --repo org/name --from-local cache/datasets/org--name \\
      --tag fable-lessons --limit 200 --teacher-port 8080
  python scripts/rewrite_traces.py --repo org/name --from-local <dir> --dry-run
  python scripts/rewrite_traces.py --selftest        # offline: no teacher needed
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(ROOT, "datasets")
LOGS = os.path.join(ROOT, "logs")
sys.path.insert(0, SCRIPTS)

import ingest_hf_dataset as ing          # single source of truth for row reading
from review_students import post, parse_json_loose  # the proven teacher protocol

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

REWRITE_SYSTEM = (
    "You distil engineering transcripts into lessons for a small assistant. "
    "Reply with ONE JSON object and nothing else - no prose, no code fences.\n"
    '{"question": "...", "answer": "...", "reasoning": "..."}\n'
    "question: what a user would ask if they needed this lesson. Generic and "
    "self-contained: no repository, project, company, file or tool names, no "
    "paths, no reference to a transcript or conversation.\n"
    "answer: the expert answer - concrete, correct, 2 to 6 sentences, teaching "
    "the technique that the transcript's work actually demonstrates. If several "
    "techniques appear, teach the one that dominated the work; never teach a "
    "detail that only appears in passing. Short code is allowed when it is the "
    "clearest way to say it.\n"
    "reasoning: at most 3 short sentences of the key insight, or an empty string."
)

# Leakage that would make a lesson agentic rather than self-contained.
LEAK_MARKERS = (
    "tool_call", "tool call", "\"arguments\"", "function_call", "the transcript",
    "this conversation", "the user's message", "repository", "repo root",
    "pull request", "working directory", "/home/", "/users/", "c:\\",
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args():
    ap = argparse.ArgumentParser(description="Teacher rewrite for agentic traces")
    ap.add_argument("--repo", default=None, help="dataset id, for provenance")
    ap.add_argument("--from-local", required=False,
                    help="dir of a pulled copy (decided the same way the ingester does)")
    ap.add_argument("--from-local-files", default=None,
                    help="comma-separated files inside --from-local")
    ap.add_argument("--local-pattern", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=200,
                    help="lessons to WRITE (the teacher is the cost)")
    ap.add_argument("--scan-limit", type=int, default=100000,
                    help="max source rows to examine")
    ap.add_argument("--teacher-port", type=int, default=8080)
    ap.add_argument("--model", default="qwen3.5-9b-teacher")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="teacher output budget (thinking is disabled)")
    ap.add_argument("--digest-chars", type=int, default=7000,
                    help="cap on the trace digest sent to the teacher")
    ap.add_argument("--task-min-chars", type=int, default=40,
                    help="skip traces whose task is shorter than this")
    ap.add_argument("--task-max-chars", type=int, default=6000,
                    help="skip traces whose task is longer (usually a preamble)")
    ap.add_argument("--min-tool-turns", type=int, default=1,
                    help="skip traces with no tool work and under 3 steps")
    ap.add_argument("--question-min", type=int, default=15)
    ap.add_argument("--question-max", type=int, default=240)
    ap.add_argument("--answer-min", type=int, default=40)
    ap.add_argument("--answer-max", type=int, default=1600)
    ap.add_argument("--reasoning-max", type=int, default=400)
    ap.add_argument("--system", default=None,
                    help="system prompt for the lesson (default: the short "
                         "assistant instruction the corpus already uses)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build one digest, show it, and print the teacher's raw reply")
    ap.add_argument("--census", type=int, default=0,
                    help="scan N traces and tally what is rewritable (no teacher calls)")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def one_line(text, limit):
    """Whitespace-collapsed, truncated text (digests must stay machine-cheap)."""
    s = re.sub(r"\s+", " ", (text or "").strip())
    return s if len(s) <= limit else s[:limit] + " …"


def build_digest(row, max_chars=7000):
    """The teacher's view of one trace: the task + what the agent did.

    Whole traces are 10k-100k tokens of tool output; the teacher only needs the
    task, the agent's own turns, and what each tool call was asked to do. Tool
    *results* are summarised by size plus a head sample, which is what keeps this
    inside a fixed budget however long the trace is.
    """
    msgs = [m for m in (row.get("messages") or []) if isinstance(m, dict)]
    context = next((ing._text(m).strip() for m in msgs
                    if m.get("role") == "system"), "")
    task = next((ing._text(m).strip() for m in msgs
                 if m.get("role") == "user" and ing._text(m).strip()), "")
    steps, tool_turns, tool_hist = [], 0, collections.Counter()
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            txt = ing._text(m).strip()
            if txt:
                steps.append("AGENT: " + one_line(txt, 400))
            for tc in (m.get("tool_calls") or []):
                tool_turns += 1
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or "?"
                tool_hist[name] += 1
                steps.append("TOOL %s(%s)"
                             % (name, one_line(str(fn.get("arguments") or ""), 160)))
        elif role == "tool":
            txt = ing._text(m).strip()
            steps.append("RESULT[%d chars]: %s" % (len(txt), one_line(txt, 160)))
    head = "TASK: %s" % one_line(task, 1600)
    if context:
        head += "\n\nAGENT INSTRUCTIONS (truncated): %s" % one_line(context, 300)
    if tool_hist:
        # What dominated the work, up front: without it the teacher latches onto
        # whichever sentence happens to be inside the (truncated) preamble.
        head += "\n\nTOOLS USED: %s" % ", ".join(
            "%s×%d" % (n, c) for n, c in tool_hist.most_common(8))
    body = "\n".join(steps)
    room = max(max_chars - len(head) - 64, 800)
    omitted = 0
    if len(body) > room:
        lines = body.splitlines()
        keep_head, keep_tail = room * 2 // 3, room // 3
        first, used = [], 0
        for ln in lines:
            if used + len(ln) > keep_head:
                break
            first.append(ln)
            used += len(ln) + 1
        last, used = [], 0
        for ln in reversed(lines):
            if used + len(ln) > keep_tail:
                break
            last.insert(0, ln)
            used += len(ln) + 1
        omitted = max(len(lines) - len(first) - len(last), 0)
        body = ("\n".join(first) + "\n… [%d middle steps omitted] …\n" % omitted
                + "\n".join(last))
    digest = head + "\n\nACTIVITY:\n" + body
    meta = {"messages": len(msgs), "tool_turns": tool_turns, "steps": len(steps),
            "omitted_steps": omitted, "digest_chars": len(digest),
            "task_chars": len(task), "top_tools": tool_hist.most_common(3)}
    return digest, meta


def trace_task(row):
    """(context, task) of a trace — used by both the digest and the pre-filter."""
    msgs = [m for m in (row.get("messages") or []) if isinstance(m, dict)]
    context = next((ing._text(m).strip() for m in msgs
                    if m.get("role") == "system"), "")
    task = next((ing._text(m).strip() for m in msgs
                 if m.get("role") == "user" and ing._text(m).strip()), "")
    return context, task


def skip_reason(row, dmeta, args):
    """Drop hopeless traces WITHOUT paying the teacher.

    Measured on the first Fable row: its "task" is the dataset's own authoring
    preamble, so a 20s teacher call produced a lesson about one sentence inside
    that preamble. Filtering these first is what makes a 10k-trace rewrite
    affordable - and honest, since the lesson would teach pipeline etiquette
    instead of engineering technique.
    """
    context, task = trace_task(row)
    blob = (task + " " + context).lower()
    for marker in ing.AGENTIC_MARKERS:
        if marker in blob:
            return "task_agentic:" + marker.split()[0]
    if len(task) < args.task_min_chars:
        return "task_too_short"
    if len(task) > args.task_max_chars:
        return "task_too_long"
    if dmeta["tool_turns"] < args.min_tool_turns and dmeta["steps"] < 3:
        return "no_work_in_trace"
    return None



def ask_teacher(args, digest):
    """One teacher call. Returns (raw_text, seconds)."""
    user = ("TRANSCRIPT DIGEST\n\n%s\n\nWrite the lesson now (JSON only)." % digest)
    payload = {
        "messages": [{"role": "system", "content": REWRITE_SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        # Thinking would spend the whole budget before any JSON arrives (measured
        # on the judge: it returned no verdict at all until this was disabled).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    data = post(args.teacher_port, payload, args.timeout)
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    return raw, time.time() - t0


_QFIELD = re.compile(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
_AFIELD = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
_RFIELD = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def _unesc(s):
    try:
        return json.loads('"%s"' % s)
    except ValueError:
        return s.replace("\\n", "\n").replace('\\"', '"').strip()


def repair_json(raw):
    """Close a JSON object the teacher ran out of budget inside of."""
    start = raw.find("{")
    if start == -1:
        return None
    body = raw[start:].rstrip()
    for suffix in ('"}', '"}}', '"', "}}", "}"):
        try:
            obj = json.loads(body + suffix)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


def extract_lesson(raw):
    """(lesson|None, repaired, reason) from the teacher's reply.

    A truncated reply must not throw away the row: the same recovery the judge
    uses for score/pass is used here for the two required fields.
    """
    obj = parse_json_loose(raw)
    repaired = False
    if not isinstance(obj, dict):
        obj = repair_json(raw)          # a reply cut off mid-string still parses
        repaired = True
    if isinstance(obj, dict) and isinstance(obj.get("question"), str) \
            and isinstance(obj.get("answer"), str):
        reason = obj.get("reasoning")
        return ({"question": obj["question"].strip(),
                 "answer": obj["answer"].strip(),
                 "reasoning": reason.strip() if isinstance(reason, str) else ""},
                repaired, None)
    q, a = _QFIELD.search(raw), _AFIELD.search(raw)
    if q and a:
        r = _RFIELD.search(raw)
        return ({"question": _unesc(q.group(1)).strip(),
                 "answer": _unesc(a.group(1)).strip(),
                 "reasoning": _unesc(r.group(1)).strip() if r else ""},
                True, None)
    return None, False, "unparsable"


def validate_lesson(lesson, args):
    """Gate a lesson on self-containedness and length.

    This is what keeps the rewrite honest: the student must not learn repo paths,
    tool names or "as shown in the transcript" habits from data that was full of
    them. Every rejection is counted, never silent.
    """
    q = re.sub(r"\s+", " ", lesson.get("question") or "").strip()
    a = (lesson.get("answer") or "").strip()
    r = re.sub(r"\s+", " ", lesson.get("reasoning") or "").strip()
    if not q or not a:
        return None, "empty_field"
    if len(q) < args.question_min:
        return None, "question_too_short"
    if len(q) > args.question_max:
        return None, "question_too_long"
    if len(a) < args.answer_min:
        return None, "answer_too_short"
    if len(a) > args.answer_max:
        return None, "answer_too_long"
    if len(r) > args.reasoning_max:
        r = r[:args.reasoning_max].rstrip() + " …"
    blob = (q + "\n" + a + "\n" + r).lower()
    for marker in LEAK_MARKERS:
        if marker in blob:
            return None, "leak:" + marker.strip().replace(" ", "_")
    for marker in ing.AGENTIC_MARKERS:
        if marker in blob:
            return None, "agentic_marker:" + marker.split()[0]
    if a.count("```") > 2:
        return None, "too_many_code_fences"
    return {"question": q, "answer": a, "reasoning": r}, None


def load_state(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"done": [], "written": 0, "drops": {}, "started_utc": now()}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, path)


def run(args):
    """Rewrite traces until --limit lessons are written (resumable)."""
    src_files = ing.local_jsonl_files(args)
    tag = args.tag or "lessons"
    out_path = os.path.join(DATASETS, tag + ".jsonl")
    state_path = os.path.join(DATASETS, tag + ".rewrite-state.json")
    state = load_state(state_path)
    done = set(state.get("done") or [])
    drops = collections.Counter(state.get("drops") or {})
    written = int(state.get("written") or 0)
    seen_q = set(state.get("questions") or [])
    lat = []
    print("[rewrite] source: %s"
          % ", ".join(os.path.relpath(f, ROOT) for f in src_files), flush=True)
    print("[rewrite] out: %s | teacher :%d | resuming with %d lesson(s)"
          % (os.path.relpath(out_path, ROOT), args.teacher_port, written),
          flush=True)
    t_start = time.time()
    scanned = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for idx, row in ing.iter_source(args, args.repo):
            scanned += 1
            if scanned > args.scan_limit or written >= args.limit:
                break
            if idx in done:
                continue
            digest, dmeta = build_digest(row, args.digest_chars)
            if not row or not digest.strip():
                drops["empty_trace"] += 1
                done.add(idx)
                continue
            why = skip_reason(row, dmeta, args)
            if why:
                drops["skip:" + why] += 1
                done.add(idx)
                continue
            try:
                raw, secs = ask_teacher(args, digest)
                lat.append(secs)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                drops["teacher_error"] += 1
                print("[rewrite] teacher error on row %d: %s — stopping so a "
                      "restart resumes here" % (idx, e), file=sys.stderr,
                      flush=True)
                break
            lesson, repaired, why = extract_lesson(raw)
            if lesson is None:
                drops["parse:" + str(why)] += 1
                done.add(idx)
                continue
            lesson, why = validate_lesson(lesson, args)
            if lesson is None:
                drops["reject:" + str(why)] += 1
                done.add(idx)
                continue
            key = hashlib.sha1(lesson["question"].lower().encode("utf-8")).hexdigest()
            if key in seen_q:
                drops["duplicate_question"] += 1
                done.add(idx)
                continue
            seen_q.add(key)
            rec = {
                "id": "L-%06d" % written,
                "source": "teacher-rewrite:" + (args.repo or "local"),
                "recipe": "rewrite_lesson",
                "system": args.system,
                "prompt": lesson["question"],
                "completion": lesson["answer"],
                "provenance": {"repo": args.repo, "trace_row": idx,
                               "digest_chars": dmeta["digest_chars"],
                               "teacher": args.model,
                               "teacher_port": args.teacher_port,
                               "rewritten_utc": now()},
                "meta": {"trace_messages": dmeta["messages"],
                         "trace_tool_turns": dmeta["tool_turns"],
                         "trace_steps": dmeta["steps"],
                         "omitted_steps": dmeta["omitted_steps"],
                         "teacher_seconds": round(secs, 1),
                         "teacher_repaired": repaired,
                         "answer_chars": len(lesson["answer"])},
            }
            if lesson["reasoning"]:
                rec["reasoning"] = lesson["reasoning"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            done.add(idx)
            state.update({"done": sorted(done), "written": written,
                          "drops": dict(drops), "questions": sorted(seen_q),
                          "source_files": [os.path.relpath(f, ROOT)
                                           for f in src_files],
                          "params": {"limit": args.limit, "tag": tag,
                                     "digest_chars": args.digest_chars,
                                     "model": args.model}})
            if written % 10 == 0 or written == args.limit:
                save_state(state_path, state)
                print("[rewrite] %d/%d lessons | %d scanned | %s/lesson | "
                      "elapsed %.0fm | drops %s"
                      % (written, args.limit, scanned,
                         ("%.1fs" % statistics.median(lat)) if lat else "-",
                         (time.time() - t_start) / 60, dict(drops)), flush=True)
    save_state(state_path, state)
    return finish(args, tag, out_path, state, lat, written, scanned, drops)


def finish(args, tag, out_path, state, lat, written, scanned, drops):
    manifest = {
        "tag": tag, "created_utc": now(), "purpose": "training",
        "params": {k: v for k, v in vars(args).items() if k != "selftest"},
        "counts": {"scanned": scanned, "written": written,
                   "dropped": dict(sorted(drops.items()))},
        "teacher": {"model": args.model, "port": args.teacher_port,
                    "median_seconds": (round(statistics.median(lat), 1)
                                       if lat else None),
                    "digest_chars": args.digest_chars},
        "provenance": {"source_files": state.get("source_files"),
                       "resumable_state": os.path.relpath(
                           out_path.replace(".jsonl", ".rewrite-state.json"),
                           ROOT)},
        "output": {"jsonl": os.path.relpath(out_path, ROOT), "rows": written},
    }
    man_path = out_path.replace(".jsonl", ".manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print("[rewrite] wrote %d lesson(s) -> %s"
          % (written, os.path.relpath(out_path, ROOT)), flush=True)
    print("[rewrite] drops: %s" % (dict(drops) or "{}"), flush=True)
    if lat:
        med = statistics.median(lat)
        print("[rewrite] teacher latency: median %.1fs, max %.1fs — %d traces "
              "would take about %.1f h"
              % (med, max(lat), args.limit, med * args.limit / 3600), flush=True)
    print("[rewrite] manifest -> %s" % os.path.relpath(man_path, ROOT), flush=True)
    return 0 if written else 1


TRACE_FIXTURE = {
    "messages": [
        {"role": "system", "content": "You are an autonomous coding agent. "
                                      "Work in the repo root and run the tests."},
        {"role": "user", "content": "The retry helper drops the backoff after a "
                                    "restart; find out why and fix it."},
        {"role": "assistant", "content": "I'll look at the retry helper first."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "read_file",
                          "arguments": '{"path": "src/retry.py"}'}}]},
        {"role": "tool", "content": "x" * 4000},
        {"role": "assistant", "content": "The counter is module-level, so it "
                                         "resets with the process."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "run_tests",
                          "arguments": '{"suite": "fast"}'}}]},
        {"role": "tool", "content": "y" * 2000},
        {"role": "assistant", "content": "Fixed by persisting the attempt count."},
    ]
}


def selftest():
    """Offline checks: digest budget, JSON repair, validation, resume state."""
    import argparse as _ap
    import tempfile
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("[selftest] %s %s%s" % ("ok  " if cond else "FAIL", name,
                                      (" — %s" % extra) if extra else ""))

    digest, dmeta = build_digest(TRACE_FIXTURE, 7000)
    check("digest has the task", "TASK: The retry helper" in digest)
    check("digest records tool calls", "TOOL read_file" in digest)
    check("digest summarises results by size", "RESULT[4000 chars]" in digest)
    check("digest counts tool turns", dmeta["tool_turns"] == 2,
          "tool_turns=%s" % dmeta["tool_turns"])
    tight, _ = build_digest(TRACE_FIXTURE, 1200)
    check("digest respects the budget", len(tight) <= 1400, "len=%d" % len(tight))
    check("digest keeps the task when trimming", "TASK: The retry helper" in tight)

    args = _ap.Namespace(question_min=15, question_max=240, answer_min=40,
                         answer_max=1600, reasoning_max=400)
    good = ('{"question": "Why does an exponential backoff reset on restart?", '
            '"answer": "Because the attempt counter lives in process memory. '
            'Persist it with the job state and reload it on start, so the delay '
            'continues where it left off instead of restarting at one.", '
            '"reasoning": "In-memory state does not survive a restart."}')
    lesson, repaired, why = extract_lesson(good)
    check("parses a clean reply", lesson and not repaired, str(why))
    wrapped = "Sure, here is the lesson:\n```json\n" + good + "\n```"
    lesson2, _, _ = extract_lesson(wrapped)
    check("parses a reply wrapped in prose/fences",
          lesson2 and lesson2["answer"] == lesson["answer"])
    lesson3, repaired3, _ = extract_lesson(good[:int(len(good) * 0.55)])
    check("recovers a truncated reply by regex",
          lesson3 is not None and repaired3, "repaired=%s" % repaired3)
    lesson4, _, why4 = extract_lesson("no json at all here")
    check("rejects a reply with no JSON", lesson4 is None, str(why4))

    valid, why = validate_lesson(lesson, args)
    check("accepts a self-contained lesson", valid is not None, str(why))
    leaked = dict(lesson, answer="Call tool_calls with the arguments from the "
                                 "transcript, working in the repo root.")
    bad, why = validate_lesson(leaked, args)
    check("rejects agentic leakage", bad is None, str(why))
    bad2, why2 = validate_lesson(dict(lesson, answer="Too short."), args)
    check("rejects a too-short answer", bad2 is None, str(why2))
    bad3, why3 = validate_lesson(dict(lesson, question="Q" * 400), args)
    check("rejects an over-long question", bad3 is None, str(why3))

    with tempfile.TemporaryDirectory(prefix="aether-rewrite-") as tmp:
        p = os.path.join(tmp, "s.json")
        save_state(p, {"done": [1, 2, 3], "written": 3, "drops": {"x": 1}})
        back = load_state(p)
        check("resume state round-trips",
              back["done"] == [1, 2, 3] and back["written"] == 3)
        check("missing state starts empty", load_state(p + ".nope")["done"] == [])
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


def census(args):
    """Count what a rewrite could actually use, without paying the teacher.

    Answers the question that decides whether a rewrite is worth running: how
    many traces carry a self-contained task, how much narration the agent left
    behind, and whether the tool calls say anything at all (the Fable traces
    record every call as `unknown_tool` with empty arguments, which is what
    makes their digests so thin).
    """
    import statistics as st
    tally = collections.Counter()
    tools = collections.Counter()
    tasks, prose, placeholders, calls = [], [], 0, 0
    scanned = 0
    for idx, row in ing.iter_source(args, args.repo):
        scanned += 1
        if scanned > args.census:
            break
        if not row:
            tally["skip:unparsable_row"] += 1
            continue
        _, dmeta = build_digest(row, args.digest_chars)
        why = skip_reason(row, dmeta, args)
        tally["skip:" + why.split(":")[0] if why else "keep"] += 1
        _, task = trace_task(row)
        tasks.append(len(task))
        for m in (row.get("messages") or []):
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            txt = ing._text(m).strip()
            if txt:
                prose.append(len(txt))
            for tc in (m.get("tool_calls") or []):
                calls += 1
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = (fn.get("name") or "").strip()
                args_str = str(fn.get("arguments") or "").strip()
                if not name or name == "unknown_tool":
                    placeholders += 1
                if args_str in ("", "{}", "[]"):
                    placeholders += 1
                tools[name or "(none)"] += 1
        if scanned % 2000 == 0:
            print("[census] %d scanned | %s" % (scanned, dict(tally)), flush=True)
    print("\n[census] scanned %d traces of %s" % (scanned, args.repo or "local"))
    print("[census] rewritable: %s" % dict(tally))
    if tasks:
        print("[census] task chars: median %d, max %d"
              % (st.median(tasks), max(tasks)))
    if prose:
        print("[census] agent narration per trace: median %d chars, max %d"
              % (st.median(prose), max(prose)))
    if calls:
        print("[census] tool calls: %d | placeholder name or empty args: %d (%.0f%%)"
              % (calls, placeholders, 100.0 * placeholders / calls))
        print("[census] tool names seen: %s"
              % ", ".join("%s×%d" % kv for kv in tools.most_common(5)))
    keep = tally.get("keep", 0)
    print("[census] => %d/%d traces (%.1f%%) could yield a lesson"
          % (keep, scanned, 100.0 * keep / scanned if scanned else 0))
    return 0


def main():
    args = parse_args()
    if args.selftest:
        return selftest()
    if args.system is None:
        args.system = ing.DEFAULT_SYSTEM
    if not args.from_local:
        raise SystemExit("--from-local <pulled dir> is required")
    if args.census:
        return census(args)
    if args.dry_run:
        shown = skipped = 0
        for idx, row in ing.iter_source(args, args.repo):
            digest, dmeta = build_digest(row, args.digest_chars)
            why = skip_reason(row, dmeta, args) if row else "unparsable_row"
            if why:
                skipped += 1
                if skipped <= 8:
                    print("[dry-run] row %d SKIPPED (%s) | %s"
                          % (idx, why, json.dumps(dmeta)), flush=True)
                if skipped >= 40:
                    print("[dry-run] 40 skips in a row — this dataset's traces "
                          "do not carry self-contained tasks", flush=True)
                    return 3
                continue
            print("[dry-run] row %d PASSES the filter | %s"
                  % (idx, json.dumps(dmeta)))
            print("[dry-run] --- digest (first 1800 chars) ---")
            print(digest[:1800])
            try:
                raw, secs = ask_teacher(args, digest)
            except Exception as e:  # noqa: BLE001 — a dry run must explain itself
                print("[dry-run] teacher call failed: %s" % e, file=sys.stderr)
                return 2
            print("[dry-run] --- teacher raw reply (%.1fs) ---" % secs)
            print(raw)
            lesson, repaired, why2 = extract_lesson(raw)
            print("[dry-run] parsed=%s repaired=%s reason=%s"
                  % (lesson is not None, repaired, why2))
            if lesson:
                valid, why3 = validate_lesson(lesson, args)
                print("[dry-run] validated=%s reason=%s"
                      % (valid is not None, why3))
                print(json.dumps(valid or lesson, indent=1)[:1500])
            shown += 1
            if shown >= 1:
                return 0
        return 1
    if args.tag is None:
        base = (args.repo or "lessons").split("/")[-1].lower()
        args.tag = re.sub(r"[^a-z0-9]+", "-", base).strip("-")[:40] + "-lessons"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())





