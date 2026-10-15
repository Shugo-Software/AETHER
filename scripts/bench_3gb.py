#!/usr/bin/env python3
"""Does Jadis fit the 3 GB GTX 1060, and how fast is it there?

`llama-bench` cannot set a context size, so this harness drives **llama-cli** on
the real runtime (KV allocation included) and measures, per (quant, ctx, backend):

  * did it load and generate at all (OOM => the card ran out of room)
  * peak VRAM sampled via nvidia-smi at 0.2 s
  * prompt-eval and eval throughput parsed from llama-cli's own timings

Device selection uses llama.cpp's device id (CUDA2 = the 3 GB card; see
docs/hardware-inventory.md §4.3 — nvidia-smi index 0 is CUDA2, the two orderings
disagree, so an index is never assumed).

Usage:
  python scripts/bench_3gb.py --dir cache/quants --ctx 2048,4096,8192
  python scripts/bench_3gb.py --gguf exports/jadis-1.0/jadis-1.0-q4_k_m.gguf \\
      --ctx 4096 --backends CUDA,Vulkan
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_single_gpu import VramSampler  # noqa: E402 — reuse the sampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDS = {
    "CUDA": os.path.join(ROOT, "builds", "cuda", "bin", "llama-cli.exe"),
    "Vulkan": os.path.join(ROOT, "builds", "vulkan", "bin", "Release",
                           "llama-cli.exe"),
}
OUT_DIR = os.path.join(ROOT, "docs", "benchmarks")
PROMPT = ("Using only what is given, explain what this function returns and name "
          "one risk when reusing it.\n\n<code>\ndef total(xs):\n    return sum("
          "x for x in xs)\n</code>")
# nvidia-smi index per llama.cpp device on this rig (hardware-inventory.md §4.3).
SMI_INDEX = {"CUDA0": 1, "CUDA1": 2, "CUDA2": 0,
             "Vulkan0": 1, "Vulkan1": 2, "Vulkan2": 0}

OOM_RE = re.compile(r"out of memory|failed to allocate|CUDA error|"
                    r"alloc_buffer|alloc failed", re.I)
# llama-cli prints a compact "[ Prompt: 476.3 t/s | Generation: 40.1 t/s ]" line
# on exit; older/verbose builds print "llama_perf_context_print: eval time ...".
PP_RE = re.compile(r"Prompt:\s*([\d.]+)\s*t/s", re.I)
TG_RE = re.compile(r"Generation:\s*([\d.]+)\s*t/s", re.I)
PP_RE_ALT = re.compile(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second", re.S)
TG_RE_ALT = re.compile(r"(?<!prompt )eval time\s*=.*?([\d.]+)\s*tokens per second",
                       re.S)


def parse_args():
    ap = argparse.ArgumentParser(description="3 GB card fit/throughput sweep")
    ap.add_argument("--dir", default=os.path.join(ROOT, "cache", "quants"),
                    help="directory of .gguf files to sweep")
    ap.add_argument("--gguf", default=None, help="a single GGUF instead of --dir")
    ap.add_argument("--ctx", default="2048,4096,8192",
                    help="comma-separated context sizes")
    ap.add_argument("--backends", default="CUDA", help="comma-separated CUDA,Vulkan")
    ap.add_argument("--device", default="2",
                    help="llama.cpp device number (2 = the 3 GB 1060)")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--n-predict", default="48")
    ap.add_argument("--timeout", type=int, default=600)
    return ap.parse_args()


def ggufs(args):
    if args.gguf:
        return [args.gguf if os.path.isabs(args.gguf)
                else os.path.join(ROOT, args.gguf)]
    return sorted(glob.glob(os.path.join(args.dir, "*.gguf")))


def run_once(exe, gguf, dev, ctx, args):
    """One llama-cli run. Returns (ok, pp_tps, tg_tps, detail)."""
    cmd = [exe, "-m", gguf, "-dev", dev, "-ngl", args.ngl, "-c", str(ctx),
           "-n", str(args.n_predict), "-p", PROMPT, "-st", "--no-warmup"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return False, None, None, "timeout after %ds" % args.timeout
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if OOM_RE.search(out):
        return False, None, None, "OOM"
    if proc.returncode != 0:
        lines = [ln for ln in out.splitlines() if ln.strip()] or [""]
        return False, None, None, "exit %d: %s" % (proc.returncode,
                                                  lines[-1][:120])
    pp = PP_RE.search(out) or PP_RE_ALT.search(out)
    tg = TG_RE.search(out) or TG_RE_ALT.search(out)
    return (True, float(pp.group(1)) if pp else None,
            float(tg.group(1)) if tg else None,
            "%.1fs" % (time.time() - t0))


def main():
    args = parse_args()
    ctxs = [int(c) for c in args.ctx.split(",") if c.strip()]
    models = ggufs(args)
    if not models:
        raise SystemExit("no .gguf found (use --dir or --gguf)")
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        exe = BUILDS.get(backend)
        if not exe or not os.path.exists(exe):
            print("[skip] %s build missing: %s" % (backend, exe))
            continue
        dev = "%s%s" % (backend, args.device)
        smi = SMI_INDEX.get(dev)
        for gguf in models:
            for ctx in ctxs:
                sampler = VramSampler()
                sampler.start()
                ok, pp, tg, detail = run_once(exe, gguf, dev, ctx, args)
                sampler.stop()
                peak = sampler.peak.get(smi) if smi is not None else None
                rows.append({
                    "timestamp_utc": stamp, "backend": backend, "device": dev,
                    "model": os.path.basename(gguf), "ctx": ctx, "ngl": args.ngl,
                    "peak_vram_mib": peak, "pp_tps": pp, "tg_tps": tg,
                    "ok": ok, "detail": detail})
                print("[3gb] %-6s %-18s ctx=%-6d %-4s peak=%s MiB pp=%s tg=%s (%s)"
                      % (backend, os.path.basename(gguf), ctx,
                         "OK" if ok else "FAIL", peak,
                         None if pp is None else round(pp, 1),
                         None if tg is None else round(tg, 1), detail),
                      flush=True)
    if not rows:
        raise SystemExit("nothing ran — check --backends/--dir")
    path = os.path.join(OUT_DIR, "three_gb_%s.csv" % stamp)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[3gb] wrote %s" % path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
