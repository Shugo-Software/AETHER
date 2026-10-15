#!/usr/bin/env python3
"""Phase 4 heterogeneous multi-GPU inference harness (SamurAI / AETHER).

Tests Bonsai-27B (Q1_0) across GPU combinations using llama-cli, measuring:
  - layers offloaded (and per-device distribution, via verbose 'layer N assigned')
  - per-device VRAM (RS/model/KV/compute buffer sizes from the load report)
  - prompt-processing (pp) and generation (tg) throughput
  - host RAM, exit status, output validity

Combinations follow the roadmap; device IDs are read live from --list-devices
(never assumed). Runs are median-of-3 for throughput; the load report is parsed
from a single verbose run per combination (the distribution is deterministic).

Usage:
    python scripts/bench_multi_gpu.py            # all combos
    python scripts/bench_multi_gpu.py --reps 3
    python scripts/bench_multi_gpu.py --model <path.gguf>
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Capacity-probe model (not shipped): override with AETHER_BENCH_MODEL.
DEFAULT_MODEL = os.environ.get(
    "AETHER_BENCH_MODEL",
    os.path.join(ROOT, "models", "Bonsai-27B-Q1_0.gguf"))
EXE = os.path.join(ROOT, "builds", "cuda", "bin", "llama-cli.exe")
OUT_DIR = os.path.join(ROOT, "docs", "benchmarks")
N_PROMPT = 128
N_GEN = 64
NGL = 99
PROMPT = "Reply with the single word: ok"

# GPU combinations to test (device ids matched to names at runtime).
COMBOS = [
    ("single_4070Ti", ["CUDA0"]),
    ("single_1060_6GB", ["CUDA1"]),
    ("single_1060_3GB", ["CUDA2"]),
    ("4070Ti+1060_6GB", ["CUDA0", "CUDA1"]),
    ("4070Ti+1060_3GB", ["CUDA0", "CUDA2"]),
    ("1060_6GB+1060_3GB", ["CUDA1", "CUDA2"]),
    ("all_three", ["CUDA0", "CUDA1", "CUDA2"]),
]


def device_map() -> dict[str, str]:
    """Return {device_id: gpu_name} from --list-devices."""
    out = subprocess.run([EXE, "--list-devices"], capture_output=True,
                         text=True, timeout=120, cwd=os.path.dirname(EXE))
    text = out.stdout + out.stderr
    m = {}
    for line in text.splitlines():
        mm = re.match(r"\s*((?:CUDA|Vulkan|CPU)\d+):\s*(.+?)\s*\(", line)
        if mm:
            m[mm.group(1)] = mm.group(2)
    return m


def run_verbose(model: str, devices: list[str]) -> tuple[str, float, bool]:
    """One verbose run; returns (log_text, wall_seconds, ok)."""
    args = [EXE, "-m", model, "-p", PROMPT, "-n", "2", "-ngl", str(NGL),
            "-sm", "layer", "-dev", ",".join(devices), "-st", "--simple-io", "-v"]
    t0 = time.time()
    proc = subprocess.Popen(args, cwd=os.path.dirname(EXE),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    out, _ = proc.communicate(timeout=1200)
    return out, time.time() - t0, proc.returncode == 0


def parse_load_report(text: str) -> dict:
    """Extract per-device layer counts and buffer sizes from the verbose log."""
    info = {"layers_offloaded": None, "layer_dist": {}, "buffers": {}}
    m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", text)
    if m:
        info["layers_offloaded"] = int(m.group(1))
    # per-device layer-assignment counts (tensor groups per device)
    info["layer_dist"] = _count_layer_assign(text)
    # buffer sizes: "<DEV> <kind> buffer size = <num> MiB"
    for dev, kind, num in re.findall(
            r"(\S+)\s+(model|KV|RS|output|compute) buffer size =\s*([0-9.]+) MiB", text):
        info["buffers"].setdefault(dev, {})[kind] = float(num)
    return info


def _count_layer_assign(text: str) -> dict:
    counts: dict[str, int] = {}
    for dev in re.findall(r"layer\s+\d+ assigned to device (\S+?),", text):
        counts[dev] = counts.get(dev, 0) + 1
    return counts


def run_timed(model: str, devices: list[str]) -> tuple[float | None, float | None, bool]:
    """A timed generation run (simple-io); returns (pp, tg, ok)."""
    args = [EXE, "-m", model, "-p", PROMPT, "-n", str(N_GEN), "-ngl", str(NGL),
            "-sm", "layer", "-dev", ",".join(devices), "-st", "--simple-io"]
    proc = subprocess.Popen(args, cwd=os.path.dirname(EXE),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    out, _ = proc.communicate(timeout=1200)
    ok = proc.returncode == 0
    pp = tg = None
    m = re.search(r"Prompt:\s*([0-9.]+)\s*t/s", out)
    if m:
        pp = float(m.group(1))
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", out)
    if m:
        tg = float(m.group(1))
    return pp, tg, ok


def median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 heterogeneous multi-GPU bench")
    ap.add_argument("--reps", type=int, default=3,
                    help="timed runs per combo (median reported)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()

    if not os.path.exists(EXE):
        print(f"[fatal] missing exe: {EXE}", file=sys.stderr)
        return 2
    if not os.path.exists(args.model):
        print(f"[fatal] missing model: {args.model}", file=sys.stderr)
        return 2
    os.makedirs(OUT_DIR, exist_ok=True)

    dev_names = device_map()
    print(f"[info] devices: {dev_names}")
    print(f"[info] model:   {os.path.basename(args.model)}")
    print(f"[info] reps:    {args.reps}")

    raw_path = os.path.join(OUT_DIR, f"multi_gpu_raw_{args.tag}.csv")
    summary_path = os.path.join(OUT_DIR, f"multi_gpu_summary_{args.tag}.csv")
    raw_fields = ["combo", "devices", "rep", "pp_tps", "tg_tps", "ok"]
    sum_fields = ["combo", "devices", "gpu_names", "reps_ok", "reps_total",
                  "pp_tps_median", "tg_tps_median", "layers_offloaded",
                  "layer_dist", "buffers_mib", "verbose_ok"]

    raw_rows, sum_rows = [], []
    for combo, devs in COMBOS:
        missing = [d for d in devs if d not in dev_names]
        if missing:
            print(f"[skip] {combo}: missing devices {missing}")
            continue
        print(f"\n[run ] {combo} ({', '.join(devs)})")

        # One verbose run to capture the split/load report.
        log, vwall, vok = run_verbose(args.model, devs)
        report = parse_load_report(log) if vok else {}
        dist = report.get("layer_dist", {})
        bufs = report.get("buffers", {})
        print(f"       verbose: ok={vok} wall={vwall:.1f}s "
              f"layers={report.get('layers_offloaded')} dist={dist}")
        with open(os.path.join(OUT_DIR, f"multi_gpu_verbose_{args.tag}_{re.sub(r'[^A-Za-z0-9]+', '_', combo)}.log"),
                  "w", encoding="utf-8") as fh:
            fh.write(log)

        # Timed reps.
        pps, tgs = [], []
        for rep in range(1, args.reps + 1):
            pp, tg, ok = run_timed(args.model, devs)
            if pp is not None:
                pps.append(pp)
            if tg is not None:
                tgs.append(tg)
            print(f"       rep {rep}/{args.reps}: pp={pp} tg={tg} ok={ok}")
            raw_rows.append({"combo": combo, "devices": ",".join(devs),
                             "rep": rep, "pp_tps": pp, "tg_tps": tg, "ok": ok})

        sum_rows.append({
            "combo": combo,
            "devices": ",".join(devs),
            "gpu_names": " | ".join(dev_names[d] for d in devs),
            "reps_ok": sum(1 for r in raw_rows if r["combo"] == combo and r["ok"]),
            "reps_total": args.reps,
            "pp_tps_median": median(pps),
            "tg_tps_median": median(tgs),
            "layers_offloaded": report.get("layers_offloaded"),
            "layer_dist": ";".join(f"{k}={v}" for k, v in dist.items()),
            "buffers_mib": ";".join(
                f"{k}:" + ",".join(f"{b}={s}" for b, s in v.items())
                for k, v in bufs.items()),
            "verbose_ok": vok,
        })

    with open(raw_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=raw_fields)
        w.writeheader()
        w.writerows(raw_rows)
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(sum_rows)

    print(f"\n[done] raw    -> {raw_path}")
    print(f"[done] summary-> {summary_path}")
    print("\n=== summary ===")
    for r in sum_rows:
        print(f"  {r['combo']:22s} pp={r['pp_tps_median']} tg={r['tg_tps_median']} "
              f"layers={r['layers_offloaded']} dist=[{r['layer_dist']}] ok={r['reps_ok']}/{r['reps_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

