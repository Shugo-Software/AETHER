#!/usr/bin/env python3
"""Phase 3 single-GPU benchmark harness (SamurAI / AETHER).

Runs llama-bench on each (backend, device) with FIXED, equivalent settings,
repeats each run N times, samples peak VRAM during execution (NVIDIA via
nvidia-smi; AMD RX 580 has no CLI monitor so VRAM is recorded as None there),
and writes machine-readable CSV.

Fixed settings (equivalent across all GPUs):
  - model/quant : Qwen2.5-0.5B-Instruct Q4_K_M (fits on every card + both backends)
  - ngl         : 99 (full offload)
  - n-prompt    : 512   (prompt-processing throughput)
  - n-gen       : 128   (generation throughput)
  - split-mode  : none  (single-GPU: no layer/tensor split)
  - repetitions : 5 per llama-bench invocation; we invoke 3 times and report median
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Probe model, not shipped: point AETHER_BENCH_MODEL at any small GGUF.
MODEL = os.environ.get("AETHER_BENCH_MODEL",
                       os.path.join(ROOT, "models",
                                    "qwen2.5-0.5b-instruct-q4_k_m.gguf"))
BUILDS = {
    "CUDA": os.path.join(ROOT, "builds", "cuda", "bin", "llama-bench.exe"),
    "Vulkan": os.path.join(ROOT, "builds", "vulkan", "bin", "Release", "llama-bench.exe"),
}
RAW_DIR = os.path.join(ROOT, "docs", "benchmarks")
N_PROMPT = 512
N_GEN = 128
NGL = 99


def list_devices(exe: str) -> list[str]:
    """Return device ids like ['CUDA0','CUDA1'] or ['Vulkan0',...]."""
    try:
        out = subprocess.run([exe, "--list-devices"], capture_output=True,
                             text=True, timeout=120, cwd=os.path.dirname(exe))
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] --list-devices failed for {exe}: {exc}")
        return []
    text = out.stdout + out.stderr
    devs = re.findall(r"^\s*((?:CUDA|Vulkan|CPU)\d+):", text, re.MULTILINE)
    seen, ordered = set(), []
    for d in devs:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


class VramSampler(threading.Thread):
    """Sample nvidia-smi max memory.used (MiB) across all GPUs while a proc runs."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = {}  # gpu_index -> peak MiB
        self._stop_evt = threading.Event()

    def _sample(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[1].isdigit():
                    idx, used = int(parts[0]), int(parts[1])
                    self.peak[idx] = max(self.peak.get(idx, 0), used)
        except Exception:  # noqa: BLE001
            pass

    def run(self):
        while not self._stop_evt.is_set():
            self._sample()
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


def parse_bench(text: str, test: str) -> float | None:
    """Extract the t/s value (mean) for a given test column (pp512 / tg128)."""
    for line in text.splitlines():
        if test in line and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            try:
                i = cells.index(test)
            except ValueError:
                continue
            val = cells[i + 1] if i + 1 < len(cells) else ""
            m = re.match(r"([0-9]+(?:\.[0-9]+)?)", val)
            if m:
                return float(m.group(1))
    return None


def run_once(exe: str, device: str) -> tuple[float | None, float | None]:
    args = [exe, "-m", MODEL, "-p", str(N_PROMPT), "-n", str(N_GEN),
            "-r", "5", "-ngl", str(NGL), "-sm", "none", "-dev", device]
    proc = subprocess.Popen(args, cwd=os.path.dirname(exe),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    out, _ = proc.communicate(timeout=900)
    return parse_bench(out, f"pp{N_PROMPT}"), parse_bench(out, f"tg{N_GEN}")


def median(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="invocations per device (median)")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = os.path.join(RAW_DIR, f"single_gpu_raw_{stamp}.csv")

    header = ["timestamp_utc", "backend", "device", "model", "quant", "ngl",
              "n_prompt", "n_gen", "split_mode", "run", "pp_tps", "tg_tps",
              "peak_vram_mib", "exit_ok"]
    rows: list[dict] = []

    for backend, exe in BUILDS.items():
        if not os.path.exists(exe):
            print(f"[skip] {backend}: bench not found at {exe}")
            continue
        devices = [d for d in list_devices(exe) if d.startswith(backend)]
        print(f"\n=== {backend}: devices {devices} ===")
        for dev in devices:
            idx = int(re.sub(r"\D", "", dev))
            for r in range(1, args.reps + 1):
                sampler = VramSampler()
                sampler.start()
                t0 = time.time()
                try:
                    pp, tg = run_once(exe, dev)
                    ok = True
                except Exception as exc:  # noqa: BLE001
                    print(f"  [!] {backend}/{dev} run {r} failed: {exc}")
                    pp = tg = None
                    ok = False
                sampler.stop()
                dt = time.time() - t0
                peak = sampler.peak.get(idx)  # None for AMD (nvidia-smi can't see it)
                rows.append({
                    "timestamp_utc": stamp, "backend": backend, "device": dev,
                    "model": os.path.basename(MODEL), "quant": "Q4_K_M", "ngl": NGL,
                    "n_prompt": N_PROMPT, "n_gen": N_GEN, "split_mode": "none",
                    "run": r, "pp_tps": pp, "tg_tps": tg,
                    "peak_vram_mib": peak, "exit_ok": ok,
                })
                print(f"  {dev} run {r}: pp512={pp} tg128={tg} peakVRAM={peak}MiB ({dt:.1f}s)")

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    summary_path = os.path.join(RAW_DIR, f"single_gpu_summary_{stamp}.csv")
    summ_header = ["backend", "device", "pp512_tps_median", "tg128_tps_median",
                   "peak_vram_mib", "runs", "model", "quant", "ngl",
                   "n_prompt", "n_gen"]
    seen_keys = []
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summ_header)
        w.writeheader()
        for row in rows:
            key = (row["backend"], row["device"])
            if key in seen_keys:
                continue
            seen_keys.append(key)
            grp = [r for r in rows if (r["backend"], r["device"]) == key]
            w.writerow({
                "backend": row["backend"], "device": row["device"],
                "pp512_tps_median": median([g["pp_tps"] for g in grp]),
                "tg128_tps_median": median([g["tg_tps"] for g in grp]),
                "peak_vram_mib": next((g["peak_vram_mib"] for g in reversed(grp)
                                       if g["peak_vram_mib"]), None),
                "runs": len(grp), "model": os.path.basename(MODEL),
                "quant": "Q4_K_M", "ngl": NGL,
                "n_prompt": N_PROMPT, "n_gen": N_GEN,
            })

    print(f"\n[+] raw     -> {raw_path}")
    print(f"[+] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
