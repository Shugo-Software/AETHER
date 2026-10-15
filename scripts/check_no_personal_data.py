#!/usr/bin/env python3
"""Fail if personal / machine-identifying data would be published.

Scans the files git would actually commit (via `git ls-files`, falling back to a
walk that honours .gitignore) for the classes of data that were deliberately
removed from this repo:

  * overlay / private network addresses (Tailscale CGNAT, RFC1918), while
    allowing documented CIDR ranges such as Tailscale's own 100.64.0.0/10
  * OS user-profile paths (`C:\\Users\\<name>`), unless written as %USERPROFILE%
  * NVIDIA GPU UUIDs (unique per-device identifiers)
  * extra terms you never want published (`--forbid "acme"`)

Nothing machine-specific is baked in: add your own terms per run or in
check_files.txt (one term per line, '#' comments allowed).

Usage:
  python scripts/check_no_personal_data.py
  python scripts/check_no_personal_data.py --forbid myhandle --forbid <your-tailnet-ip>
Exit: 0 clean, 1 violations found (prints path:line:rule).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERMS_FILE = os.path.join(ROOT, "check_files.txt")
SKIP_SUFFIXES = (".gguf", ".safetensors", ".bin", ".png", ".jpg", ".zip")

RULES = [
    # A trailing /NN means a documented CIDR range (e.g. Tailscale's own
    # 100.64.0.0/10), not this machine's address — do not flag those.
    ("overlay_net", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])"
                               r"\.\d{1,3}\.\d{1,3}\b(?!\s*/\s*\d)"),
     "Tailscale/overlay IP address"),
    ("private_net", re.compile(r"\b(?:192\.168|10\.\d{1,3})\.\d{1,3}\.\d{1,3}\b"
                               r"(?!\s*/\s*\d)"
                               r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
                               r"(?!\s*/\s*\d)"),
     "private (RFC1918) IP address"),
    ("user_profile", re.compile(r"[A-Za-z]:\\+Users\\+[A-Za-z0-9._-]+"),
     "absolute OS user path — use %USERPROFILE% instead"),
    ("gpu_uuid", re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
     "NVIDIA GPU UUID (unique device identifier)"),
]


def tracked_files():
    """Files git would publish; falls back to a walk if git is unavailable."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            return [os.path.join(ROOT, p) for p in out.stdout.split("\n") if p]
        if out.returncode == 0:
            return []          # repo exists but nothing staged/committed yet
    except (OSError, subprocess.SubprocessError):
        pass
    skip = {".git", "venvs", "builds", "toolchains", "models", "exports",
            "adapters", "cache", "corpus", "logs", "__pycache__"}
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if not fn.endswith(SKIP_SUFFIXES):
                found.append(os.path.join(dirpath, fn))
    return found


def extra_terms(cli_terms):
    terms = list(cli_terms)
    if os.path.isfile(TERMS_FILE):
        with open(TERMS_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    return terms


def main():
    ap = argparse.ArgumentParser(description="personal-data guard for the repo")
    ap.add_argument("--forbid", action="append", default=[],
                    help="extra literal term that must not appear (repeatable)")
    args = ap.parse_args()

    files = tracked_files()
    print("[guard] scanning %d file(s) %s"
          % (len(files), "(git ls-files)" if files else "(filesystem walk)"))
    if not files:
        print("[guard] nothing to scan yet — commit the sources first")
        return 0

    terms = extra_terms(args.forbid)
    violations = []
    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    for name, rx, why in RULES:
                        if rx.search(line):
                            violations.append((rel, lineno, name, why))
                    for term in terms:
                        if term and term in line:
                            violations.append((rel, lineno, "forbidden-term",
                                               "matches %r" % term))
        except (OSError, UnicodeDecodeError):
            continue

    if violations:
        print("\n[guard] FAIL — %d violation(s):" % len(violations))
        for rel, lineno, name, why in violations:
            print("  %s:%d  [%s] %s" % (rel, lineno, name, why))
        print("\nFix the file (or extend .gitignore) before pushing.")
        return 1
    print("[guard] OK — no personal or machine-identifying data found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
