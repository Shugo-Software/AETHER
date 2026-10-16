#!/usr/bin/env python3
"""AETHER process environment: every cache and temp path belongs on the repo drive.

C: is critically small and volatile on this rig (docs/plan.md), and a default
cache there has already cost real work: a parquet row count materialised into
`C:\\Users\\<user>\\.cache\\huggingface\\datasets` and died with
`OSError: [Errno 28] No space left on device` — after a 289 GB download had
completed. Nothing in this project should write to C: at all.

`pin_caches()` sets, in the current process (so every child inherits them):

    HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE / HF_XET_CACHE -> <root>/cache/hf
    TORCH_HOME / TORCHINDUCTOR_CACHE_DIR                      -> <root>/cache/torch
    PIP_CACHE_DIR                                             -> <root>/cache/pip
    TEMP / TMP / TMPDIR                                       -> <root>/cache/temp

Entry points call it at startup; `--selftest` proves that a child process
launched the way the pipeline launches them resolves nothing to C:.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")

PATHS = {
    "HF_HOME": os.path.join(CACHE, "hf"),
    "HF_HUB_CACHE": os.path.join(CACHE, "hf"),
    "HF_DATASETS_CACHE": os.path.join(CACHE, "hf"),
    "HF_XET_CACHE": os.path.join(CACHE, "hf", "xet"),
    "TORCH_HOME": os.path.join(CACHE, "torch"),
    "TORCHINDUCTOR_CACHE_DIR": os.path.join(CACHE, "torch", "inductor"),
    "PIP_CACHE_DIR": os.path.join(CACHE, "pip"),
    "TEMP": os.path.join(CACHE, "temp"),
    "TMP": os.path.join(CACHE, "temp"),
    "TMPDIR": os.path.join(CACHE, "temp"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "PYTHONIOENCODING": "utf-8",
}

# The dirs worth creating up front (a file value like the telemetry flag is not).
DIRS = ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "TEMP")


def cache_env(base=None, create=True):
    """A copy of `base` (default: this process's environment), caches on G:."""
    env = dict(os.environ if base is None else base)
    env.update(PATHS)
    if create:
        for key in DIRS:
            try:
                os.makedirs(env[key], exist_ok=True)
            except OSError:
                pass
    return env


def pin_caches():
    """Apply the G: paths to this process, so children launched later inherit them."""
    env = cache_env()
    os.environ.update({k: env[k] for k in PATHS})
    return env


def off_drive():
    """Any of our cache vars still pointing at a drive other than the repo's.

    `pin_caches()` makes this empty; it exists so a process can *check* rather
    than assume (and so the selftest can assert it).
    """
    wanted = os.path.splitdrive(ROOT)[0].upper()
    bad = {}
    for key in PATHS:
        val = os.environ.get(key)
        if val and os.path.splitdrive(val)[0].upper() not in ("", wanted):
            bad[key] = val
    return bad


_CHILD = ("import json, tempfile\n"
          "import huggingface_hub.constants as c\n"
          "import datasets\n"
          "print(json.dumps({'TEMP': str(tempfile.gettempdir()),\n"
          "                  'HF_HOME': str(c.HF_HOME),\n"
          "                  'HF_HUB_CACHE': str(c.HF_HUB_CACHE),\n"
          "                  'HF_XET_CACHE': str(getattr(c, 'HF_XET_CACHE', '')),\n"
          "                  'HF_DATASETS_CACHE':\n"
          "                      str(datasets.config.HF_DATASETS_CACHE)}))\n")


def selftest():
    """Offline checks that nothing resolves to a foreign drive.

    Runs the real libraries in a child process, because that is where the
    resolution happens — the environment we hand down is the thing under test.
    """
    ok = True
    drive = os.path.splitdrive(ROOT)[0].upper()

    def check(good, label):
        nonlocal ok
        ok = ok and bool(good)
        print("[selftest] %s %s" % ("ok  " if good else "FAIL", label))

    env = cache_env()
    out = None
    try:
        out = subprocess.run([sys.executable, "-c", _CHILD], env=env,
                             capture_output=True, text=True, timeout=180)
        seen = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout else {}
    except Exception as exc:  # noqa: BLE001 — report, do not raise
        seen = {}
        print("[selftest]   child failed: %s" % exc)
    if out is not None and not seen:
        print("[selftest]   rc=%s stderr=%s"
              % (out.returncode, (out.stderr or "")[-400:]))
    check(seen, "a child process reports its caches (%d values)" % len(seen))
    for key, val in sorted(seen.items()):
        check(os.path.splitdrive(val)[0].upper() in ("", drive),
              "%s -> %s" % (key, val))
    pin_caches()          # exactly what every entry point does at startup
    check(not off_drive(),
          "after pin_caches() this process resolves every cache to %s" % drive)
    check(all(os.path.splitdrive(v)[0].upper() in ("", drive)
              for v in PATHS.values()),
          "every path in PATHS is on %s" % drive)
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print("ROOT  : %s" % ROOT)
    for key in sorted(PATHS):
        print("  %-24s %s" % (key, PATHS[key]))
    bad = off_drive()
    print("off-drive right now  : %s" % (bad or "nothing (C: is not used)"))
