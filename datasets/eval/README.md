# `datasets/eval/` — measurement sets, never training data

These datasets exist to **measure** the students, not to teach them:

| File | Source | Why it is not training data |
|---|---|---|
| `openai-math.jsonl` | `FineEnvs/openai-math` (apache-2.0) | The source is Lean4/Harbor **RL environment** scaffolding (Dockerfile, `Challenge.lean`, `grade.py`), not prompt→answer pairs. It is used to score *approach/explanation quality* of a student that cannot possibly prove research-level theorems. |

The training control plane refuses anything under this directory: datasets are
launched by bare filename from `datasets/`, so `eval/openai-math.jsonl` cannot be
selected from the dashboard or the API.

Ingest (re)creates these files plus their manifests:

```powershell
python scripts\ingest_hf_dataset.py --source openai-math
```
