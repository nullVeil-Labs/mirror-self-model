# mirror-self-model — MIRROR-00

Causal Probing of Self-Prediction in Language Models (exploratory pilot).

**Status (2026-11-09): complete.**
- MIRROR-00 → **Refine** (`results/summary.md`).
- Follow-up MIRROR-01 (fixed targets, fresh held-out data) → **H3** under the pre-registered rule, decided by one
  item out of 400 (`results/mirror01/mirror01_report.md`). Per the pre-agreed rule, this direction-based line is closed
  as a negative result.
Frozen protocol and amendments: `results/protocol.md`.

## Layout
```
mirror-self-model/
  .venv/        Python 3.13, torch 2.11.0+cu128, transformers 5.18 (lock: results/requirements.lock.txt)
  models/       Model A weights (Qwen3-4B-Instruct-2507 @ cdbee75f)
  data/         pg1342.txt (neutral text for perplexity), generated datasets
  scripts/      common.py (loading, hooks), stage scripts
  results/      protocol, stage reports, CSVs, metrics, plots, summary.md
```

## Models
| Role | Model | Path | Identity |
|---|---|---|---|
| A (hooked) | Qwen/Qwen3-4B-Instruct-2507 | `models/qwen3-4b-instruct-2507/` | revision cdbee75f…0554, files checked against Hub sha256 |
| B (never hooked) | google/gemma-4-E4B-it | `models/gemma-4/` by default | sha256 cfbd3d2f…5503 = Hub main @ 4ecc2ba6 |
| B fallback | google/gemma-3-4b-it | `models/gemma-3-4b-it/` by default | local sha256 in `results/sha256_gemma_local.txt` |

If the Gemma weights live elsewhere, point to them with `local_paths.json` (gitignored) or the env vars
`MIRROR_GEMMA4` / `MIRROR_GEMMA3`. See `AGENTS.md` for the anonymity rules before publishing anything.

## Run
```
.venv\Scripts\python.exe scripts\stage1_smoke.py
```
