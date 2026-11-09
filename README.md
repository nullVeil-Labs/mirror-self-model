# mirror-self-model — MIRROR-00 and MIRROR-01

**Causal probing of self-prediction in language models.** Two small, pre-registered, exploratory studies.

When a language model predicts something about its own answer ("What would be the first letter of your answer?"),
does that prediction depend on an identifiable internal representation, and one that is specific to predicting *itself*
rather than another model? We extract a "self vs other" direction from the residual stream of Qwen3-4B-Instruct-2507,
remove it, and check whether self-prediction breaks selectively, against random-direction, second-person and paraphrase
controls.

These are zero-shot pilots inspired by, **not replications of**, Binder et al., *Looking Inward: Language Models Can
Learn About Themselves by Introspection* (2024; ICLR 2025, [arXiv:2410.13787](https://arxiv.org/abs/2410.13787)).
Nothing here concerns consciousness or subjective experience. The questions are narrow and mechanical.

## Status (2026-11-09): complete — negative result

| Study | Design | Verdict |
|---|---|---|
| **MIRROR-00** | Find the self−other direction on dev data; ablate it; regenerate answers and predictions; held-out test | **Refine** |
| **MIRROR-01** | Same frozen direction, ablated during prediction only; scored against *fixed* answers; 400 fresh items; 20 strength-matched random directions | **H3** (pre-registered rule), decided by one item out of 400 |

Following the rule agreed before MIRROR-01 was run, this direction-based line of work is **closed as a negative result**.

### Key findings
- Removing the direction (block 12, α = 2) lowers Qwen's self-prediction accuracy by about **9 points** on held-out
  items, at no cost to factual accuracy or perplexity. The generic "you vs she" direction does nothing comparable.
- **MIRROR-00:** the drop is not selective. Predictions about the other model (Gemma) also fall, and much of the
  effect comes from the intervention changing Qwen's *answers* while its predictions do not follow.
- **MIRROR-01:** with the answers held fixed, the 9-point drop remains, and only self-prediction falls (selectivity
  S_cas = +0.087 [0.045, 0.130]; strength-matched random directions damage all predictions alike). But one of the 20
  random directions lowered self-prediction by one item more (37 vs 36 of 400), so under the pre-registered rule the
  verdict is H3. The difference is a statistical tie, and the reports set out this tension openly rather than overriding the rule.
- Baseline caveat: Qwen largely predicts "the other model" by simulating itself (same prediction for both framings on
  64–83% of items), which limits how selective any effect can appear.

**Read first:** [`results/summary.md`](results/summary.md) (MIRROR-00) and
[`results/mirror01/mirror01_report.md`](results/mirror01/mirror01_report.md) (MIRROR-01).

## Method discipline
- Protocols were frozen before data: [`results/protocol.md`](results/protocol.md) (with dated amendments) and
  [`results/mirror01/protocol_01.md`](results/mirror01/protocol_01.md).
- The test data was sealed until the final stage. The layer and strength were written to
  `results/stage3_selection.json` before any intervention run, and never re-tuned.
- All raw generations, parsed values and metrics are included, so every number in the reports can be recomputed.

## Models
| Role | Model | Identity |
|---|---|---|
| A (the only hooked model) | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | revision `cdbee75f17c01a7cc42f958dc650907174af0554`; shard sha256 in `results/sha256_qwen.txt` |
| B (comparison, never hooked) | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | weights sha256 `cfbd3d2f…5503` = Hub revision `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` |

bf16, greedy decoding, thinking disabled, batch size 1, seed 0. Run on a single 24 GB GPU (peak about 16 GB).

## Reproduce
```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r results/requirements.lock.txt   # Python 3.13; transformers 5.18.0

hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir models/qwen3-4b-instruct-2507
hf download google/gemma-4-E4B-it --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c --local-dir models/gemma-4
```
If the weights live elsewhere, set `MIRROR_GEMMA4` (and so on) or create a gitignored `local_paths.json`.

Run from `scripts/`, in order (each script writes to `results/`):

| Step | Scripts |
|---|---|
| MIRROR-00 Stage 1 | `stage1_smoke.py` |
| MIRROR-00 Stage 2 | `make_dataset.py`, `stage2_format_pilot.py`, `stage2_baseline.py`, `stage2_metrics.py` |
| MIRROR-00 Stage 3 | `stage3_direction.py` |
| MIRROR-00 Stage 4 | `stage4_interventions.py` |
| MIRROR-00 Stage 5 | `stage5_test.py` |
| MIRROR-01 | `mirror01_make_dataset.py`, `mirror01_strength.py`, `mirror01_run.py`, `mirror01_score.py`, `mirror01_rescore_m00.py` (exploratory) |

Shared code: `common.py` (loading, hooks), `prompts.py`, `parsers.py` (with self-tests: `python parsers.py`), `interventions.py`.

## Layout
```
data/      dataset.csv (MIRROR-00), dataset_01.csv (MIRROR-01), pg1342.txt (Project Gutenberg, neutral text for perplexity)
scripts/   all code
results/   protocols, stage reports, raw CSVs, metrics, plots, summary.md; mirror01/ for the follow-up
models/    weights (gitignored)
```

## Limitations
Zero-shot; one hooked model; one property (first letter); small samples (190–400 items); random-direction nulls are
coarse (10 in MIRROR-00, 20 in MIRROR-01); greedy bf16 decoding. See the reports for details.

## AI assistance
The study was designed with an AI co-designer, and implemented, run and written up with Claude (Anthropic), under
human direction and review.

## Licence
Code: MIT (see `LICENSE`). `data/pg1342.txt` is a Project Gutenberg text and is public domain in the USA.
The model weights are not included; they are subject to their own licences (Qwen: Apache-2.0; Gemma 4: Apache-2.0).
