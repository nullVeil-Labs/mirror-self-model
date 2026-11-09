# Stage 3 report — candidate direction and (L, α) selection (dev split only), 2026-11-09

**Selected: L = 12, α = 2.0** (written to `stage3_selection.json` at 2026-11-09T11:26:09Z, before any Stage 4 run).
L and α are now frozen. The sweep is close to a null: the selected setting has S = +0.039 on dev-select with a
95% CI of [−0.053, +0.132], and no setting in the sweep has an S whose CI excludes 0.
No test-split item has been run through any model.

Files: `scripts/stage3_direction.py`, `scripts/interventions.py` (shared Stage 3–4 helpers), `scripts/common.py`
(`Ablate` extended, below); `results/stage3_split.json`, `stage3_directions.pt`, `stage3_cosines.csv`,
`stage3_sweep.csv`, `stage3_sweep_raw.csv`, `stage3_run.json`, `stage3_selection.json`, `stage3_log.txt`.

## 1. What was run
- **Items:** the 190 dev first_letter items (Amendment 3: only first_letter passed Gate 1). Split stratified by
  subcategory, seed 0: **dev-fit 114**, **dev-select 76** (each choice and factual subcategory 6/4; completion 30/20).
  Item ids in `stage3_split.json`.
- **Directions** (dev-fit only): last-input-token post-block residual of A (Qwen3-4B-Instruct-2507) at all 36 blocks
  for the S1, O1, S2 and O2 first_letter prediction prompts (exact Stage 2 prompts). d_L = mean(S1) − mean(O1),
  d_L′ = mean(S2) − mean(O2), each normalised per layer. d_you: 60 topics drawn with seed 0 from a fixed list of
  80 neutral topics; "What do you think about X?" minus "What does she think about X?", same procedure.
- **Sweep** (dev-select only): blocks 3, 6, …, 33 × α ∈ {1, 2} = 22 settings, hook h ← h − α (h·d_L) d_L on the
  output of block L at every position (prefill and cached decoding). Under each setting: A's object answers
  (dev-select + all 40 dev factual items), S1 and O1 predictions on dev-select, factual accuracy, neutral-text
  perplexity (Stage 2 procedure). Self scored against A's **intervened** object answers; cross against Gemma's
  Stage 2 dev answers. Greedy, cached, bf16, batch size 1.
- **Unhooked baseline on dev-select** was regenerated first: **252/252 outputs identical to Stage 2**
  (factual 0.950, perplexity 13.887 — also identical).

## 2. Cosines (dev-fit)
| Block | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cos(d_L, d_L′) | 0.845 | 0.593 | 0.482 | **0.662** | 0.709 | 0.620 | 0.661 | 0.674 | 0.649 | 0.597 | 0.573 |
| cos(d_you, d_L) | 0.452 | 0.215 | 0.240 | **0.134** | 0.054 | 0.077 | 0.055 | 0.080 | 0.081 | 0.066 | 0.084 |

All 36 blocks are in `stage3_cosines.csv` (cos(d_L, d_L′) ranges 0.39–0.85; cos(d_you, d_L) −0.03 to 0.45, highest
in blocks 0–5). The self-vs-other contrast is moderately stable across the two wordings (~0.6–0.7 in mid and late
blocks) and nearly orthogonal to the generic second-person direction beyond block 12. With 114 items and
2,560 dimensions, part of each mean difference is noise, so these cosines are lower bounds on the shared signal
and not evidence of a "self" representation by themselves. The raw (unnormalised) d_L norm is 4.4 at block 12
against a mean last-token residual norm of 33 (~13%).

## 3. Selection sweep (dev-select, n = 76 eligible at every setting)
Baseline on dev-select: self 0.447, cross 0.342, factual 0.950 (all 40 dev factual items), perplexity 13.887.
Drops are baseline − intervened (positive = worse). S = self drop − cross drop, paired bootstrap 95% CI.

| L | α | Self | Self drop | Cross drop | **S** [95% CI] | Cross-as-self drop | Factual | PPL (rise) | Answer change | \|h·d\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 0.447 | 0.000 | 0.013 | −0.013 [−0.09, 0.07] | 0.039 | 0.950 | 13.91 (+0.2%) | 0.17 | 1.1 |
| 3 | 2 | 0.434 | 0.013 | 0.013 | 0.000 [−0.08, 0.08] | 0.053 | 0.950 | 14.00 (+0.8%) | 0.21 | 1.1 |
| 6 | 1 | 0.526 | −0.079 | 0.000 | −0.079 [−0.18, 0.03] | −0.066 | 0.950 | 14.00 (+0.8%) | 0.18 | 6.8 |
| 6 | 2 | 0.447 | 0.000 | 0.039 | −0.039 [−0.14, 0.07] | −0.013 | 0.950 | 14.48 (+4.3%) | 0.26 | 6.8 |
| 9 | 1 | 0.500 | −0.053 | 0.013 | −0.066 [−0.16, 0.01] | 0.039 | 0.950 | 14.04 (+1.1%) | 0.17 | 6.2 |
| 9 | 2 | 0.513 | −0.066 | 0.026 | −0.092 [−0.20, 0.00] | −0.026 | 0.950 | 14.43 (+3.9%) | 0.34 | 6.2 |
| 12 | 1 | 0.421 | 0.026 | 0.039 | −0.013 [−0.11, 0.07] | 0.053 | 0.950 | 13.90 (+0.1%) | 0.11 | 6.1 |
| **12** | **2** | **0.382** | **0.066** | **0.026** | **+0.039 [−0.05, 0.13]** | 0.079 | 0.950 | 13.98 (+0.6%) | 0.28 | 6.1 |
| 15 | 1 | 0.447 | 0.000 | 0.000 | 0.000 [−0.07, 0.07] | 0.039 | 0.950 | 13.91 (+0.2%) | 0.14 | 6.7 |
| 15 | 2 | 0.447 | 0.000 | 0.000 | 0.000 [−0.09, 0.09] | 0.039 | 0.950 | 14.09 (+1.5%) | 0.22 | 6.7 |
| 18 | 1 | 0.461 | −0.013 | 0.026 | −0.039 [−0.14, 0.07] | 0.053 | 0.950 | 14.31 (+3.0%) | 0.29 | 10.5 |
| 18 | 2 | 0.461 | −0.013 | 0.053 | −0.066 [−0.18, 0.05] | 0.013 | 0.950 | 15.19 (+9.4%) | 0.45 | 10.4 |
| 21 | 1 | 0.368 | 0.079 | 0.079 | 0.000 [−0.11, 0.11] | 0.158 | 0.950 | 14.35 (+3.3%) | 0.38 | 17.9 |
| 21 | 2 | 0.289 | 0.158 | 0.171 | −0.013 [−0.16, 0.13] | 0.263 | 0.925 | 15.74 (**+13.4%**) | 0.62 | 17.9 |
| 24 | 1 | 0.434 | 0.013 | 0.053 | −0.039 [−0.12, 0.04] | 0.066 | 0.950 | 14.27 (+2.8%) | 0.26 | 18.7 |
| 24 | 2 | 0.368 | 0.079 | 0.105 | −0.026 [−0.17, 0.12] | 0.224 | 0.950 | 15.27 (+10.0%) | 0.49 | 18.7 |
| 27 | 1 | 0.461 | −0.013 | 0.026 | −0.039 [−0.13, 0.04] | 0.079 | 0.950 | 14.35 (+3.3%) | 0.22 | 25.3 |
| 27 | 2 | 0.368 | 0.079 | 0.039 | +0.039 [−0.09, 0.17] | 0.132 | 0.975 | 15.45 (**+11.2%**) | 0.45 | 25.3 |
| 30 | 1 | 0.447 | 0.000 | 0.013 | −0.013 [−0.05, 0.03] | 0.053 | 0.950 | 13.85 (−0.3%) | 0.01 | 14.7 |
| 30 | 2 | 0.461 | −0.013 | 0.039 | −0.053 [−0.14, 0.04] | 0.066 | 0.950 | 13.84 (−0.4%) | 0.09 | 14.7 |
| 33 | 1 | 0.447 | 0.000 | 0.013 | −0.013 [−0.07, 0.04] | 0.039 | 0.950 | 13.90 (+0.1%) | 0.03 | 17.2 |
| 33 | 2 | 0.421 | 0.026 | 0.013 | +0.013 [−0.07, 0.09] | 0.079 | 0.950 | 13.94 (+0.4%) | 0.11 | 17.2 |

(\|h·d\| = mean |h·d_L| at the hooked block over all positions processed during generation, before the edit; the
amount removed per position is α × that. "Answer change" = share of dev-select object answers whose raw text differs
from baseline. Factual drop ≤ 2.5 points everywhere; validity of self and cross predictions 0.97–1.00 everywhere.)

**Selection rule applied:** 20 of 22 settings met both constraints (L21 α2 and L27 α2 fail the perplexity limit).
Among them, the highest S is L12 α2 (+0.039). L27 α2 has the same S but fails perplexity; no tie-break was needed.
The tie-break (higher S, then lower α, then lower layer) and the violation measure for the no-feasible-setting case
were fixed in the script before it ran.

## 4. What this means (dev-select, before Stage 4)
- **The sweep is essentially null.** Across 22 settings S ranges from −0.092 to +0.039 and every CI includes 0.
  The maximum of 20 noisy estimates on 76 items is expected to be positive by chance, so the selected S = +0.039 is
  itself no evidence of an effect; Stage 4 (all 190 dev items, random-direction controls) is the real check, and
  Stage 5 test the confirmatory one.
- **Where the self drop is largest (L21, L24, L27 at α = 2), cross drops by as much or more**, and cross-as-self drops
  even more: removing d_L there disrupts prediction in general (and the object answers: 45–62% change), not
  self-prediction specifically.
- At the selected L12 α2, 28% of object answers change (20% of first letters), yet factual accuracy and perplexity
  are unchanged. Greedy answers on open-ended choice items are fragile (Stage 1 already saw a random direction change
  an answer), so the random directions in Stage 4 are needed to interpret the change rate.
- Some settings *improve* self-prediction (L6 α1: +0.079, L9 α2: +0.066). This is within noise.

## 5. Engineering decisions
- **`Ablate` extended** (`scripts/common.py`): optional `alpha` (default 1.0). For α = 1 the arithmetic is literally
  the original expression (h − (h·d) d); α ≠ 1 uses h − α (h·d) d; α = 0 is a measure-only hook that returns the
  output untouched. The hook accumulates |h·d| (float32 projection, before the edit) over every processed position;
  `mean_abs_proj()` reports the mean. Stage 1–2 behaviour is unchanged.
- "Last input token" = the last token of the chat-formatted prompt (the newline after the assistant header), i.e.
  the position whose output is the first generated token; captured with `common.Capture` in a single forward pass.
- Block indices are 0-based outputs of `model.model.layers[i]`; "3, 6, …, n−3" with n = 36 gives 3–33.
- dev-fit / dev-select split stratified by subcategory (the same strata as the Stage 2 dev/test split); 60% rounded.
- Factual accuracy in the sweep uses all 40 dev factual items (16 are in dev-select, 24 in dev-fit); for dev-fit items
  only the object answer is generated. Perplexity: the unchanged Stage 2 function (40 × 256-token windows, batch 4).
- Eligibility: an item counts when A's object answer has a valid first letter both at baseline and under the
  intervention and Gemma's target is valid (76/76 at every setting).
- Violation measure for the fallback (unused): max(0, factual drop − 5 pts)/5 pts + max(0, PPL rise − 10%)/10%.
- d_you prompts carry no format line and no task; they are plain chat turns.

## 6. Caveats
- dev-select has 76 items: an S difference of one item is 0.013, and CIs are ±0.07–0.15. Layer selection is noisy
  (Amendment 3, item 4).
- d_L is fitted on prompts that differ in wording ("you … your" vs "a different AI model called Gemma … its"), so
  d_L mixes self-reference with lexical and name differences; d_you and d_L′ are the controls for that.
- One model, one seed, greedy decoding, first_letter only.
