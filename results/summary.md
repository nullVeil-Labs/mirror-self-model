# MIRROR-00 — summary of an exploratory pilot (2026-11-09)

**Recommendation: Refine.**

This is a zero-shot pilot inspired by, but not a replication of, Binder et al. (2024; ICLR 2025, arXiv:2410.13787). Nothing here concerns subjective experience or consciousness. The question is narrow and mechanical.

## The question
When a language model predicts a property of its own answer ("What would be the first letter of your answer?"), does it rely on internal information that it does not use when predicting another model's answer? And can we find a direction in its activations whose removal harms self-prediction specifically?

## What was done
- **Models.** A = Qwen3-4B-Instruct-2507, the only model whose activations were edited. B = Gemma-4-E4B-it, used only as the "other" model whose answers A tries to predict. bf16, greedy decoding, no fine-tuning (zero-shot).
- **Data.** 500 generated prompts (open choices, numbers, phrase completions, factual questions), split 50/50 into dev and test before anything was run. Three properties were planned (first letter, word-length bucket, parity). Only **first letter** passed the pre-registered Gate 1 on dev (self-prediction clearly above the majority-class baseline), so everything after that uses first letter only: 190 items per split, 150 of them non-factual.
- **Direction and edit.** On 60% of dev, d_L = mean activation for "you" prediction prompts minus the mean for "Gemma" prompts, at each layer. On the other 40% of dev, one layer and strength were chosen (block 12, α = 2, i.e. the component along d_L is reflected), subject to limits on factual accuracy and perplexity.
- **Controls.** 10 random directions; d_you, a generic "you vs she" direction from unrelated prompts; a second wording of the prediction prompts (S2/O2) and a direction fitted on it (d_L′); factual accuracy; perplexity on a neutral text; the size of the projection removed.
- **Main measure.** S = (drop in self-prediction accuracy) − (drop in cross-prediction accuracy), with a paired bootstrap 95% CI. Self is scored against A's own (possibly changed) answer; cross against Gemma's answer.
- **Dev/test discipline.** Protocol frozen before data; layer and strength frozen in a file before Stage 4; test run once, at the end, with nothing re-fitted.
- **Amendments** (dated, in `protocol.md`): (1) before any data, a co-designer's review added the paraphrase tests, d_you as a full condition and the S measure; (2) after the smoke test, strict format lines, because A ignored "one word" instructions; (3) after the dev baselines, first letter only (Gate 1), a secondary analysis without factual items (whose first letter follows from the question), a "cross-as-self" diagnostic, and batch size 1 (batched decoding was not output-identical).

## Key results (first letter; d_L removed at block 12, α = 2)
| | Dev, all (190) | **Test, all (190)** | Dev, non-factual (150) | Test, non-factual (150) |
|---|---|---|---|---|
| Baseline self / cross accuracy | 0.442 / 0.326 | **0.447 / 0.321** | 0.333 / 0.200 | 0.327 / 0.193 |
| Self drop under d_L [95% CI] | +0.105 [0.047, 0.163] | **+0.089 [0.032, 0.147]** | +0.140 [0.073, 0.213] | +0.100 [0.027, 0.173] |
| Cross drop | +0.047 | **+0.058** | +0.033 | +0.053 |
| S [95% CI] | +0.058 [−0.005, 0.121] | **+0.032 [−0.042, 0.105]** | +0.107 [0.033, 0.180] | +0.047 [−0.040, 0.133] |
| Random directions: self drop range | −0.032 to +0.026 | **−0.016 to +0.032** | −0.033 to +0.033 | −0.020 to +0.040 |
| Random directions: S range | −0.063 to +0.032 | **−0.042 to +0.026** | −0.073 to +0.047 | −0.047 to +0.047 |
| d_you self drop | +0.021 | **−0.016** | +0.020 | −0.020 |
| Paraphrase (S2/O2): self drop / S | +0.074 / +0.068 | **+0.111 / +0.105 [0.053, 0.158]** | +0.113 / +0.107 | +0.133 / +0.127 |
| Cross-as-self drop (S1/O1) | +0.105 | **+0.095** | +0.107 | +0.100 |
| Factual accuracy; perplexity rise | 0.950 → 0.950; +0.6% | **0.950 → 0.950; +0.6%** | (same) | (same) |

Decision rule on the primary test results: (a) self drop CI excludes 0 — **pass**; (b) larger than all 10 random drops — **pass**; (c) S CI excludes 0 and S above all random S — **fail** (CI includes 0); (d) factual and perplexity limits — **pass**; (e) d_you drop under half of d_L's — **pass**; not fragile across wordings. Details: `stage4_report.md` (dev) and `stage5_report.md` (test).

## What this shows
- Removing d_L at block 12 lowers how often A's first-letter predictions match A's own answers, by about 9–10 points on held-out items. This is more than any of 10 random directions (including one with a larger projection), more than the generic second-person direction, and costs nothing measurable in factual accuracy or perplexity. It replicated from dev to test.

## What it does not show
- **No selective impairment of self-prediction.** A's predictions about Gemma also get worse, and S is not distinguishable from 0 on test.
- **A predicts "Gemma" by simulating itself.** Already at baseline (Stage 2, dev), A gives the same first-letter prediction for itself and for Gemma on 74% of items with the first wording and 83% with the second, and its Gemma predictions match its own answers better than Gemma's. The baseline self − cross gap is therefore what any model would show that predicts "what I would say" in both frames.
- **Cross-as-self falls as much as self.** Scoring A's Gemma-predictions against A's own answers, the drop under d_L equals the self drop with the main wording (dev 0.105 vs 0.105; test 0.095 vs 0.089). The direction affects the match between A's answers and A's predictions in *either* frame, not something specific to "you". With the second wording the picture is less clear-cut: on test, cross-as-self drops 0.063 against a self drop of 0.111 (non-factual: 0.080 vs 0.133). This was not pre-specified and has no CI here; it is a hint worth checking, not a finding.
- **Scoring asymmetry.** The self target moves with the intervention (A's answers are regenerated under the edit); the cross target is Gemma's fixed answers. Most of the effect sits on items whose answer changed (d_L changes about a fifth of A's first letters, and largely stops A echoing the phrase in completion items), while A's predictions do not follow. This inflates the self drop relative to the cross drop, and it also applies to the paraphrase result.
- Nothing here bears on self-awareness, introspection in a rich sense, or experience.

## Limitations
Zero-shot (no training to self-predict, unlike Binder et al.); one hooked model and one comparison model; small n (190 / 150 items, one item ≈ 0.5–0.7 points; layer selection on 76 items was itself close to null); completion answers that echo the prompt are scored on the echoed word; bf16 greedy decoding (cached and uncached paths can differ on long outputs); a single property (first letter) — the two others failed Gate 1; ten random directions give only a coarse null. Two post-hoc counts of completion echoes differ slightly between the Stage 4 and Stage 5 reports (see `stage5_report.md`).

## Next steps (brief)
1. Score with a fixed self target as well (A's unedited answers) and edit only the prediction prompts, to separate "the answer moved" from "the prediction lost information".
2. Fix the completion echo in the object prompts, or drop completion items.
3. Use a cross model whose answers A does not share, or a fine-tuned self-prediction setting as in Binder et al., so that self and cross are not the same prediction scored against two targets.
4. More items and more random directions; a second hooked model; a pre-registered test of the wording-2 cross-as-self gap.

## Recommendation: Refine
The decision rule, applied mechanically to the primary test results, gives **Refine**: (a) and (b) hold — d_L's self-prediction drop is real on held-out items and larger than every random direction's — but (c) fails, because the selective measure S has a 95% CI that includes 0; (d), (e) and the paraphrase check pass. The diagnostics point further towards a negative reading than the rule's verdict does: with the main wording, cross-as-self falls as much as self, and much of the drop comes from d_L changing A's answers under a scoring rule whose self target moves while the cross target stays fixed. So the tension is plain: the rule says the effect is worth refining; the diagnostics say that, in its current form, the effect is not shown to be about self-prediction specifically. We keep the rule's verdict, and read "Refine" narrowly: the design must be changed (steps 1–3 above) before any self-specific claim could be tested, and continuing with the present design would not be justified.
