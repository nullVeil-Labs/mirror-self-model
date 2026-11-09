# Stage 2 report — dataset and behavioural baselines (dev split only), 2026-11-09

**Gate 1 verdict: `first_letter` passes; `length_bucket` and `parity` fail.** Only `first_letter` goes on to Stage 3.
No test-split item has been run through any model. Nothing from Stage 3 (directions, ablations) has been run.

Files: `data/dataset.csv`; `scripts/make_dataset.py`, `prompts.py`, `parsers.py`, `stage2_format_pilot.py`,
`stage2_baseline.py`, `stage2_metrics.py`; `results/stage2_format_pilot.csv` (+ `_raw.csv`, `_log.txt`),
`results/stage2_dev_raw.csv`, `results/stage2_dev_run.json`, `results/stage2_dev_metrics.json`,
`results/stage2_baseline_log.txt`. Every number below comes from these files.

## 1. What was run
- **Dataset** (`make_dataset.py`, seed 0, deterministic: two runs give byte-identical files). 500 items:
  choice 200 (10 subcategories × 20, 3 templates × 10 qualifiers each), numeric 120 (4 templates, random ranges),
  short completion 100 (3 templates, determiner + adjective + noun stems), factual 80 (capitals, chemical symbols,
  arithmetic with the answer written as a word, opposites; gold answers and accepted aliases). Split 50/50 stratified
  by subcategory (hence by category): dev = choice 100, numeric 60, completion 50, factual 40.
- **Parsers** (`parsers.py`) as in the protocol; `python parsers.py` runs the self-test (markdown bold, quotes,
  trailing full stop, "The answer is X", lowercase letters, out-of-range numbers, multi-word answers): all pass.
- **Format pilot** (Amendment 2) on 20 dev items with A (section 3).
- **Baselines on dev** (`stage2_baseline.py`): A object answers; A predictions S1, O1, S2, O2 for every applicable
  property; B object answers; B self-predictions (S1, exploratory); A factual accuracy; A perplexity. Greedy, cached,
  bf16, no system prompt, thinking off, `max_new_tokens` 16 (object) / 8 (predictions). 2,500 generations in total.
- **Metrics** (`stage2_metrics.py`): 10,000 bootstrap resamples, seed 0, percentile 95% CIs; differences paired
  over items. Invalid object answer → item ineligible for that property; invalid prediction → wrong.

## 2. Main results (dev)
Self = A predicts A (target: A's property). Cross = A predicts B (target: B's property). Majority and
frequency-matched random baselines are computed on A's targets (B's in brackets).

| Property | n | Self S1 [95% CI] | Cross O1 [95% CI] | Self − cross (paired) [95% CI] | Majority A (B) | Freq.-random A (B) | Validity S1 / O1 |
|---|---|---|---|---|---|---|---|
| first_letter | 190 | **0.442** [0.374, 0.511] | 0.326 [0.263, 0.395] | +0.116 [0.053, 0.179] | 0.095 (0.126) | 0.065 (0.071) | 0.984 / 0.995 |
| length_bucket | 150 | 0.393 [0.313, 0.473] | 0.427 [0.347, 0.507] | −0.033 [−0.127, 0.060] | 0.587 (0.647) | 0.452 (0.486) | 0.933 / 0.987 |
| parity | 60 | 0.567 [0.433, 0.683] | 0.433 [0.317, 0.550] | +0.133 [−0.050, 0.317] | 0.667 (0.667) | 0.556 (0.556) | 1.000 / 1.000 |

Alternative wording (Amendment 1):

| Property | Self S2 [95% CI] | Cross O2 [95% CI] | Self − cross [95% CI] | Validity S2 / O2 |
|---|---|---|---|---|
| first_letter | 0.411 [0.342, 0.479] | 0.326 [0.258, 0.395] | +0.084 [0.026, 0.147] | 0.984 / 0.989 |
| length_bucket | 0.440 [0.360, 0.520] | 0.433 [0.353, 0.513] | +0.007 [−0.087, 0.100] | 0.993 / 1.000 |
| parity | 0.550 [0.417, 0.683] | 0.367 [0.250, 0.483] | +0.183 [0.000, 0.367] | 1.000 / 1.000 |

B predicting B (S1, exploratory): first_letter 0.521 [0.447, 0.589]; length_bucket 0.493 [0.413, 0.573];
parity 0.633 [0.517, 0.750] (B said "odd" on 58 of 60 items, so this is close to a constant guess; majority 0.667).
B's prediction validity: 0.995, 1.000, 1.000.

Object answers: all A and B object answers were valid for every property (object validity 1.000 for both).
Noncompliant (multi-word) answers: A 11.6% of first_letter items and 14.7% of length_bucket items; B 0%.

Class distributions (A's targets): first_letter has 22 letters, the largest class B with 18/190; length_bucket
medium 88, short 47, long 15; parity odd 40, even 20 (B's parity targets have the same 40/20 split).

### A and B agreement (self-simulation check)
| Property | A and B give the same property value | A's S1 and O1 predictions identical | O1 scored against A's own target | S1 scored against B's target |
|---|---|---|---|---|
| first_letter | 0.558 | 0.747 | 0.389 | 0.389 |
| length_bucket | 0.660 | 0.847 | 0.387 | 0.393 |
| parity | 0.567 | 0.733 | 0.467 | 0.467 |

**Flag:** A and B agree on 56–66% of property values, and A gives the same prediction for itself and for "Gemma"
on 73–85% of items. A's cross-predictions match A's own answers (0.389 for first_letter) more often than they match
B's (0.326). So A's predictions about Gemma look largely like predictions about itself, and cross accuracy is
mostly earned where B happens to agree with A. The self − cross gap for first_letter is therefore what one would
expect from a model that predicts the same thing in both frames and is scored against two different targets; it
is not, on its own, evidence of privileged self-access.

### Factual accuracy and perplexity
- A factual accuracy (exact gold/alias match, dev, n = 40): **0.950** [0.875, 1.000]. By subcategory: capitals 1.0,
  chemical symbols 1.0, opposites 1.0, arithmetic 0.8 (5 + 9 → "fifteen"; 12 + 7 → "Eighteen"). B: 1.000.
- A perplexity on the neutral text (40 windows × 256 tokens from the first line of Chapter I; 10,200 scored tokens):
  mean NLL 2.631, **perplexity 13.887**.

## 3. Format pilot (Amendment 2)
20 dev items (5 per category: 5 choice subcategories, 5 numeric, 5 completion, one per factual subcategory + 1),
model A only, three wordings of the format line. Selection rule written in the script before it ran: highest
minimum validity over {object, S1, O1}, then highest mean, then the earlier wording.

| Wording | Object validity | S1 validity | O1 validity | Noncompliant object answers |
|---|---|---|---|---|
| **W1** (Amendment 2 examples, e.g. "Reply with only the single letter and nothing else.") | 20/20 | 30/30 | 30/30 | 3/20 |
| W2 ("Your whole reply must be that one letter: no words, no explanation, no punctuation.") | 20/20 | 30/30 | 30/30 | 2/20 |
| W3 ("Output format: <letter>. Write nothing else.") | 20/20 | 18/30 | 15/30 | 5/20 |

W1 and W2 tie on validity (1.0); W1 was fixed by the tie-break. W3 failed because A copied the template literally
("<o>", "<e>"). The full W1 lines are in `scripts/prompts.py`; they are now fixed for all conditions and both splits.
Validity in the full dev run stayed high (0.93–1.00), but see the completion caveat below: validity does not
capture noncompliance, and the selection rule did not weigh it.

## 4. Gate 1 (protocol: S1 self-accuracy 95% lower bound > majority baseline, ≥ 40 eligible items)
| Property | n eligible | Self S1 | CI lower bound | Majority baseline | Verdict |
|---|---|---|---|---|---|
| first_letter | 190 | 0.442 | 0.374 | 0.095 | **Pass** |
| length_bucket | 150 | 0.393 | 0.313 | 0.587 | **Fail** (below the majority baseline and below frequency-matched random, 0.452) |
| parity | 60 | 0.567 | 0.433 | 0.667 | **Fail** (point estimate below the majority baseline) |

Exploratory sensitivity checks for first_letter (not part of the gate; same bootstrap):
- choice items only (n = 100): self 0.370 [0.28, 0.46], majority 0.130, cross 0.260;
- all non-factual items (n = 150): self 0.333 [0.26, 0.41], majority 0.113, cross 0.200;
- compliant (single-word) answers only (n = 168): self 0.435 [0.363, 0.512], majority 0.107;
- completion items only (n = 50): self 0.260 [0.14, 0.38], majority 0.200 — does not clear its own baseline.
The pass is not only a factual-item effect, but factual items (self 0.85, cross 0.80; A and B agree on 95%) raise
the pooled number a lot: there, the first letter follows from the question for any competent model.

## 5. Illustrative raw examples (dev; A = Qwen, B = Gemma)
| Item | A answer | A S1 | A O1 | B answer | B S1 | Comment |
|---|---|---|---|---|---|---|
| fac424 capital of Spain (first_letter) | Madrid | M | M | Madrid | M | Correct everywhere; also trivially predictable. |
| cho008 animal (first_letter) | dog | d | d | Dog | D | Self correct (lowercase letter parsed as D). |
| cho013 animal (first_letter) | Lion | z | z | Lion | L | A predicts "z" for both itself and Gemma; B gets itself right. |
| cho192 plant grown at home (first_letter) | rose | g | p | Basil | P | Self wrong; cross right — A's two frames differ here. |
| com320 'Every sudden window ...' | Every sudden window \*breaks\*. | E | e | shatters | S | A echoes the phrase (noncompliant, scored on "Every"); its self-prediction "E" matches that echo. |
| com377 'One golden letter ...' | One golden letter \*\*E\*\*. | O | O | alphabet | G | Echo plus an odd completion; target "O". |
| cho001 animal (length_bucket) | eagle | bird | short | Eagle | short | Invalid self-prediction: A answered a word, not a bucket (counted wrong). |
| num208 integer (parity) | 42 | odd | even | 42 | odd | A picks 42 but predicts "odd" for itself. A's numbers cluster (73 on 12 of 60 items). |
| fac469 5 plus 9 (factual) | fifteen | F | s | Fourteen | F | A's factual error; its self-prediction still matches its own wrong answer. |

## 6. Batched vs unbatched generation
- A: the pre-check (10 dev items, one left-padded batch vs one-by-one) found 2/20 mismatches (S1 predictions:
  "p" vs "g", "a" vs "j") → **A ran at batch size 1**.
- B: the pre-check matched 20/20, but a post-check on 10 random object answers from the full batched run found
  1 mismatch ("42" vs "41") → all B outputs were **discarded and regenerated at batch size 1**. The discarded batched
  B run differed from the batch-size-1 run on 13 object-answer rows and 7 prediction rows (of 400).
- The whole script was then re-run once from scratch: A's 1,850 outputs were identical to the first run (repeat
  determinism). All reported results come from batch size 1, i.e. the same path as the single-prompt `generate`.
- Wall-clock (final run): 216.9 s in total; A 116.0 s for 1,850 generations, B 71.8 s for 650; peak VRAM 10.0 GB (A),
  16.6 GB (B). Format pilot: 38 s.

## 7. Surprises and problems
1. **A often echoes completion phrases.** 40% of A's completion answers (20/50) repeat the phrase, sometimes with the
   completion in markdown ("Every sudden window \*breaks\*."). The protocol scores these on the first word, so the
   target becomes the echoed determiner (A/The/One/Every). This is per protocol, not a parser bug, but for these
   items first_letter and length_bucket measure the echo, not the chosen word. The format pilot only saw 2 such
   cases (validity counts them as valid) and the 20-item budget is spent, so the wording was not changed. A's
   self-predictions sometimes track the echo (com320 above). Flagged for the co-designer; no protocol change made.
2. **length_bucket self-predictions are biased to "short"** (74 of 150 S1 predictions; true targets are mostly
   "medium", 88/150), e.g. "short" for Paris, Tennis, Skiing. 10 S1 predictions were invalid because A answered
   the object word itself ("bird", "cheese") or added the answer before the bucket.
3. **Parity**: A's own answers are odd 2:1, its S1 predictions are near-balanced (odd 34, even 26), so it scores below
   the majority baseline. B's self-predictions are almost always "odd".
4. A's S1 and O1 predictions coincide on 75% of first_letter items (section 2), so most of the self − cross gap comes
   from B answering differently, not from A predicting differently.

## 8. Caveats
- Self > cross on first_letter (+0.116 [0.053, 0.179]; +0.084 [0.026, 0.147] with S2/O2) is **not** evidence of
  introspection. A's predictions largely ignore which model is named; any model that predicts "what I would say"
  will score higher on its own answers than on another model's. The S − O gap mostly reflects A–B disagreement.
- Gate 1 compares with the majority class; for first_letter (22 classes) that is a low bar (0.095). A model that
  simply guesses the most typical answer for the question would also clear it. Stages 3–4 must test whether a
  direction is causally involved, with the random and d_you controls.
- Dev only, one seed, greedy decoding, n = 60–190 per property; CIs are wide for parity.
- One B model (Gemma-4-E4B-it); "Gemma" is named in O1, and A may hold priors about Gemma.

## 9. Engineering decisions
- `scripts/prompts.py` holds the format lines, property questions and S1/O1/S2/O2 frames. The quoted task in a
  prediction prompt is the **full object prompt** (task + its format line), verbatim, so the prediction is about the
  exact prompt that produced the target. The S2/O2 sentences lowercase the property question after the comma.
- Property questions: first_letter as in the protocol; parity "Would your answer be an even or an odd number? Answer
  with even or odd only."; length_bucket states the bucket boundaries ("short (4 or fewer letters), medium (5 to 7)
  or long (8 or more)").
- Arithmetic factual items ask for the number written as a word (gold "seven", alias "7") so first_letter applies.
  Capitals were chosen to be single words.
- Split stratified by subcategory (finer than, and implies, stratification by category).
- Parser interpretations (all in `parsers.py` docstring): the first non-empty line is used; markdown marks, quotes
  and trailing punctuation are stripped repeatedly; "The answer is X" is not rewritten (first letter T, noncompliant);
  accented letters fold to A–Z; parity requires the whole normalised answer to be one integer in [lo, hi] (bounds
  inclusive); predictions must be exactly the single token after normalisation; gold match is an exact,
  case-insensitive match of the normalised answer to gold or an alias.
- Perplexity windows start at "It is a truth universally acknowledged" (first line of Chapter I, after the Gutenberg
  header and front matter); no BOS token is added (Qwen uses none); 255 scored tokens per window.
- `common.py` gained `generate_batch` (manual left padding of the exact `chat_ids` token ids). Batch-size fallback
  triggers on either the pre-check or the post-check (the post-check rule was added after the first run exposed a
  B mismatch the pre-check missed; that first run was discarded and everything re-run).
- The format-pilot selection rule (highest minimum validity, tie-break by mean, then order) was fixed in the script
  before it ran.
