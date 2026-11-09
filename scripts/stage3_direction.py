"""MIRROR-00 Stage 3: candidate direction and (L, alpha) selection. DEV split only; no test item is touched.

1. Split the dev first_letter items into dev-fit (60%) / dev-select (40%), stratified by subcategory, seed 0.
2. On dev-fit: last-input-token post-block residuals of A at every layer for S1/O1 (-> d_L) and S2/O2 (-> d_L').
3. d_you from 60 paired unrelated prompts ("What do you think about X?" vs "What does she think about X?").
4. Sweep on dev-select: layers 3, 6, ..., n-3; alpha in {1, 2}. Regenerate A's object answers, S1 and O1 under the
   intervention; factual accuracy on all dev factual items; neutral-text perplexity (Stage 2 procedure).
5. Selection rule (protocol): maximise S = self-drop - cross-drop subject to factual drop <= 5 points and perplexity
   increase <= 10%; otherwise the smallest constraint violation. Writes results/stage3_selection.json.
Outputs: stage3_split.json, stage3_directions.pt, stage3_cosines.csv, stage3_sweep.csv, stage3_sweep_raw.csv,
stage3_run.json, stage3_selection.json.
"""
import datetime
import json
import random
import time

import pandas as pd
import torch

from common import RESULTS, SEED, Capture, chat_ids, decoder_layers, load, set_seed
from interventions import (PROP, factual_accuracy, fit_select_split, fl_items, run_condition, score_items,
                           summarise)
from prompts import prediction_prompt

# Neutral topics for the second-person control; 60 are drawn with seed 0.
TOPICS = ["modern architecture", "jazz music", "electric cars", "public libraries", "remote work", "space tourism",
          "vegetarian food", "classical ballet", "social media", "city parks", "board games", "online shopping",
          "nuclear energy", "street art", "winter holidays", "high-speed trains", "reality television",
          "minimalist design", "tea ceremonies", "football tactics", "abstract painting", "open-plan offices",
          "mountain hiking", "fast fashion", "video games", "poetry readings", "self-driving trucks", "school uniforms",
          "four-day work weeks", "museum audio guides", "urban gardening", "crossword puzzles", "folk music",
          "electric scooters", "silent films", "coffee culture", "wind farms", "science fiction novels",
          "reality-based documentaries", "public transport", "handwritten letters", "camping trips", "opera",
          "smart watches", "local markets", "chess tournaments", "cooking shows", "skyscrapers", "bird watching",
          "podcasts", "historical dramas", "tiny houses", "marathon running", "cruise holidays", "vinyl records",
          "graphic novels", "cycling lanes", "frozen food", "theme parks", "language learning apps",
          "solar panels", "wedding traditions", "stand-up comedy", "botanical gardens", "pop art", "night markets",
          "dog shows", "electric guitars", "photography exhibitions", "airport design", "home baking",
          "space telescopes", "sculpture parks", "summer festivals", "recycling schemes", "rock climbing",
          "film soundtracks", "old lighthouses", "sushi", "tax software"]
N_YOU = 60
ALPHAS = (1.0, 2.0)
FACT_MAX_DROP, PPL_MAX_RISE = 0.05, 0.10


@torch.no_grad()
def last_token_acts(model, tok, prompts):
    """List of prompts -> tensor [n_layers, n_prompts, hidden] of last-input-token post-block residuals."""
    cap = Capture(model)
    out = []
    try:
        for p in prompts:
            cap.acts = {}
            model(input_ids=chat_ids(tok, p)[0])
            out.append(torch.stack([cap.acts[i] for i in range(len(cap.acts))]))
    finally:
        cap.remove()
    return torch.stack(out, 1)


def unit_diff(a, b):
    raw = a.mean(1) - b.mean(1)  # [layers, hidden]
    return raw / raw.norm(dim=-1, keepdim=True), raw.norm(dim=-1)


def selection(sweep):
    s = sweep.copy()
    s["fact_violation"] = (s.factual_drop - FACT_MAX_DROP).clip(lower=0) / FACT_MAX_DROP
    s["ppl_violation"] = (s.ppl_rise - PPL_MAX_RISE).clip(lower=0) / PPL_MAX_RISE
    s["violation"] = s.fact_violation + s.ppl_violation
    ok = s[s.violation == 0]
    if len(ok):
        # Highest S; ties -> lower alpha, then lower layer.
        best = ok.sort_values(["S", "alpha", "layer"], ascending=[False, True, True]).iloc[0]
        rule = "max S among settings meeting both constraints"
    else:
        best = s.sort_values(["violation", "S", "alpha", "layer"], ascending=[True, False, True, True]).iloc[0]
        rule = "NO setting met both constraints: smallest normalised violation, then max S"
    return best, rule, int(len(ok))


def main():
    t_start = time.time()
    set_seed()
    fl, b_raw = fl_items()
    fit_ids, sel_ids = fit_select_split(fl)
    (RESULTS / "stage3_split.json").write_text(json.dumps(
        {"property": PROP, "stratified_by": "subcategory", "seed": SEED, "dev_fit": fit_ids, "dev_select": sel_ids,
         "n_fit": len(fit_ids), "n_select": len(sel_ids)}, indent=1))
    fit = fl[fl.id.isin(fit_ids)].reset_index(drop=True)
    sel = fl[fl.id.isin(sel_ids)].reset_index(drop=True)
    fact = fl[fl.category == "factual"]
    obj_df = fl[fl.id.isin(set(sel_ids) | set(fact.id))].reset_index(drop=True)  # dev-select + all dev factual
    print(f"dev-fit {len(fit)}, dev-select {len(sel)}, object items per setting {len(obj_df)}", flush=True)

    model, tok = load("A")
    n_layers = len(decoder_layers(model))
    run = {"split": "dev", "n_layers": n_layers}

    # --- directions ---
    t0 = time.time()
    acts = {fr: last_token_acts(model, tok, [prediction_prompt(fr, op, PROP) for op in fit.object_prompt])
            for fr in ("S1", "O1", "S2", "O2")}
    rng = random.Random(SEED)
    topics = rng.sample(TOPICS, N_YOU)
    you = last_token_acts(model, tok, [f"What do you think about {x}?" for x in topics])
    she = last_token_acts(model, tok, [f"What does she think about {x}?" for x in topics])
    d_L, n_L = unit_diff(acts["S1"], acts["O1"])
    d_Lp, n_Lp = unit_diff(acts["S2"], acts["O2"])
    d_you, n_you = unit_diff(you, she)
    resid_norm = acts["S1"].norm(dim=-1).mean(1)
    torch.save({"d_L": d_L, "d_Lp": d_Lp, "d_you": d_you, "raw_norm_d_L": n_L, "raw_norm_d_Lp": n_Lp,
                "raw_norm_d_you": n_you, "mean_resid_norm_S1": resid_norm, "you_topics": topics,
                "dev_fit_ids": fit_ids, "note": "index = decoder block (output of block i); unit vectors, float32"},
               RESULTS / "stage3_directions.pt")
    cos = pd.DataFrame({"layer": range(n_layers),
                        "cos_dL_dLp": torch.nn.functional.cosine_similarity(d_L, d_Lp, dim=-1).numpy(),
                        "cos_dyou_dL": torch.nn.functional.cosine_similarity(d_you, d_L, dim=-1).numpy(),
                        "cos_dyou_dLp": torch.nn.functional.cosine_similarity(d_you, d_Lp, dim=-1).numpy(),
                        "raw_norm_dL": n_L.numpy(), "raw_norm_dLp": n_Lp.numpy(), "raw_norm_dyou": n_you.numpy(),
                        "mean_resid_norm_S1": resid_norm.numpy()})
    cos.to_csv(RESULTS / "stage3_cosines.csv", index=False)
    run["directions_s"] = round(time.time() - t0, 1)
    print(cos.round(3).to_string(index=False), flush=True)

    # --- baseline on dev-select (no hook) ---
    base_obj, base_preds, base_rec = run_condition(model, tok, obj_df, sel)
    base_fact, n_fact = factual_accuracy(fl, base_obj)
    base_ppl = base_rec["perplexity"]["ppl"]
    s2_same = sum(base_obj[i] == b_raw.loc[i, "A_obj_raw"] for i in obj_df.id) + sum(
        base_preds[fr][i] == b_raw.loc[i, f"A_{fr}_raw"] for fr in ("S1", "O1") for i in sel.id)
    run["baseline"] = {**base_rec, "factual_acc": base_fact, "n_factual": n_fact,
                       "identical_to_stage2": f"{s2_same}/{len(obj_df) + 2 * len(sel)}"}
    print(f"baseline: factual {base_fact:.3f} ppl {base_ppl:.3f}; identical to Stage 2 {s2_same}", flush=True)

    # --- sweep ---
    layers = list(range(3, n_layers - 2, 3))
    rows, raws = [], []
    for L in layers:
        for a in ALPHAS:
            obj, preds, rec = run_condition(model, tok, obj_df, sel, d=d_L[L], layer=L, alpha=a)
            t = score_items(sel, b_raw, base_obj, base_preds, obj, preds)
            m = summarise(t)
            fa, _ = factual_accuracy(fl, obj)
            ppl = rec["perplexity"]["ppl"]
            row = {"layer": L, "alpha": a, "n_eligible": m["n_eligible"], "self_base": m["self_base"],
                   "self_cond": m["self_cond"], "self_drop": m["self_drop"], "cross_base": m["cross_base"],
                   "cross_cond": m["cross_cond"], "cross_drop": m["cross_drop"], "S": m["S"],
                   "S_ci95_lo": m["S_ci95"][0], "S_ci95_hi": m["S_ci95"][1], "cas_drop": m["cas_drop"],
                   "factual_acc": fa, "factual_drop": base_fact - fa, "ppl": ppl, "ppl_rise": ppl / base_ppl - 1,
                   "object_validity": m["object_validity"], "answer_change_rate": m["object_answer_change_rate"],
                   "letter_change_rate": m["object_letter_change_rate"], "self_validity": m["self_validity"],
                   "cross_validity": m["cross_validity"], "mean_abs_proj": rec["mean_abs_proj_gen"],
                   "gen_s": rec["gen_s"]}
            rows.append(row)
            raws.append(t.assign(layer=L, alpha=a))
            print(f"L{L:2d} a{a:.0f}: self {m['self_cond']:.3f} (drop {m['self_drop']:+.3f}) cross {m['cross_cond']:.3f} "
                  f"(drop {m['cross_drop']:+.3f}) S {m['S']:+.3f} | fact {fa:.3f} ppl {ppl:.2f} "
                  f"| chg {m['object_answer_change_rate']:.2f} |hd| {rec['mean_abs_proj_gen']:.2f} "
                  f"[{rec['gen_s']}s]", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "stage3_sweep.csv", index=False)
    pd.concat(raws).to_csv(RESULTS / "stage3_sweep_raw.csv", index=False)
    sweep = pd.DataFrame(rows)

    best, rule, n_ok = selection(sweep)
    selection_rec = {
        "layer": int(best.layer), "alpha": float(best.alpha), "rule": rule, "n_settings": len(sweep),
        "n_settings_meeting_constraints": n_ok,
        "criterion": {k: float(best[k]) for k in ("S", "S_ci95_lo", "S_ci95_hi", "self_drop", "cross_drop",
                                                  "self_base", "self_cond", "cross_base", "cross_cond",
                                                  "factual_acc", "factual_drop", "ppl", "ppl_rise",
                                                  "answer_change_rate", "mean_abs_proj")},
        "n_eligible_dev_select": int(best.n_eligible),
        "baseline_dev_select": {"factual_acc": base_fact, "ppl": base_ppl},
        "constraints": {"factual_max_drop": FACT_MAX_DROP, "ppl_max_rel_rise": PPL_MAX_RISE},
        "tie_break": "higher S, then lower alpha, then lower layer",
        "split": "dev-select only; no test item has been run",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (RESULTS / "stage3_selection.json").write_text(json.dumps(selection_rec, indent=2))
    run["wall_clock_s"] = round(time.time() - t_start, 1)
    run["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (RESULTS / "stage3_run.json").write_text(json.dumps(run, indent=2, default=str))
    print("SELECTED", json.dumps(selection_rec, indent=1), flush=True)
    print(f"total {run['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
