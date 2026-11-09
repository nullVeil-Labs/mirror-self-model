"""Shared helpers for MIRROR-00 Stages 3-4 (DEV split only): item sets, one intervention condition, scoring.

Reuses the Stage 2 prompts, parsers, generation settings, perplexity procedure and bootstrap unchanged.
Only first_letter is used (Amendment 3). All generation is greedy, cached, batch size 1.
"""
import random
import time

import numpy as np
import pandas as pd

import parsers as P
from common import RESULTS, SEED, Ablate, generate_batch
from prompts import prediction_prompt
from stage2_baseline import OBJ_TOK, PRED_TOK, dev_items, perplexity
from stage2_metrics import boot_ci

PROP = "first_letter"


def fl_items():
    """All DEV items carrying the first_letter property, plus Gemma's (B's) Stage 2 dev targets."""
    dev = dev_items()
    fl = dev[dev.properties.str.split("|").apply(lambda ps: PROP in ps)].reset_index(drop=True)
    raw = pd.read_csv(RESULTS / "stage2_dev_raw.csv", keep_default_na=False, na_values=[""])
    raw = raw[raw.prop == PROP].set_index("id")
    assert set(fl.id) == set(raw.index)
    return fl, raw


def fit_select_split(fl, frac_fit=0.6):
    """Stratified by subcategory (as the Stage 2 dev/test split), seed 0."""
    rng = random.Random(SEED)
    fit, sel = [], []
    for _, g in fl.groupby("subcategory", sort=True):
        ids = sorted(g.id)
        rng.shuffle(ids)
        k = round(frac_fit * len(ids))
        fit += ids[:k]
        sel += ids[k:]
    return sorted(fit), sorted(sel)


def run_condition(model, tok, obj_df, pred_df, frames=("S1", "O1"), d=None, layer=None, alpha=1.0, ppl=True):
    """Generate object answers for obj_df and predictions (frames) for pred_df under one hook (or none).

    obj_df may be None (object answers reused from another condition). Returns raw generations and hook stats.
    """
    hook = Ablate(model, layer, d, alpha) if d is not None else None
    rec = {}
    try:
        t0 = time.time()
        obj = {}
        if obj_df is not None:
            obj = dict(zip(obj_df.id, generate_batch(model, tok, list(obj_df.object_prompt), batch_size=1,
                                                     max_new_tokens=OBJ_TOK)))
        preds = {}
        for fr in frames:
            prompts = [prediction_prompt(fr, op, PROP) for op in pred_df.object_prompt]
            preds[fr] = dict(zip(pred_df.id, generate_batch(model, tok, prompts, batch_size=1,
                                                            max_new_tokens=PRED_TOK)))
        rec["gen_s"] = round(time.time() - t0, 1)
        rec["n_generations"] = len(obj) + sum(len(v) for v in preds.values())
        if hook is not None:
            rec["mean_abs_proj_gen"] = hook.mean_abs_proj()
            rec["n_positions_gen"] = hook.n_pos
            hook.reset_stats()
        if ppl:
            rec["perplexity"] = perplexity(model, tok)
            if hook is not None:
                rec["mean_abs_proj_ppl"] = hook.mean_abs_proj()
    finally:
        if hook is not None:
            hook.remove()
    return obj, preds, rec


def factual_accuracy(fl, obj):
    f = fl[fl.category == "factual"]
    x = [bool(P.gold_match(obj[r.id], r.gold, r.aliases)) for r in f.itertuples()]
    return float(np.mean(x)), len(x)


def score_items(items, b_raw, base_obj, base_preds, cond_obj, cond_preds, sf="S1", of="O1"):
    """Per-item paired table. Self = sf prediction vs A's own object answer IN THAT CONDITION; cross = of prediction
    vs Gemma's dev answer; cross-as-self (Amendment 3) = of prediction vs A's own answer in that condition."""
    rows = []
    for r in items.itertuples():
        tb, tc = P.first_letter(base_obj[r.id]), P.first_letter(cond_obj[r.id])
        bt = b_raw.loc[r.id, "B_target"]
        bt = bt if isinstance(bt, str) else None
        row = dict(id=r.id, category=r.category, subcategory=r.subcategory,
                   A_obj_base=base_obj[r.id], A_obj_cond=cond_obj[r.id], A_target_base=tb, A_target_cond=tc,
                   B_target=bt, obj_changed=base_obj[r.id] != cond_obj[r.id], target_changed=tb != tc)
        for tag, preds in (("base", base_preds), ("cond", cond_preds)):
            t = tb if tag == "base" else tc
            sp, op = P.parse_prediction(PROP, preds[sf][r.id]), P.parse_prediction(PROP, preds[of][r.id])
            row[f"self_raw_{tag}"], row[f"cross_raw_{tag}"] = preds[sf][r.id], preds[of][r.id]
            row[f"self_pred_{tag}"], row[f"cross_pred_{tag}"] = sp, op
            row[f"self_ok_{tag}"] = float(sp is not None and t is not None and sp == t)
            row[f"cross_ok_{tag}"] = float(op is not None and bt is not None and op == bt)
            row[f"cas_ok_{tag}"] = float(op is not None and t is not None and op == t)
        row["eligible"] = tb is not None and tc is not None and bt is not None
        rows.append(row)
    return pd.DataFrame(rows)


def _ci(x):
    return [round(v, 4) if v is not None else None for v in boot_ci(x)]


def summarise(t):
    """Metrics over eligible items of a per-item table (from score_items). Drops are before - after."""
    e = t[t.eligible]
    out = {"n_items": int(len(t)), "n_eligible": int(len(e)),
           "object_validity": float(t.A_target_cond.notna().mean()),
           "object_answer_change_rate": float(t.obj_changed.mean()),
           "object_letter_change_rate": float(t.target_changed.mean()),
           "self_validity": float(t.self_pred_cond.notna().mean()),
           "cross_validity": float(t.cross_pred_cond.notna().mean())}
    for k in ("self", "cross", "cas"):
        b, c = e[f"{k}_ok_base"].to_numpy(), e[f"{k}_ok_cond"].to_numpy()
        out[f"{k}_base"] = float(b.mean())
        out[f"{k}_cond"] = float(c.mean())
        out[f"{k}_cond_ci95"] = _ci(c)
        out[f"{k}_drop"] = float((b - c).mean())
        out[f"{k}_drop_ci95"] = _ci(b - c)
    sd = e.self_ok_base.to_numpy() - e.self_ok_cond.to_numpy()
    cd = e.cross_ok_base.to_numpy() - e.cross_ok_cond.to_numpy()
    ad = e.cas_ok_base.to_numpy() - e.cas_ok_cond.to_numpy()
    out["S"] = float((sd - cd).mean())
    out["S_ci95"] = _ci(sd - cd)
    out["self_drop_minus_cas_drop"] = float((sd - ad).mean())
    out["self_drop_minus_cas_drop_ci95"] = _ci(sd - ad)
    return out
