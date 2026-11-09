"""MIRROR-00 Stage 5: held-out TEST split, first_letter only (Amendment 3), frozen L and alpha.

Nothing is fitted or selected here. L and alpha come from results/stage3_selection.json; d_L, d_L' and d_you are the
dev-fit directions in results/stage3_directions.pt; random directions use seeds 1-10 exactly as in Stage 4.

Steps
 1. Model B (Gemma, never hooked): object answers for every test first_letter item, plus its S1 self-predictions
    (exploratory, as in Stage 2). Batch size 1. B is then freed.
 2. Model A: the Stage 2 baseline on test (object answers; S1, O1, S2, O2 predictions) under a measure-only hook
    (alpha = 0, output untouched) that records |h . d_L|; then every Stage 4 condition on test: d_L (with S2/O2 =
    paraphrase), d_you, d_L' scored on S1/O1 (transfer), random_1..10. Same code path as Stage 4
    (interventions.run_condition / score_items / summarise; stage4_interventions.criteria).
 3. Test Gate-1-style numbers via stage2_metrics.prop_metrics (information only; they do not gate anything).

Paraphrase-fragility rule (Amendment 1), fixed here before the test run: the effect is "fragile across wordings" if,
with d_L removed and the S2/O2 wording, the self-prediction drop's 95% CI does not lie above 0 (lower bound <= 0).

Outputs: results/stage5_test_raw.csv, stage5_test_metrics.json, stage5_run.json, stage5_random_distribution.png.
"""
import gc
import json
import time

import numpy as np
import pandas as pd
import torch

import parsers as P
from common import MODELS, RESULTS, ROOT, generate_batch, load, set_seed
from interventions import PROP, factual_accuracy, run_condition, score_items, summarise
from prompts import FIXED_WORDING, prediction_prompt
from stage2_baseline import OBJ_TOK, PRED_TOK
from stage2_metrics import prop_metrics
from stage4_interventions import N_RANDOM, criteria, random_dir, sets


def test_fl_items():
    df = pd.read_csv(ROOT / "data" / "dataset.csv", keep_default_na=False)
    t = df[df.split == "test"].reset_index(drop=True)
    assert (t.split == "test").all() and len(t) > 0
    return t[t.properties.str.split("|").apply(lambda ps: PROP in ps)].reset_index(drop=True)


def run_B(fl):
    set_seed()
    t0 = time.time()
    model, tok = load("B")
    rec = {"load_s": round(time.time() - t0, 1)}
    t1 = time.time()
    obj = generate_batch(model, tok, list(fl.object_prompt), batch_size=1, max_new_tokens=OBJ_TOK)
    s1 = generate_batch(model, tok, [prediction_prompt("S1", op, PROP) for op in fl.object_prompt], batch_size=1,
                        max_new_tokens=PRED_TOK)
    rec["gen_s"] = round(time.time() - t1, 1)
    rec["n_generations"] = len(obj) + len(s1)
    rec["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return dict(zip(fl.id, obj)), dict(zip(fl.id, s1)), rec


def gate_table(fl, a_obj, a_preds, b_obj, b_s1):
    """Stage 2-style per-item table for first_letter on test, in the column layout prop_metrics expects."""
    rows = []
    for r in fl.itertuples():
        row = dict(id=r.id, category=r.category, subcategory=r.subcategory, object_prompt=r.object_prompt,
                   gold=r.gold, aliases=r.aliases)
        for m, objs in (("A", a_obj), ("B", b_obj)):
            ans = objs[r.id]
            row[f"{m}_obj_raw"] = ans
            row[f"{m}_target"] = P.first_letter(ans)
            row[f"{m}_noncompliant"] = r.answer_type == "word" and P.noncompliant(ans)
            row[f"{m}_gold_correct"] = P.gold_match(ans, r.gold, r.aliases)
        for fr in ("S1", "O1", "S2", "O2"):
            row[f"A_{fr}_raw"] = a_preds[fr][r.id]
            row[f"A_{fr}_pred"] = P.parse_prediction(PROP, a_preds[fr][r.id])
        row["B_S1_raw"] = b_s1[r.id]
        row["B_S1_pred"] = P.parse_prediction(PROP, b_s1[r.id])
        rows.append(row)
    d = pd.DataFrame(rows)
    for c in ("A_target", "B_target", "A_S1_pred", "A_O1_pred", "A_S2_pred", "A_O2_pred", "B_S1_pred"):
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    return d


def plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, aqua, grey, ink = "#2a78d6", "#eb6834", "#1baf7a", "#8c8c86", "#2b2b29"
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharey="row")
    for j, setname in enumerate(("primary", "secondary")):
        m = metrics[setname]
        for i, (key, lab) in enumerate((("self_drop", "Self-prediction drop"), ("S", "S = self drop − cross drop"))):
            ax = axes[i, j]
            rv = [m[f"random_{s}"][key] for s in range(1, N_RANDOM + 1)]
            ax.scatter(rv, np.zeros(len(rv)), s=40, color=grey, alpha=0.8, label="random directions (10)", zorder=2)
            for name, c, y in (("d_L", blue, 0.3), ("d_you", orange, -0.3), ("d_Lp", aqua, 0.6)):
                v, ci = m[name][key], m[name][f"{key}_ci95"]
                ax.errorbar([v], [y], xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", ms=7, color=c, lw=1.5, capsize=3,
                            label={"d_L": "d_L (95% CI)", "d_you": "d_you (95% CI)",
                                   "d_Lp": "d_L′ on S1/O1 (95% CI)"}[name], zorder=3)
            ax.axvline(0, color=ink, lw=0.8, alpha=0.5)
            ax.set_yticks([])
            ax.set_ylim(-0.8, 1.0)
            ax.set_xlabel(lab, fontsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
            if i == 0:
                ax.set_title(f"{setname} set (n = {m['d_L']['n_eligible']})", fontsize=10)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Stage 5 (TEST): d_L vs random directions and controls", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage5_random_distribution.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    sel = json.loads((RESULTS / "stage3_selection.json").read_text())
    L, alpha = sel["layer"], sel["alpha"]
    assert (L, alpha) == (12, 2.0), "frozen setting changed"
    D = torch.load(RESULTS / "stage3_directions.pt", weights_only=True)
    fl = test_fl_items()
    print(f"Stage 5 on {len(fl)} TEST first_letter items at L={L}, alpha={alpha}", flush=True)
    run = {"split": "test", "layer": L, "alpha": alpha, "n_items": len(fl), "format_wording": FIXED_WORDING,
           "models": {k: MODELS[k]["name"] for k in ("A", "B")}, "batch_size": 1}

    # 1. Gemma (B), unhooked.
    b_obj, b_s1, run["B"] = run_B(fl)
    print(f"[B] {run['B']['n_generations']} generations in {run['B']['gen_s']}s", flush=True)
    b_raw = pd.DataFrame({"id": fl.id, "B_target": [P.first_letter(b_obj[i]) for i in fl.id]}).set_index("id")

    # 2. Model A.
    set_seed()
    model, tok = load("A")
    hidden = D["d_L"].shape[1]
    dirs = {"d_L": D["d_L"][L], "d_you": D["d_you"][L], "d_Lp": D["d_Lp"][L]}
    dirs.update({f"random_{s}": random_dir(s, hidden) for s in range(1, N_RANDOM + 1)})
    gens = {"baseline": run_condition(model, tok, fl, fl, frames=("S1", "O1", "S2", "O2"), d=dirs["d_L"], layer=L,
                                      alpha=0.0)}
    r = gens["baseline"][2]
    print(f"baseline [{r['gen_s']}s] |h.d_L| {r['mean_abs_proj_gen']:.3f} ppl {r['perplexity']['ppl']:.3f}", flush=True)
    order = ["d_L", "d_you", "d_Lp"] + [f"random_{s}" for s in range(1, N_RANDOM + 1)]
    for name in order:
        frames = ("S1", "O1", "S2", "O2") if name == "d_L" else ("S1", "O1")
        gens[name] = run_condition(model, tok, fl, fl, frames=frames, d=dirs[name], layer=L, alpha=alpha)
        r = gens[name][2]
        print(f"{name}: [{r['gen_s']}s] |h.d| {r['mean_abs_proj_gen']:.3f} ppl {r['perplexity']['ppl']:.3f}",
              flush=True)
    run["A_vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    base_obj, base_preds, base_rec = gens["baseline"]
    fact_base, n_fact = factual_accuracy(fl, base_obj)
    ppl_base = base_rec["perplexity"]["ppl"]

    # 3. Gate-1-style numbers on test (information only).
    gt = gate_table(fl, base_obj, base_preds, b_obj, b_s1)
    gate_info = prop_metrics(gt)
    bfact = gt[gt.category == "factual"].B_gold_correct.astype(bool).mean()

    tables = {"baseline": score_items(fl, b_raw, base_obj, base_preds, base_obj, base_preds)}
    for name in order:
        tables[name] = score_items(fl, b_raw, base_obj, base_preds, gens[name][0], gens[name][1])
    tables["baseline_S2O2"] = score_items(fl, b_raw, base_obj, base_preds, base_obj, base_preds, "S2", "O2")
    tables["d_L_S2O2"] = score_items(fl, b_raw, base_obj, base_preds, gens["d_L"][0], gens["d_L"][1], "S2", "O2")
    src = {"baseline_S2O2": "baseline", "d_L_S2O2": "d_L"}
    raw = pd.concat([t.assign(condition=k, wording="S2/O2" if k.endswith("S2O2") else "S1/O1")
                     for k, t in tables.items()])
    raw["B_obj"] = raw.id.map(b_obj)
    raw["B_S1_raw"] = raw.id.map(b_s1)
    raw.to_csv(RESULTS / "stage5_test_raw.csv", index=False)

    metrics = {"split": "test", "layer": L, "alpha": alpha, "n_bootstrap": 10_000, "bootstrap_seed": 0,
               "baseline_factual_acc": fact_base, "n_factual": n_fact, "baseline_ppl": ppl_base,
               "B_factual_acc": float(bfact),
               "gate1_style_info_only": gate_info,
               "paraphrase_fragility_rule": "fragile if d_L self drop on S2/O2 has 95% CI lower bound <= 0"}
    for setname in ("primary", "secondary"):
        m = {}
        for k, t in tables.items():
            o, _, r = gens[src.get(k, k)]
            fa, _ = factual_accuracy(fl, o)
            m[k] = {**summarise(sets(t)[setname]), "factual_acc": fa, "factual_drop": fact_base - fa,
                    "ppl": r["perplexity"]["ppl"], "ppl_rise": r["perplexity"]["ppl"] / ppl_base - 1,
                    "mean_abs_proj": r.get("mean_abs_proj_gen"),
                    "mean_abs_removed": None if r.get("mean_abs_proj_gen") is None else
                    (0.0 if k.startswith("baseline") else alpha * r["mean_abs_proj_gen"]),
                    "mean_abs_proj_ppl": r.get("mean_abs_proj_ppl")}
        rnd = [m[f"random_{s}"] for s in range(1, N_RANDOM + 1)]
        m["random_summary"] = {key: {"min": float(min(x[key] for x in rnd)), "max": float(max(x[key] for x in rnd)),
                                     "mean": float(np.mean([x[key] for x in rnd]))}
                               for key in ("self_drop", "cross_drop", "S", "cas_drop", "factual_drop", "ppl_rise",
                                           "object_answer_change_rate", "object_letter_change_rate",
                                           "mean_abs_proj")}
        c = criteria(m, ppl_base, fact_base)
        c["paraphrase_fragile"] = bool(m["d_L_S2O2"]["self_drop_ci95"][0] <= 0)
        c["e_ok"] = c["e_dyou_lt_half"]
        if not (c["a_ok"] and c["b_ok"]):
            c["verdict"] = "Stop"
        elif c["c_ok"] and c["d_ok"] and c["e_ok"] and not c["paraphrase_fragile"]:
            c["verdict"] = "Continue"
        else:
            c["verdict"] = "Refine"
        m["criteria_test"] = c
        metrics[setname] = m
    metrics["decision_rule_applies_to"] = "primary"
    metrics["recommendation"] = metrics["primary"]["criteria_test"]["verdict"]
    (RESULTS / "stage5_test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    run["conditions"] = {k: v[2] for k, v in gens.items()}
    run["wall_clock_s"] = round(time.time() - t_start, 1)
    (RESULTS / "stage5_run.json").write_text(json.dumps(run, indent=2, default=str))
    plot(metrics)

    g = gate_info["wording1"]
    print(f"\nGate-1-style (info): n={gate_info['n_eligible_A']} self {g['self']['acc']:.3f} {g['self']['ci95']} "
          f"cross {g['cross']['acc']:.3f} maj {gate_info['targets_A']['majority_baseline']:.3f} "
          f"pass={gate_info['gate1']['pass']}")
    for setname in ("primary", "secondary"):
        m = metrics[setname]
        print(f"\n== {setname} ==")
        for k in ["baseline", "d_L", "d_you", "d_Lp"] + [f"random_{s}" for s in range(1, N_RANDOM + 1)] + [
                "baseline_S2O2", "d_L_S2O2"]:
            x = m[k]
            print(f"{k:14s} n={x['n_eligible']:3d} self {x['self_cond']:.3f} drop {x['self_drop']:+.3f} "
                  f"{x['self_drop_ci95']} cross {x['cross_cond']:.3f} ({x['cross_drop']:+.3f}) S {x['S']:+.3f} "
                  f"{x['S_ci95']} cas {x['cas_cond']:.3f} fact {x['factual_acc']:.3f} ppl {x['ppl']:.2f} "
                  f"chg {x['object_answer_change_rate']:.2f} |hd| {x['mean_abs_proj']}")
        print(json.dumps(m["criteria_test"], indent=1, default=str))
    print(f"total {run['wall_clock_s']}s")


if __name__ == "__main__":
    main()
