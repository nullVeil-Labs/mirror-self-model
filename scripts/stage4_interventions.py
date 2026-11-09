"""MIRROR-00 Stage 4: interventions and controls on ALL dev first_letter items at the frozen (L, alpha).

Reads results/stage3_selection.json and results/stage3_directions.pt; no test item is touched.
Conditions (each regenerates A's object answers and predictions under the hook, batch size 1):
  baseline (no edit; a measure-only hook records |h . d_L|), d_L, random_1..10 (seeds 1-10), d_you, d_Lp (d_L'
  removed, scored on S1/O1 = transfer, Amendment 1). The d_L and baseline runs also generate S2/O2 predictions, giving
  the paraphrase condition (d_L removed, S2/O2 wording, scored against the S2/O2 baseline).
Outputs: stage4_dev_raw.csv, stage4_dev_metrics.json, stage4_run.json, stage4_random_distribution.png,
stage4_sweep_and_cosines.png.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS, load, set_seed
from interventions import factual_accuracy, fl_items, run_condition, score_items, summarise

N_RANDOM = 10


def random_dir(seed, hidden):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(hidden, generator=g)
    return d / d.norm()


def sets(t):
    return {"primary": t, "secondary": t[t.category != "factual"]}


def criteria(m, ppl_base, fact_base):
    """Protocol decision criteria (a)-(e), evaluated on DEV for one analysis set."""
    dl, dy = m["d_L"], m["d_you"]
    rnd = [m[f"random_{s}"] for s in range(1, N_RANDOM + 1)]
    lo, hi = dl["self_drop_ci95"]
    slo, shi = dl["S_ci95"]
    out = {
        "a_self_drop_ci_excludes_0": bool(lo > 0 or hi < 0), "a_self_drop_positive_ci": bool(lo > 0),
        "b_self_drop_gt_all_random": bool(all(dl["self_drop"] > r["self_drop"] for r in rnd)),
        "b_rank_self_drop": int(1 + sum(r["self_drop"] >= dl["self_drop"] for r in rnd)),
        "c_S_ci_excludes_0": bool(slo > 0 or shi < 0), "c_S_positive_ci": bool(slo > 0),
        "c_S_gt_all_random": bool(all(dl["S"] > r["S"] for r in rnd)),
        "c_rank_S": int(1 + sum(r["S"] >= dl["S"] for r in rnd)),
        "d_factual_drop": fact_base - dl["factual_acc"], "d_ppl_rise": dl["ppl"] / ppl_base - 1,
        "e_dyou_self_drop": dy["self_drop"], "e_half_dL_self_drop": 0.5 * dl["self_drop"],
        "e_dyou_lt_half": bool(dy["self_drop"] < 0.5 * dl["self_drop"]),
        "paraphrase_S2O2_self_drop": m["d_L_S2O2"]["self_drop"], "paraphrase_S2O2_S": m["d_L_S2O2"]["S"],
        "paraphrase_S2O2_self_drop_ci95": m["d_L_S2O2"]["self_drop_ci95"], "paraphrase_S2O2_S_ci95":
            m["d_L_S2O2"]["S_ci95"],
    }
    out["d_ok"] = bool(out["d_factual_drop"] <= 0.05 and out["d_ppl_rise"] <= 0.10)
    out["c_ok"] = out["c_S_positive_ci"] and out["c_S_gt_all_random"]
    out["a_ok"] = out["a_self_drop_positive_ci"]
    out["b_ok"] = out["b_self_drop_gt_all_random"]
    return out


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
            for k, (name, c, y) in enumerate((("d_L", blue, 0.3), ("d_you", orange, -0.3), ("d_Lp", aqua, 0.6))):
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
    fig.suptitle("Stage 4 (DEV): d_L vs random directions and controls", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage4_random_distribution.png", dpi=150)
    plt.close(fig)

    sw = pd.read_csv(RESULTS / "stage3_sweep.csv")
    cos = pd.read_csv(RESULTS / "stage3_cosines.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.plot(cos.layer, cos.cos_dL_dLp, color=blue, lw=2, label="cos(d_L, d_L′)")
    ax.plot(cos.layer, cos.cos_dyou_dL, color=orange, lw=2, label="cos(d_you, d_L)")
    ax.axhline(0, color=ink, lw=0.8, alpha=0.5)
    ax.set_xlabel("layer (output of block)", fontsize=9)
    ax.set_title("Direction cosines (dev-fit)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for a, c in ((1.0, blue), (2.0, orange)):
        s = sw[sw.alpha == a]
        ax.plot(s.layer, s.S, color=c, lw=2, marker="o", ms=5, label=f"S, α = {a:g}")
    ax.axhline(0, color=ink, lw=0.8, alpha=0.5)
    ax.set_xlabel("layer", fontsize=9)
    ax.set_title("Selection sweep on dev-select: S", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "stage4_sweep_and_cosines.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    set_seed()
    sel = json.loads((RESULTS / "stage3_selection.json").read_text())
    L, alpha = sel["layer"], sel["alpha"]
    D = torch.load(RESULTS / "stage3_directions.pt", weights_only=True)
    fl, b_raw = fl_items()
    model, tok = load("A")
    hidden = D["d_L"].shape[1]
    print(f"Stage 4 on {len(fl)} dev first_letter items at L={L}, alpha={alpha}", flush=True)

    dirs = {"d_L": D["d_L"][L], "d_you": D["d_you"][L], "d_Lp": D["d_Lp"][L]}
    dirs.update({f"random_{s}": random_dir(s, hidden) for s in range(1, N_RANDOM + 1)})
    run = {"split": "dev", "layer": L, "alpha": alpha, "n_items": len(fl), "conditions": {}}
    gens = {}

    # Baseline: no edit (alpha = 0 hook only measures |h . d_L|); all four wordings.
    obj, preds, rec = run_condition(model, tok, fl, fl, frames=("S1", "O1", "S2", "O2"), d=dirs["d_L"], layer=L,
                                    alpha=0.0)
    gens["baseline"] = (obj, preds, rec)
    same = sum(obj[i] == b_raw.loc[i, "A_obj_raw"] for i in fl.id) + sum(
        preds[fr][i] == b_raw.loc[i, f"A_{fr}_raw"] for fr in ("S1", "O1", "S2", "O2") for i in fl.id)
    run["baseline_identical_to_stage2"] = f"{same}/{5 * len(fl)}"
    print(f"baseline done [{rec['gen_s']}s]; identical to Stage 2: {run['baseline_identical_to_stage2']}; "
          f"|h.d_L| {rec['mean_abs_proj_gen']:.3f}; ppl {rec['perplexity']['ppl']:.3f}", flush=True)

    order = ["d_L", "d_you", "d_Lp"] + [f"random_{s}" for s in range(1, N_RANDOM + 1)]
    for name in order:
        frames = ("S1", "O1", "S2", "O2") if name == "d_L" else ("S1", "O1")
        gens[name] = run_condition(model, tok, fl, fl, frames=frames, d=dirs[name], layer=L, alpha=alpha)
        r = gens[name][2]
        print(f"{name}: [{r['gen_s']}s] |h.d| {r['mean_abs_proj_gen']:.3f} ppl {r['perplexity']['ppl']:.3f}",
              flush=True)

    base_obj, base_preds, base_rec = gens["baseline"]
    fact_base, n_fact = factual_accuracy(fl, base_obj)
    ppl_base = base_rec["perplexity"]["ppl"]
    tables = {"baseline": score_items(fl, b_raw, base_obj, base_preds, base_obj, base_preds)}
    for name in order:
        tables[name] = score_items(fl, b_raw, base_obj, base_preds, gens[name][0], gens[name][1])
    tables["baseline_S2O2"] = score_items(fl, b_raw, base_obj, base_preds, base_obj, base_preds, "S2", "O2")
    tables["d_L_S2O2"] = score_items(fl, b_raw, base_obj, base_preds, gens["d_L"][0], gens["d_L"][1], "S2", "O2")
    src = {"baseline_S2O2": "baseline", "d_L_S2O2": "d_L"}

    raw = pd.concat([t.assign(condition=k, wording="S2/O2" if k.endswith("S2O2") else "S1/O1")
                     for k, t in tables.items()])
    raw.to_csv(RESULTS / "stage4_dev_raw.csv", index=False)

    metrics = {"split": "dev", "layer": L, "alpha": alpha, "n_bootstrap": 10_000, "bootstrap_seed": 0,
               "baseline_factual_acc": fact_base, "n_factual": n_fact, "baseline_ppl": ppl_base,
               "baseline_identical_to_stage2": run["baseline_identical_to_stage2"]}
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
                                           "object_answer_change_rate", "mean_abs_proj")}
        m["criteria_dev"] = criteria(m, ppl_base, fact_base)
        metrics[setname] = m
    (RESULTS / "stage4_dev_metrics.json").write_text(json.dumps(metrics, indent=2))
    run["conditions"] = {k: v[2] for k, v in gens.items()}
    run["wall_clock_s"] = round(time.time() - t_start, 1)
    run["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (RESULTS / "stage4_run.json").write_text(json.dumps(run, indent=2, default=str))
    plot(metrics)

    for setname in ("primary", "secondary"):
        m = metrics[setname]
        print(f"\n== {setname} ==")
        for k in ["baseline", "d_L", "d_you", "d_Lp"] + [f"random_{s}" for s in range(1, N_RANDOM + 1)] + [
                "baseline_S2O2", "d_L_S2O2"]:
            x = m[k]
            print(f"{k:14s} n={x['n_eligible']:3d} self {x['self_cond']:.3f} ({-x['self_drop']:+.3f}) cross "
                  f"{x['cross_cond']:.3f} ({-x['cross_drop']:+.3f}) S {x['S']:+.3f} {x['S_ci95']} cas "
                  f"{x['cas_cond']:.3f} fact {x['factual_acc']:.3f} ppl {x['ppl']:.2f} chg "
                  f"{x['object_answer_change_rate']:.2f} |hd| {x['mean_abs_proj']}")
        print(json.dumps(m["criteria_dev"], indent=1))
    print(f"total {run['wall_clock_s']}s")


if __name__ == "__main__":
    main()
