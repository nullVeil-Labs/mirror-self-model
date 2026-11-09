"""MIRROR-00 Stage 2 metrics and Gate 1 from results/stage2_dev_raw.csv (DEV only).

Scoring (protocol): invalid object answer -> item ineligible for that property (for that target model);
invalid prediction -> wrong. Bootstrap: 10,000 resamples, seed 0, percentile 95% CI; paired over items for
differences. Gate 1: S1 self-accuracy lower bound > majority-class baseline of A's targets, with >= 40 eligible items.
Writes results/stage2_dev_metrics.json.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from common import RESULTS, SEED

N_BOOT = 10_000
PROPS = ["first_letter", "length_bucket", "parity"]
PAIRS = {"1": ("S1", "O1"), "2": ("S2", "O2")}


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(SEED)
    m = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def acc(pred, target):
    """Correct = valid prediction equal to target (NaN prediction = invalid = wrong)."""
    return (pred.notna() & (pred == target)).astype(float)


def summary(x):
    return {"n": int(len(x)), "acc": float(x.mean()) if len(x) else None, "ci95": boot_ci(x)}


def baselines(target):
    c = Counter(target)
    n = sum(c.values())
    freqs = {k: v / n for k, v in sorted(c.items())}
    maj = max(freqs, key=freqs.get) if freqs else None
    return {"class_counts": dict(sorted(c.items())), "majority_class": maj,
            "majority_baseline": freqs.get(maj), "freq_random_baseline": float(sum(p * p for p in freqs.values()))}


def pred_dist(p):
    return dict(Counter(p.fillna("INVALID")).most_common(8))


def prop_metrics(d):
    a_ok, b_ok = d.A_target.notna(), d.B_target.notna()
    out = {"n_items": int(len(d)), "n_eligible_A": int(a_ok.sum()), "n_eligible_B": int(b_ok.sum()),
           "object_validity_A": float(a_ok.mean()), "object_validity_B": float(b_ok.mean()),
           "noncompliant_rate_A": float(d.A_noncompliant.mean()), "noncompliant_rate_B": float(d.B_noncompliant.mean()),
           "targets_A": baselines(d.A_target[a_ok]), "targets_B": baselines(d.B_target[b_ok])}
    both = a_ok & b_ok
    out["AB_same_value"] = {"n": int(both.sum()),
                            "rate": float((d.A_target[both] == d.B_target[both]).mean()) if both.any() else None}
    for w, (s, o) in PAIRS.items():
        sp, op = d[f"A_{s}_pred"], d[f"A_{o}_pred"]
        self_c, cross_c = acc(sp, d.A_target), acc(op, d.B_target)
        diff = (self_c - cross_c)[both]
        out[f"wording{w}"] = {
            "self_frame": s, "cross_frame": o,
            "self": summary(self_c[a_ok]), "cross": summary(cross_c[b_ok]),
            "self_minus_cross_paired": {"n": int(both.sum()), "diff": float(diff.mean()), "ci95": boot_ci(diff)},
            "self_on_paired_items": float(self_c[both].mean()), "cross_on_paired_items": float(cross_c[both].mean()),
            # Diagnostics: does the cross prediction track A's own answer (self-simulation)?
            "cross_pred_vs_A_target": summary(acc(op, d.A_target)[a_ok]),
            "self_pred_vs_B_target": summary(acc(sp, d.B_target)[b_ok]),
            "S_O_predictions_identical": float((sp.fillna("INV") == op.fillna("INV")).mean()),
            "validity_self": float(sp.notna().mean()), "validity_cross": float(op.notna().mean()),
            "pred_dist_self": pred_dist(sp), "pred_dist_cross": pred_dist(op)}
    bs = acc(d.B_S1_pred, d.B_target)
    out["B_self_S1"] = {**summary(bs[b_ok]), "validity": float(d.B_S1_pred.notna().mean()),
                        "pred_dist": pred_dist(d.B_S1_pred)}
    out["self_S1_by_category"] = {c: summary(acc(g.A_S1_pred, g.A_target)[g.A_target.notna()])["acc"]
                                  for c, g in d.groupby("category")}
    w1 = out["wording1"]
    lb, maj = w1["self"]["ci95"][0], out["targets_A"]["majority_baseline"]
    n = out["n_eligible_A"]
    out["gate1"] = {"n_eligible": n, "self_acc": w1["self"]["acc"], "self_ci95_lower": lb, "majority_baseline": maj,
                    "pass": bool(n >= 40 and lb is not None and lb > maj)}
    return out


def main():
    # Only empty cells are missing (= invalid); keep_default_na=False stops letters like "N" being read as NaN.
    raw = pd.read_csv(RESULTS / "stage2_dev_raw.csv", keep_default_na=False, na_values=[""])
    for c in ("A_noncompliant", "B_noncompliant"):
        raw[c] = raw[c].astype(str) == "True"
    run = json.loads((RESULTS / "stage2_dev_run.json").read_text())
    m = {"split": "dev", "n_bootstrap": N_BOOT, "bootstrap_seed": SEED, "format_wording": run["format_wording"],
         "properties": {p: prop_metrics(raw[raw.prop == p].reset_index(drop=True)) for p in PROPS}}
    fact = raw[(raw.category == "factual") & (raw.prop == "first_letter")]
    for k in ("A", "B"):
        x = fact[f"{k}_gold_correct"].map(lambda v: str(v) == "True").astype(float)
        m[f"factual_accuracy_{k}"] = {**summary(x), "by_subcategory": fact.assign(x=x).groupby("subcategory").x.mean()
                                      .to_dict()}
    m["perplexity_A"] = run["A"]["perplexity"]
    m["batching"] = {k: {"pre_check_equal": run[k]["batch_check"]["equal"],
                         "pre_check_mismatches": run[k]["batch_check"]["mismatches"],
                         "batch_sizes": run[k]["batch_sizes"],
                         "post_check_object_mismatches": len(run[k].get("post_check_object", {}).get("mismatches", [])),
                         "post_check_prediction_mismatches":
                             len(run[k].get("post_check_prediction", {}).get("mismatches", []))}
                     for k in ("A", "B")}
    m["gate1"] = {p: m["properties"][p]["gate1"] for p in PROPS}
    m["gate1_any_pass"] = any(g["pass"] for g in m["gate1"].values())
    (RESULTS / "stage2_dev_metrics.json").write_text(json.dumps(m, indent=2))
    for p in PROPS:
        pm = m["properties"][p]
        for w in PAIRS:
            x = pm[f"wording{w}"]
            print(f"{p:14s} w{w} self {x['self']['acc']:.3f} {np.round(x['self']['ci95'], 3)} n={x['self']['n']} | "
                  f"cross {x['cross']['acc']:.3f} n={x['cross']['n']} | diff {x['self_minus_cross_paired']['diff']:+.3f} "
                  f"{np.round(x['self_minus_cross_paired']['ci95'], 3)} | val {x['validity_self']:.2f}/{x['validity_cross']:.2f}")
        print(f"   maj A {pm['targets_A']['majority_baseline']:.3f} rand {pm['targets_A']['freq_random_baseline']:.3f} "
              f"| maj B {pm['targets_B']['majority_baseline']:.3f} | AB same {pm['AB_same_value']['rate']:.3f} "
              f"| Bself {pm['B_self_S1']['acc']:.3f} | gate1 {pm['gate1']['pass']}")
    print("factual A", m["factual_accuracy_A"]["acc"], "ppl", m["perplexity_A"]["ppl"])


if __name__ == "__main__":
    main()
