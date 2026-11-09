"""MIRROR-01 step 5 — EXPLORATORY, not part of any decision: re-score MIRROR-00 dev (stage4_dev_raw.csv) and test
(stage5_test_raw.csv) against FIXED targets.

In MIRROR-00 the prediction prompts were generated under the same hook as the object answers, but in separate
prompts, so predictions under a condition do not depend on that condition's object answers. Here:
self = self_pred_cond vs A_target_base (A's unhooked answer); cross = cross_pred_cond vs B_target (unchanged);
cas = cross_pred_cond vs A_target_base. Baseline = the *_pred_base columns against the same targets.
MIRROR-00 random directions (seeds 1-10) were NOT strength-matched. Sets: all items, and non-factual items.
Output: results/mirror01/exploratory_m00_rescore.json.
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS
from mirror01_score import ci

OUT = RESULTS / "mirror01"
FILES = {"dev": "stage4_dev_raw.csv", "test": "stage5_test_raw.csv"}


def score(t):
    e = t[t.A_target_base.notna() & t.B_target.notna()]
    eq = lambda p, y: (e[p].notna() & (e[p] == e[y])).astype(float).to_numpy()
    sb, sc = eq("self_pred_base", "A_target_base"), eq("self_pred_cond", "A_target_base")
    xb, xc = eq("cross_pred_base", "B_target"), eq("cross_pred_cond", "B_target")
    ab, ac = eq("cross_pred_base", "A_target_base"), eq("cross_pred_cond", "A_target_base")
    sd, xd, ad = sb - sc, xb - xc, ab - ac
    return {"n": int(len(e)), "self_base": float(sb.mean()), "self_acc": float(sc.mean()),
            "self_drop": float(sd.mean()), "self_drop_ci95": ci(sd),
            "cross_drop": float(xd.mean()), "cross_drop_ci95": ci(xd),
            "cas_drop": float(ad.mean()), "cas_drop_ci95": ci(ad),
            "S_cross": float((sd - xd).mean()), "S_cross_ci95": ci(sd - xd),
            "S_cas": float((sd - ad).mean()), "S_cas_ci95": ci(sd - ad)}


def main():
    res = {"label": "EXPLORATORY re-scoring of already-seen MIRROR-00 data; not part of the MIRROR-01 decision"}
    for split, f in FILES.items():
        raw = pd.read_csv(RESULTS / f, keep_default_na=False, na_values=[""])
        for setname, sub in (("all", raw), ("non_factual", raw[raw.category != "factual"])):
            m = {c: score(g) for c, g in sub.groupby("condition") if not c.startswith("baseline")}
            rnd = [m[f"random_{s}"] for s in range(1, 11)]
            m["random_summary"] = {k: {"min": float(min(r[k] for r in rnd)), "max": float(max(r[k] for r in rnd)),
                                       "mean": float(np.mean([r[k] for r in rnd]))}
                                   for k in ("self_drop", "cross_drop", "cas_drop", "S_cross", "S_cas")}
            m["d_L_self_drop_gt_all_random"] = bool(all(m["d_L"]["self_drop"] > r["self_drop"] for r in rnd))
            m["d_L_S_cas_gt_all_random"] = bool(all(m["d_L"]["S_cas"] > r["S_cas"] for r in rnd))
            res[f"{split}_{setname}"] = m
    (OUT / "exploratory_m00_rescore.json").write_text(json.dumps(res, indent=2))
    for k, m in res.items():
        if k == "label":
            continue
        print(f"\n== {k}")
        for c in ("d_L", "d_L_S2O2", "d_Lp", "d_you"):
            x = m[c]
            print(f"{c:9s} n={x['n']} self {x['self_base']:.3f}->{x['self_acc']:.3f} drop {x['self_drop']:+.3f} "
                  f"{x['self_drop_ci95']} cross {x['cross_drop']:+.3f} cas {x['cas_drop']:+.3f} "
                  f"S_cross {x['S_cross']:+.3f} {x['S_cross_ci95']} S_cas {x['S_cas']:+.3f} {x['S_cas_ci95']}")
        print("random:", {k2: (round(v['min'], 3), round(v['max'], 3)) for k2, v in m["random_summary"].items()},
              "dL>all rnd:", m["d_L_self_drop_gt_all_random"], "S_cas>all:", m["d_L_S_cas_gt_all_random"])


if __name__ == "__main__":
    main()
