"""MIRROR-01 step 4: scoring with FIXED targets, the Gate and the H3 -> H1/H2 decision rule (protocol_01.md).

Written before any MIRROR-01 generation was inspected. Inputs: results/mirror01/objects.csv, predictions.csv,
run.json, strength_matching.json. Outputs: results/mirror01/scored_items.csv, metrics.json,
random_distribution.png.

Scoring (per wording; primary = S1/O1, secondary = S2/O2):
- A_t = first letter of A's UNHOOKED object answer; B_t = first letter of Gemma's object answer.
- self = self-frame prediction == A_t; cross = other-frame prediction == B_t; cas = other-frame prediction == A_t.
- Invalid predictions are wrong. Eligibility: self and cas need a valid A_t; cross needs a valid B_t.
- Drops = baseline - condition, paired over items; S_cross = self_drop - cross_drop and S_cas = self_drop - cas_drop
  are computed per item on the items eligible for both terms. Paired bootstrap (10,000 resamples, seed 0,
  percentile 95% CI) via stage2_metrics.boot_ci.
Gate: baseline self accuracy 95% CI lower bound > majority-class share of A_t (eligible items).
Decision (d_L; random = the 20 strength-matched directions; d_you = the strength-matched d_you):
  H3 if self_drop CI includes 0 (lo <= 0 <= hi) or self_drop is not strictly larger than every random self_drop.
  Else H1 if S_cas CI lo > 0, S_cross CI lo > 0, S_cas > every random S_cas, and matched d_you self_drop
  < 0.5 * d_L self_drop. Else H2.
Fragility (fixed here): the secondary (S2/O2) analysis is run with the same gate and rule; the primary verdict is
"fragile" if the secondary verdict differs from it (a secondary gate failure counts as a differing verdict).
"""
import json

import numpy as np
import pandas as pd

import parsers as P
from common import RESULTS, ROOT
from stage2_metrics import boot_ci

OUT = RESULTS / "mirror01"
WORDINGS = {"primary": ("S1", "O1"), "secondary": ("S2", "O2")}


def ci(x):
    return [round(v, 4) if v is not None else None for v in boot_ci(x)]


def echo_flags(answer, phrase):
    """Echo of the completion phrase. strict (Stage 5 rule): multi-word answer whose first word repeats the phrase's
    first word. broad: the answer's first word is any word of the phrase, or the answer contains the whole phrase."""
    w = [x.casefold() for x in P.words(answer)]
    pw = [x.casefold() for x in phrase.split()]
    strict = len(w) > 1 and w[0] == pw[0]
    broad = bool(w) and (w[0] in pw or phrase.casefold() in P.normalise(answer).casefold())
    return strict, broad


def build_table(df, obj, preds, cond):
    rows = []
    for r in df.itertuples():
        a, b = obj.loc[r.id, "A_obj"], obj.loc[r.id, "B_obj"]
        row = dict(id=r.id, category=r.category, subcategory=r.subcategory, condition=cond,
                   A_obj=a, B_obj=b, A_t=P.first_letter(a), B_t=P.first_letter(b))
        for fr in ("S1", "O1", "S2", "O2"):
            raw = preds.get((r.id, cond, fr))
            row[f"{fr}_raw"] = raw
            row[f"{fr}_pred"] = P.parse_prediction("first_letter", raw)
        rows.append(row)
    return pd.DataFrame(rows)


def ok(t, sf, of):
    a_ok, b_ok = t.A_t.notna(), t.B_t.notna()
    self_ = (t[f"{sf}_pred"].notna() & (t[f"{sf}_pred"] == t.A_t)).astype(float)
    cross = (t[f"{of}_pred"].notna() & (t[f"{of}_pred"] == t.B_t)).astype(float)
    cas = (t[f"{of}_pred"].notna() & (t[f"{of}_pred"] == t.A_t)).astype(float)
    return self_.to_numpy(), cross.to_numpy(), cas.to_numpy(), a_ok.to_numpy(), b_ok.to_numpy()


def summarise(base, cond, sf, of):
    bs, bc, ba, a_ok, b_ok = ok(base, sf, of)
    cs, cc, ca, _, _ = ok(cond, sf, of)
    both = a_ok & b_ok
    out = {"n_eligible_self": int(a_ok.sum()), "n_eligible_cross": int(b_ok.sum()), "n_eligible_both": int(both.sum()),
           "self_validity": float(cond[f"{sf}_pred"].notna().mean()),
           "other_validity": float(cond[f"{of}_pred"].notna().mean()),
           "self_other_same_pred": float((cond[f"{sf}_pred"].notna() &
                                          (cond[f"{sf}_pred"] == cond[f"{of}_pred"])).mean()),
           "self_pred_changed_vs_base": float((cond[f"{sf}_raw"] != base[f"{sf}_raw"]).mean()),
           "other_pred_changed_vs_base": float((cond[f"{of}_raw"] != base[f"{of}_raw"]).mean())}
    for k, b, c, m in (("self", bs, cs, a_ok), ("cross", bc, cc, b_ok), ("cas", ba, ca, a_ok)):
        out[f"{k}_base"] = float(b[m].mean())
        out[f"{k}_acc"] = float(c[m].mean())
        out[f"{k}_acc_ci95"] = ci(c[m])
        out[f"{k}_drop"] = float((b[m] - c[m]).mean())
        out[f"{k}_drop_ci95"] = ci(b[m] - c[m])
    sd, xd, ad = bs - cs, bc - cc, ba - ca
    out["S_cross"] = float((sd - xd)[both].mean())
    out["S_cross_ci95"] = ci((sd - xd)[both])
    out["S_cas"] = float((sd - ad)[a_ok].mean())
    out["S_cas_ci95"] = ci((sd - ad)[a_ok])
    return out


def gate(base, sf):
    e = base[base.A_t.notna()]
    acc = (e[f"{sf}_pred"].notna() & (e[f"{sf}_pred"] == e.A_t)).astype(float).to_numpy()
    freq = e.A_t.value_counts(normalize=True)
    lo, hi = ci(acc)
    return {"n": int(len(e)), "self_acc": float(acc.mean()), "self_ci95": [lo, hi],
            "majority_letter": str(freq.index[0]), "majority_baseline": float(freq.iloc[0]),
            "freq_random_baseline": float((freq ** 2).sum()), "pass": bool(lo > freq.iloc[0])}


def decide(m, g, randoms):
    if not g["pass"]:
        return {"verdict": "gate failed (no hypothesis tested)"}
    dl, dy = m["d_L"], m["d_you_matched"]
    rnd = [m[r] for r in randoms]
    lo, hi = dl["self_drop_ci95"]
    c = {"self_drop_ci_includes_0": bool(lo <= 0 <= hi),
         "self_drop_gt_all_random": bool(all(dl["self_drop"] > r["self_drop"] for r in rnd)),
         "self_drop_rank": int(1 + sum(r["self_drop"] >= dl["self_drop"] for r in rnd)),
         "max_random_self_drop": float(max(r["self_drop"] for r in rnd))}
    c["H3"] = c["self_drop_ci_includes_0"] or not c["self_drop_gt_all_random"]
    c["h1_S_cas_ci_gt_0"] = bool(dl["S_cas_ci95"][0] > 0)
    c["h1_S_cross_ci_gt_0"] = bool(dl["S_cross_ci95"][0] > 0)
    c["h1_S_cas_gt_all_random"] = bool(all(dl["S_cas"] > r["S_cas"] for r in rnd))
    c["S_cas_rank"] = int(1 + sum(r["S_cas"] >= dl["S_cas"] for r in rnd))
    c["max_random_S_cas"] = float(max(r["S_cas"] for r in rnd))
    c["h1_dyou_matched_lt_half"] = bool(dy["self_drop"] < 0.5 * dl["self_drop"])
    c["dyou_matched_self_drop"], c["half_dL_self_drop"] = dy["self_drop"], 0.5 * dl["self_drop"]
    h1 = all(c[k] for k in ("h1_S_cas_ci_gt_0", "h1_S_cross_ci_gt_0", "h1_S_cas_gt_all_random",
                            "h1_dyou_matched_lt_half"))
    c["verdict"] = "H3" if c["H3"] else ("H1" if h1 else "H2")
    return c


def plot(metrics, randoms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, red, grey, ink = "#2a78d6", "#eb6834", "#c23b3b", "#8c8c86", "#2b2b29"
    keys = (("self_drop", "Self drop"), ("S_cross", "S_cross"), ("S_cas", "S_cas"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharey=True)
    for i, setname in enumerate(("primary", "secondary")):
        m = metrics[setname]["conditions"]
        for j, (key, lab) in enumerate(keys):
            ax = axes[i, j]
            rv = [m[r][key] for r in randoms]
            ax.scatter(rv, np.random.default_rng(0).uniform(-0.15, 0.15, len(rv)), s=30, color=grey, alpha=0.8,
                       label="matched random (20)", zorder=2)
            for name, col, y in (("d_L", blue, 0.5), ("d_you", orange, -0.45), ("d_you_matched", red, -0.75)):
                v, (lo, hi) = m[name][key], m[name][f"{key}_ci95"]
                ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", ms=6, color=col, lw=1.4, capsize=3,
                            label=f"{name} (95% CI)", zorder=3)
            ax.axvline(0, color=ink, lw=0.8, alpha=0.5)
            ax.set_yticks([])
            ax.set_ylim(-1.0, 0.9)
            ax.set_xlabel(f"{lab} ({'S1/O1' if setname == 'primary' else 'S2/O2'})", fontsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("MIRROR-01: fixed-target drops, d_L vs strength-matched random directions", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "random_distribution.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "data" / "dataset_01.csv", keep_default_na=False)
    obj = pd.read_csv(OUT / "objects.csv", keep_default_na=False, na_values=[]).set_index("id")
    pr = pd.read_csv(OUT / "predictions.csv", keep_default_na=False, na_values=[])
    preds = {(r.id, r.condition, r.frame): r.raw for r in pr.itertuples()}
    run = json.loads((OUT / "run.json").read_text())
    conds = list(run["conditions"])
    randoms = [c for c in conds if c.startswith("random_")]
    assert len(randoms) == 20
    tables = {c: build_table(df, obj, preds, c) for c in conds}
    pd.concat(tables.values()).to_csv(OUT / "scored_items.csv", index=False)
    base = tables["baseline"]

    # Descriptives (baseline; object answers are unhooked, so the same in every condition).
    comp = base[base.category == "completion"]
    ph = df.set_index("id").phrase
    ech = {m: [echo_flags(a, ph[i]) for i, a in zip(comp.id, comp[f"{m}_obj"])] for m in ("A", "B")}
    desc = {
        "n_items": int(len(df)), "n_choice": int((df.category == "choice").sum()),
        "n_completion": int((df.category == "completion").sum()),
        "object_validity": {m: float(base[f"{m}_t"].notna().mean()) for m in ("A", "B")},
        "object_noncompliant_multiword": {m: float(base[f"{m}_obj"].map(P.noncompliant).mean()) for m in ("A", "B")},
        "A_letter_counts": base.A_t.value_counts().to_dict(), "B_letter_counts": base.B_t.value_counts().to_dict(),
        "n_distinct_A_letters": int(base.A_t.nunique()),
        "A_B_same_first_letter": float((base.A_t.notna() & (base.A_t == base.B_t)).mean()),
        "completion_echo": {m: {"strict_rate": float(np.mean([e[0] for e in ech[m]])),
                                "broad_rate": float(np.mean([e[1] for e in ech[m]])),
                                "n": len(ech[m])} for m in ("A", "B")},
        "prediction_validity": {c: {fr: float(tables[c][f"{fr}_pred"].notna().mean())
                                    for fr in ("S1", "O1", "S2", "O2")} for c in conds},
        "mean_abs_proj": {c: run["conditions"][c]["mean_abs_proj_gen"] for c in conds},
        "mean_abs_removed": {c: run["conditions"][c]["mean_abs_removed"] for c in conds},
        "alpha": {c: run["conditions"][c]["alpha"] for c in conds},
    }
    e = base[base.A_t.notna()]
    desc["baseline_by_category"] = {
        cat: {"n": int((e.category == cat).sum()),
              "self_S1": float((e[e.category == cat].S1_pred == e[e.category == cat].A_t).mean()),
              "cas_O1": float((e[e.category == cat].O1_pred == e[e.category == cat].A_t).mean())}
        for cat in ("choice", "completion")}

    metrics = {"descriptives": desc, "n_bootstrap": 10_000, "bootstrap_seed": 0,
               "fragility_rule": "fragile if the S2/O2 verdict (same gate and rule) differs from the S1/O1 verdict"}
    for setname, (sf, of) in WORDINGS.items():
        g = gate(base, sf)
        m = {c: summarise(base, tables[c], sf, of) for c in conds if c != "baseline"}
        m["baseline"] = summarise(base, base, sf, of)
        rs = {k: {"min": float(min(m[r][k] for r in randoms)), "max": float(max(m[r][k] for r in randoms)),
                  "mean": float(np.mean([m[r][k] for r in randoms]))}
              for k in ("self_drop", "cross_drop", "cas_drop", "S_cross", "S_cas", "self_validity", "other_validity")}
        metrics[setname] = {"wording": f"{sf}/{of}", "gate": g, "conditions": m, "random_summary": rs,
                            "decision": decide(m, g, randoms)}
    pv, sv = metrics["primary"]["decision"]["verdict"], metrics["secondary"]["decision"]["verdict"]
    metrics["verdict"] = {"primary": pv, "secondary": sv, "fragile": pv != sv}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    plot(metrics, randoms)

    for setname in WORDINGS:
        x = metrics[setname]
        print(f"\n== {setname} ({x['wording']}) gate: {x['gate']}")
        for c in ["baseline", "d_L", "d_you", "d_you_matched"] + randoms:
            y = x["conditions"][c]
            print(f"{c:14s} self {y['self_acc']:.3f} drop {y['self_drop']:+.3f} {y['self_drop_ci95']} "
                  f"cross {y['cross_acc']:.3f} ({y['cross_drop']:+.3f}) cas {y['cas_acc']:.3f} ({y['cas_drop']:+.3f}) "
                  f"S_cross {y['S_cross']:+.3f} {y['S_cross_ci95']} S_cas {y['S_cas']:+.3f} {y['S_cas_ci95']}")
        print(json.dumps(x["decision"], indent=1))
    print(json.dumps(metrics["verdict"]))


if __name__ == "__main__":
    main()
