"""MIRROR-00 Stage 2 format pilot (Amendment 2): at most 20 DEV items, model A only.

Tries each candidate format-line wording in prompts.FORMAT_WORDINGS for A's object answers and S1/O1 predictions,
and logs validity per wording and condition. Only the format line varies; tasks and properties are unchanged.
Selection rule (fixed before running): highest minimum validity over {object, S1, O1}; ties -> higher mean; then
the earlier wording.
Outputs: results/stage2_format_pilot.csv (summary), results/stage2_format_pilot_raw.csv (every generation).
"""
import json
import random
import time

import pandas as pd

import parsers as P
from common import RESULTS, ROOT, SEED, generate, load, set_seed
from prompts import FORMAT_WORDINGS, object_prompt, prediction_prompt


def pilot_items():
    df = pd.read_csv(ROOT / "data" / "dataset.csv", keep_default_na=False)
    dev = df[df.split == "dev"]
    rng = random.Random(SEED)
    picks = []
    for cat in ["choice", "numeric", "completion", "factual"]:  # 5 per category = 20
        ids = sorted(dev[dev.category == cat].id)
        if cat in ("choice", "factual"):  # spread over subcategories
            subs = sorted(dev[dev.category == cat].subcategory.unique())
            rng.shuffle(subs)
            ids = [rng.choice(sorted(dev[dev.subcategory == s].id)) for s in subs[:5]] if cat == "choice" else \
                [rng.choice(sorted(dev[dev.subcategory == s].id)) for s in subs] + \
                [rng.choice(sorted(dev[(dev.category == cat)].id))]
        else:
            ids = rng.sample(ids, 5)
        picks += ids[:5]
    assert len(set(picks)) == 20
    return dev.set_index("id").loc[picks].reset_index()


def main():
    t0 = time.time()
    set_seed()
    items = pilot_items()
    model, tok = load("A")
    raw, summ = [], []
    for w in FORMAT_WORDINGS:
        for _, r in items.iterrows():
            op = object_prompt(r.task, r.answer_type, w)
            ans = generate(model, tok, op, max_new_tokens=16)
            props = r.properties.split("|")
            ok = all(P.object_property(p, ans, r.lo or None, r.hi or None) is not None for p in props)
            nc = r.answer_type == "word" and P.noncompliant(ans)
            raw.append(dict(wording=w, id=r.id, condition="object", prop="", prompt=op, raw=ans, valid=ok,
                            noncompliant=nc))
            for p in props:
                for fr in ("S1", "O1"):
                    pp = prediction_prompt(fr, op, p, w)
                    g = generate(model, tok, pp, max_new_tokens=8)
                    raw.append(dict(wording=w, id=r.id, condition=fr, prop=p, prompt=pp, raw=g,
                                    valid=P.parse_prediction(p, g) is not None, noncompliant=False))
        print(f"{w} done ({time.time() - t0:.0f}s)", flush=True)
    raw = pd.DataFrame(raw)
    for (w, c), g in raw.groupby(["wording", "condition"], sort=False):
        summ.append(dict(wording=w, condition=c, n=len(g), n_valid=int(g.valid.sum()),
                         validity=round(g.valid.mean(), 3), n_noncompliant=int(g.noncompliant.sum()),
                         format_lines=json.dumps(FORMAT_WORDINGS[w])))
    summ = pd.DataFrame(summ)
    agg = summ.groupby("wording", sort=False).validity.agg(["min", "mean"])
    best = max(agg.index, key=lambda w: (agg.loc[w, "min"], agg.loc[w, "mean"], -list(agg.index).index(w)))
    summ["selected"] = summ.wording == best
    summ.to_csv(RESULTS / "stage2_format_pilot.csv", index=False)
    raw.to_csv(RESULTS / "stage2_format_pilot_raw.csv", index=False)
    print(summ.drop(columns="format_lines").to_string())
    print(f"selected: {best}; items: {list(items.id)}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
