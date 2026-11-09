"""MIRROR-00 Stage 2: behavioural baselines on the DEV split only.

Generates (greedy, cached, left-padded batches; batched == unbatched is checked first):
  A object answers; A predictions S1/O1/S2/O2; B object answers; B self-predictions (S1, exploratory);
  A factual accuracy; A neutral-text perplexity (data/pg1342.txt).
Writes results/stage2_dev_raw.csv (one row per item x property) and results/stage2_dev_run.json (run facts).
Metrics are computed by stage2_metrics.py from the raw CSV.
"""
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch

import parsers as P
from common import MODELS, RESULTS, ROOT, SEED, generate, generate_batch, load, set_seed
from prompts import FIXED_WORDING, prediction_prompt

OBJ_TOK, PRED_TOK = 16, 8
BS_OBJ, BS_PRED = 16, 32
A_FRAMES = ["S1", "O1", "S2", "O2"]


def dev_items():
    df = pd.read_csv(ROOT / "data" / "dataset.csv", keep_default_na=False)
    dev = df[df.split == "dev"].reset_index(drop=True)
    assert (dev.split == "dev").all() and len(dev) > 0
    return dev


def pred_jobs(dev, frames):
    return [(r.id, p, fr, prediction_prompt(fr, r.object_prompt, p))
            for r in dev.itertuples() for p in r.properties.split("|") for fr in frames]


def batch_check(model, tok, dev):
    """Pre-check on 10 dev items: object prompts and S1 prompts, each as one left-padded batch vs one-by-one."""
    rng = random.Random(SEED)
    ids = rng.sample(sorted(dev.id), 10)
    sub = dev.set_index("id").loc[ids]
    objs = list(sub.object_prompt)
    preds = [prediction_prompt("S1", op, props.split("|")[0]) for op, props in zip(sub.object_prompt, sub.properties)]
    res = {"items": ids, "mismatches": []}
    for kind, prompts, n in (("object", objs, OBJ_TOK), ("S1", preds, PRED_TOK)):
        b = generate_batch(model, tok, prompts, batch_size=len(prompts), max_new_tokens=n)
        u = [generate(model, tok, p, max_new_tokens=n) for p in prompts]
        res["mismatches"] += [dict(kind=kind, id=i, batched=x, unbatched=y) for i, x, y in zip(ids, b, u) if x != y]
    res["equal"] = not res["mismatches"]
    return res


def post_check(model, tok, prompts, outs, n_tok, k=10):
    """Re-generate k random outputs from the full run one-by-one and compare."""
    rng = random.Random(SEED + 1)
    idx = rng.sample(range(len(prompts)), k)
    bad = [dict(i=i, batched=outs[i], unbatched=g) for i in idx
           if (g := generate(model, tok, prompts[i], max_new_tokens=n_tok)) != outs[i]]
    return {"n": k, "mismatches": bad}


def run_model(key, dev, frames, run):
    set_seed()
    t0 = time.time()
    model, tok = load(key)
    rec = {"load_s": round(time.time() - t0, 1)}
    rec["batch_check"] = batch_check(model, tok, dev)
    bs_o, bs_p = (BS_OBJ, BS_PRED) if rec["batch_check"]["equal"] else (1, 1)
    rec["batch_sizes"] = {"object": bs_o, "prediction": bs_p}
    print(f"[{key}] batch check equal={rec['batch_check']['equal']}; bs={bs_o}/{bs_p}", flush=True)

    t1 = time.time()
    obj = generate_batch(model, tok, list(dev.object_prompt), batch_size=bs_o, max_new_tokens=OBJ_TOK)
    jobs = pred_jobs(dev, frames)
    preds = generate_batch(model, tok, [j[3] for j in jobs], batch_size=bs_p, max_new_tokens=PRED_TOK)
    rec["gen_s"] = round(time.time() - t1, 1)
    rec["n_generations"] = len(obj) + len(preds)
    if bs_o > 1:
        rec["post_check_object"] = post_check(model, tok, list(dev.object_prompt), obj, OBJ_TOK)
        rec["post_check_prediction"] = post_check(model, tok, [j[3] for j in jobs], preds, PRED_TOK)
        if rec["post_check_object"]["mismatches"] or rec["post_check_prediction"]["mismatches"]:
            # Batched != unbatched somewhere in the full run: discard and regenerate everything one by one.
            print(f"[{key}] post-check mismatch -> regenerating with batch size 1", flush=True)
            rec["batch_sizes"] = {"object": 1, "prediction": 1, "note": "fallback after post-check mismatch"}
            t1 = time.time()
            obj = generate_batch(model, tok, list(dev.object_prompt), batch_size=1, max_new_tokens=OBJ_TOK)
            preds = generate_batch(model, tok, [j[3] for j in jobs], batch_size=1, max_new_tokens=PRED_TOK)
            rec["gen_s"] = round(time.time() - t1, 1)
    print(f"[{key}] {rec['n_generations']} generations in {rec['gen_s']}s", flush=True)
    if key == "A":
        rec["perplexity"] = perplexity(model, tok)
        print(f"[A] perplexity {rec['perplexity']['ppl']:.3f}", flush=True)
    rec["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    run[key] = rec
    return dict(zip(dev.id, obj)), {(i, p, fr): (pp, g) for (i, p, fr, pp), g in zip(jobs, preds)}


@torch.no_grad()
def perplexity(model, tok, n_win=40, win=256, bs=4):
    text = (ROOT / "data" / "pg1342.txt").read_text(encoding="utf-8")
    start = text.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.index("It is a truth universally acknowledged", start)  # first line of Chapter I
    ids = tok(text[start:], add_special_tokens=False, return_tensors="pt")["input_ids"][0][: n_win * win]
    assert len(ids) == n_win * win
    wins = ids.view(n_win, win).to("cuda")
    nll, n = 0.0, 0
    for i in range(0, n_win, bs):
        x = wins[i:i + bs]
        logp = torch.log_softmax(model(input_ids=x).logits[:, :-1].float(), -1)
        tgt = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        nll -= tgt.sum().item()
        n += tgt.numel()
    return {"windows": n_win, "window_tokens": win, "scored_tokens": n, "mean_nll": nll / n,
            "ppl": float(np.exp(nll / n)), "start": "first line of Chapter I, after the Gutenberg header"}


def main():
    t0 = time.time()
    dev = dev_items()
    run = {"split": "dev", "n_items": len(dev), "format_wording": FIXED_WORDING,
           "models": {k: MODELS[k]["name"] for k in ("A", "B")}}
    a_obj, a_pred = run_model("A", dev, A_FRAMES, run)
    b_obj, b_pred = run_model("B", dev, ["S1"], run)

    rows = []
    for r in dev.itertuples():
        lo, hi = (r.lo or None), (r.hi or None)
        for p in r.properties.split("|"):
            row = dict(id=r.id, category=r.category, subcategory=r.subcategory, prop=p, object_prompt=r.object_prompt,
                       gold=r.gold, aliases=r.aliases)
            for m, objs in (("A", a_obj), ("B", b_obj)):
                ans = objs[r.id]
                row[f"{m}_obj_raw"] = ans
                row[f"{m}_obj_norm"] = P.normalise(ans)
                row[f"{m}_target"] = P.object_property(p, ans, lo, hi)
                row[f"{m}_noncompliant"] = r.answer_type == "word" and P.noncompliant(ans)
                row[f"{m}_gold_correct"] = P.gold_match(ans, r.gold, r.aliases)
            for m, frames, preds in (("A", A_FRAMES, a_pred), ("B", ["S1"], b_pred)):
                for fr in frames:
                    pp, g = preds[(r.id, p, fr)]
                    row[f"{m}_{fr}_prompt"] = pp
                    row[f"{m}_{fr}_raw"] = g
                    row[f"{m}_{fr}_pred"] = P.parse_prediction(p, g)
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "stage2_dev_raw.csv", index=False)
    run["wall_clock_s"] = round(time.time() - t0, 1)
    (RESULTS / "stage2_dev_run.json").write_text(json.dumps(run, indent=2, default=str))
    print(f"wrote {len(raw)} rows; total {run['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
