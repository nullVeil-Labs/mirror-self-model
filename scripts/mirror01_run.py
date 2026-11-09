"""MIRROR-01 step 3: generations on the fresh dataset (data/dataset_01.csv).

1. Model B (Gemma, never hooked): object answers. B is then freed.
2. Model A (Qwen): object answers with NO hook at all. Then S1/O1/S2/O2 first_letter predictions under each
   condition, the hook active only while the prediction prompts are processed (interventions.run_condition with
   obj_df=None): baseline (measure-only hook, alpha = 0, records |h . d_L|), d_L (alpha 2), d_you (alpha 2),
   d_you matched, random_1001..1020 matched (alphas read from results/mirror01/strength_matching.json, which must
   exist before this runs). Greedy, cached, bf16, batch size 1.

Outputs (results/mirror01/): objects.csv, predictions.csv (long: id, condition, frame, raw), run.json.
Generations are checkpointed per condition in gens_checkpoint.json so an interrupted run resumes.
"""
import gc
import json
import time

import pandas as pd
import torch

from common import MODELS, RESULTS, ROOT, generate_batch, load, set_seed
from interventions import run_condition
from mirror01_strength import ALPHA, L, RANDOM_SEEDS, directions
from stage2_baseline import OBJ_TOK

OUT = RESULTS / "mirror01"
FRAMES = ("S1", "O1", "S2", "O2")
CKPT = OUT / "gens_checkpoint.json"


def items():
    return pd.read_csv(ROOT / "data" / "dataset_01.csv", keep_default_na=False)


def conditions():
    sm = json.loads((OUT / "strength_matching.json").read_text())
    am = sm["alpha_matched"]
    conds = [("baseline", "d_L", 0.0), ("d_L", "d_L", ALPHA), ("d_you", "d_you", ALPHA),
             ("d_you_matched", "d_you", am["d_you"]["alpha"])]
    conds += [(f"random_{s}", f"random_{s}", am[f"random_{s}"]["alpha"]) for s in RANDOM_SEEDS]
    return conds


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    t_start = time.time()
    df = items()
    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {"objects": {}, "preds": {}, "rec": {}}
    run = {"n_items": len(df), "layer": L, "models": {k: MODELS[k]["name"] for k in ("A", "B")}, "batch_size": 1,
           "hook_scope": "prediction prompts only (prefill + decoding); object answers unhooked"}

    for key in ("B", "A"):
        if key in ck["objects"] and (key == "B" or len(ck["preds"]) == len(conditions())):
            continue
        set_seed()
        t0 = time.time()
        model, tok = load(key)
        if key not in ck["objects"]:
            t1 = time.time()
            ck["objects"][key] = dict(zip(df.id, generate_batch(model, tok, list(df.object_prompt), batch_size=1,
                                                                 max_new_tokens=OBJ_TOK)))
            ck["rec"][f"{key}_objects_s"] = round(time.time() - t1, 1)
            CKPT.write_text(json.dumps(ck))
            print(f"[{key}] objects done in {ck['rec'][f'{key}_objects_s']}s", flush=True)
        if key == "A":
            dirs = directions()
            for name, dname, alpha in conditions():
                if name in ck["preds"]:
                    continue
                _, preds, rec = run_condition(model, tok, None, df, frames=FRAMES, d=dirs[dname], layer=L,
                                              alpha=alpha, ppl=False)
                rec.update(direction=dname, alpha=alpha,
                           mean_abs_removed=alpha * rec["mean_abs_proj_gen"])
                ck["preds"][name] = preds
                ck["rec"][name] = rec
                CKPT.write_text(json.dumps(ck))
                print(f"{name}: alpha {alpha:.3f} [{rec['gen_s']}s] |h.d| {rec['mean_abs_proj_gen']:.3f} "
                      f"removed {rec['mean_abs_removed']:.3f}", flush=True)
        ck["rec"][f"{key}_vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        ck["rec"][f"{key}_session_s"] = round(time.time() - t0, 1)
        CKPT.write_text(json.dumps(ck))
        del model, tok
        free()

    obj = pd.DataFrame({"id": df.id, "A_obj": df.id.map(ck["objects"]["A"]), "B_obj": df.id.map(ck["objects"]["B"])})
    obj.to_csv(OUT / "objects.csv", index=False)
    rows = [dict(id=i, condition=c, frame=f, raw=v) for c, p in ck["preds"].items() for f, d in p.items()
            for i, v in d.items()]
    pd.DataFrame(rows).to_csv(OUT / "predictions.csv", index=False)
    run["conditions"] = {c: ck["rec"][c] for c in ck["preds"]}
    run["timing"] = {k: v for k, v in ck["rec"].items() if k not in ck["preds"]}
    run["n_generations"] = 2 * len(df) + len(rows)
    run["wall_clock_this_invocation_s"] = round(time.time() - t_start, 1)
    (OUT / "run.json").write_text(json.dumps(run, indent=2))
    print(f"done: {run['n_generations']} generations, {run['wall_clock_this_invocation_s']}s")


if __name__ == "__main__":
    main()
