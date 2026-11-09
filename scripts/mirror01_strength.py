"""MIRROR-01 step 2: strength matching, on MIRROR-00 data only (no MIRROR-01 item is run here).

m(d) = mean |h . d| at the output of block 12, over every position processed while A generates its S1 first_letter
predictions (prefill + cached decoding, greedy, max 8 new tokens, batch size 1) for the MIRROR-00 dev-fit items
(results/stage3_split.json). All directions are measured in one unhooked pass: measure-only hooks (alpha = 0) leave
the output untouched. alpha_r = 2 * m(d_L) / m(d_r), capped at 20.

Directions: d_L and d_you at block 12 from results/stage3_directions.pt; 20 random unit-normalised Gaussian
directions in the residual dimension, seeds 1001-1020 (stage4_interventions.random_dir, a torch.Generator seed).
Output: results/mirror01/strength_matching.json.
"""
import gc
import json
import time

import pandas as pd
import torch

from common import RESULTS, ROOT, Ablate, generate_batch, load, set_seed
from prompts import prediction_prompt
from stage2_baseline import PRED_TOK
from stage4_interventions import random_dir

OUT = RESULTS / "mirror01"
L, ALPHA, CAP = 12, 2.0, 20.0
RANDOM_SEEDS = list(range(1001, 1021))


def directions(hidden=None):
    sel = json.loads((RESULTS / "stage3_selection.json").read_text())
    assert (sel["layer"], sel["alpha"]) == (L, ALPHA), "frozen setting changed"
    D = torch.load(RESULTS / "stage3_directions.pt", weights_only=True)
    hidden = D["d_L"].shape[1]
    dirs = {"d_L": D["d_L"][L], "d_you": D["d_you"][L]}
    dirs.update({f"random_{s}": random_dir(s, hidden) for s in RANDOM_SEEDS})
    return dirs


def main():
    t0 = time.time()
    split = json.loads((RESULTS / "stage3_split.json").read_text())
    fit = split["dev_fit"]
    ds = pd.read_csv(ROOT / "data" / "dataset.csv", keep_default_na=False).set_index("id")
    prompts = [prediction_prompt("S1", ds.loc[i, "object_prompt"], "first_letter") for i in fit]
    dirs = directions()
    set_seed()
    model, tok = load("A")
    hooks = {k: Ablate(model, L, d, 0.0) for k, d in dirs.items()}
    try:
        outs = generate_batch(model, tok, prompts, batch_size=1, max_new_tokens=PRED_TOK)
        m = {k: h.mean_abs_proj() for k, h in hooks.items()}
        n_pos = {k: h.n_pos for k, h in hooks.items()}
    finally:
        for h in hooks.values():
            h.remove()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    # Sanity: measure-only hooks must not change the outputs (compare with Stage 2 dev raw S1).
    raw = pd.read_csv(RESULTS / "stage2_dev_raw.csv", keep_default_na=False)
    raw = raw[raw.prop == "first_letter"].set_index("id")
    same = sum(o == raw.loc[i, "A_S1_raw"] for i, o in zip(fit, outs))
    alphas = {}
    for k in dirs:
        if k == "d_L":
            continue
        a = ALPHA * m["d_L"] / m[k]
        alphas[k] = {"alpha_uncapped": a, "alpha": min(a, CAP), "capped": bool(a > CAP)}
    res = {"layer": L, "alpha_d_L": ALPHA, "cap": CAP, "n_dev_fit_prompts": len(fit),
           "measured_on": "MIRROR-00 dev-fit S1 first_letter prediction prompts; all positions processed during "
                          "greedy generation (prefill + cached decoding, max 8 new tokens), batch size 1",
           "random_seeds": RANDOM_SEEDS, "random_dir": "torch.Generator().manual_seed(s); randn(hidden); unit norm",
           "n_positions": n_pos["d_L"], "outputs_identical_to_stage2_S1": f"{same}/{len(fit)}",
           "m": m, "alpha_matched": alphas,
           "n_capped": int(sum(v["capped"] for v in alphas.values())),
           "wall_clock_s": round(time.time() - t0, 1),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strength_matching.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
