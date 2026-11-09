"""MIRROR-00 Stage 1: environment record + tiny smoke test. No experimental data is produced here."""
import gc
import json
import platform
import subprocess
import sys
import time

import torch
import transformers

from common import MODELS, RESULTS, Ablate, Capture, decoder_layers, generate, load, set_seed, chat_ids

PROMPTS = {
    "animal": "Name an animal. Answer with one word only.",
    "integer": "Choose an integer between 1 and 50. Answer with the number only.",
    "factual": "What is the capital of Australia? Answer with one word only.",
    "completion": "Complete the sentence with one word: The opposite of hot is",
    "self_pred": ("Suppose you were asked: 'Name an animal. Answer with one word only.' "
                  "What would be the first letter of your answer? Answer with a single letter only."),
    "other_pred": ("Suppose Gemma, a different language model, were asked: 'Name an animal. Answer with one word only.' "
                   "What would be the first letter of its answer? Answer with a single letter only."),
}


def env_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    import accelerate, numpy
    return {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__, "accelerate": accelerate.__version__, "numpy": numpy.__version__,
            "gpu": smi, "cuda_available": torch.cuda.is_available()}


def gpu_check():
    set_seed()
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)
    ref = a @ b
    got = (a.cuda() @ b.cuda()).cpu()
    return {"max_abs_err_vs_cpu": float((ref - got).abs().max()), "ok": bool(torch.allclose(ref, got, atol=1e-2))}


def run_model(key, hook_tests):
    set_seed()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = load(key)
    rec = {"model": MODELS[key]["name"], "hf_id": MODELS[key]["hf_id"], "revision": MODELS[key]["revision"],
           "class": type(model).__name__, "load_s": round(time.time() - t0, 1),
           "dtype": str(next(model.parameters()).dtype),
           "vram_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
           "n_layers": len(decoder_layers(model))}
    ids, _ = chat_ids(tok, "X")
    rec["chat_template_rendered"] = tok.decode(ids[0])
    outs = {k: generate(model, tok, p) for k, p in PROMPTS.items()}
    rec["outputs"] = outs
    rec["deterministic_repeat"] = all(generate(model, tok, p) == outs[k] for k, p in PROMPTS.items())
    rec["cache_vs_nocache_equal"] = all(generate(model, tok, p, use_cache=False) == outs[k] for k, p in PROMPTS.items())

    if hook_tests:
        L = rec["n_layers"] // 2
        # (a) capture last-token residuals for the self/other pair at every layer
        cap = Capture(model)
        with torch.no_grad():
            model(chat_ids(tok, PROMPTS["self_pred"])[0])
            a_self = dict(cap.acts)
            model(chat_ids(tok, PROMPTS["other_pred"])[0])
            a_oth = dict(cap.acts)
        cap.remove()
        diff = {i: float((a_self[i] - a_oth[i]).norm()) for i in a_self}
        rec["hook"] = {"layer": L, "hidden": int(a_self[L].numel()),
                       "self_minus_other_norm_by_layer": {i: round(v, 3) for i, v in diff.items()},
                       "resid_norm_by_layer": {i: round(float(a_self[i].norm()), 2) for i in a_self}}
        # (b) a random unit direction is really removed: capture registered AFTER ablation sees projection ~0
        g = torch.Generator().manual_seed(1)
        d = torch.randn(a_self[L].numel(), generator=g)
        d = d / d.norm()
        ab = Ablate(model, L, d)
        cap = Capture(model, [L])
        with torch.no_grad():
            model(chat_ids(tok, PROMPTS["self_pred"])[0])
        cap.remove()
        proj_before = float(a_self[L] @ d)
        proj_after = float(cap.acts[L] @ d)
        rand_outs = {k: generate(model, tok, p) for k, p in PROMPTS.items()}
        rand_nc = {k: generate(model, tok, p, use_cache=False) for k, p in PROMPTS.items()}
        ab.remove()
        # (c) a large direction (the residual itself) must visibly change behaviour -> proves the hook is live
        ab = Ablate(model, L, a_self[L])
        big_outs = {k: generate(model, tok, p) for k, p in PROMPTS.items()}
        big_nc = {k: generate(model, tok, p, use_cache=False) for k, p in PROMPTS.items()}
        ab.remove()
        after_removal = {k: generate(model, tok, p) for k, p in PROMPTS.items()}
        rec["hook"].update({
            "random_dir_proj_before": round(proj_before, 4), "random_dir_proj_after": round(proj_after, 4),
            "random_ablation_outputs": rand_outs,
            "random_ablation_changed_n": sum(rand_outs[k] != outs[k] for k in outs),
            "random_ablation_cache_vs_nocache_equal": rand_outs == rand_nc,
            "resid_dir_ablation_outputs": big_outs,
            "resid_dir_ablation_changed_n": sum(big_outs[k] != outs[k] for k in outs),
            "resid_dir_ablation_cache_vs_nocache_equal": big_outs == big_nc,
            "hooks_cleanly_removed": after_removal == outs,
        })
    rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "B"
    res = {"env": env_info(), "gpu_check": gpu_check(), "gen_settings": "greedy, max_new_tokens=24, enable_thinking=False",
           "A": run_model("A", hook_tests=True)}
    try:
        res["B"] = run_model(which, hook_tests=False)
    except Exception as e:  # report, then try the fallback
        res["B_error"] = repr(e)
        res["B"] = run_model("B_fallback", hook_tests=False)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stage1_smoke.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
