"""Shared helpers for MIRROR-00: model loading, chat formatting, greedy generation, residual hooks.

Hook placement (documented for the protocol):
  We hook the OUTPUT of decoder block `L` (`model.model.layers[L]`), i.e. the post-block residual
  stream after block L has added its attention and MLP contributions. Projection removal
  h <- h - (h . d) d is applied at EVERY position the block processes: all prompt tokens during
  prefill and each new token during cached decoding. Because the ablated residual feeds all later
  blocks, the K/V cache of later blocks is computed from ablated states, so cached and uncached
  generation are equivalent (verified in stage1_smoke.py).
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 0

# Machine-specific model locations live in local_paths.json (gitignored) or MIRROR_* env vars.
_LOCAL = json.loads((ROOT / "local_paths.json").read_text()) if (ROOT / "local_paths.json").exists() else {}


def _path(var, default):
    return Path(os.environ.get(var) or _LOCAL.get(var) or ROOT / "models" / default)

MODELS = {
    "A": {"name": "Qwen3-4B-Instruct-2507", "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
          "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
          "path": ROOT / "models" / "qwen3-4b-instruct-2507"},
    # May be a read-only copy owned by another project; never modified.
    "B": {"name": "Gemma-4-E4B-it", "hf_id": "google/gemma-4-E4B-it",
          "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c (weights sha256 match)",
          "path": _path("MIRROR_GEMMA4", "gemma-4")},
    "B_fallback": {"name": "Gemma-3-4B-it", "hf_id": "google/gemma-3-4b-it", "revision": "unrecorded (see sha256)",
                   "path": _path("MIRROR_GEMMA3", "gemma-3-4b-it")},
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load(key):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    path = MODELS[key]["path"]
    cfg = AutoConfig.from_pretrained(path)
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in cfg.architectures[0] else AutoModelForCausalLM
    model = cls.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def decoder_layers(model):
    """The text decoder's ModuleList of blocks."""
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) > 4 and hasattr(m[0], "mlp")
             and (n.endswith("model.layers") or n.endswith("language_model.layers"))]
    cands.sort(key=lambda nm: "language" not in nm[0])
    return cands[0][1]


def chat_ids(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True)
    return ids["input_ids"].to("cuda"), ids["attention_mask"].to("cuda")


GEN_KW = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=24)


@torch.no_grad()
def generate(model, tok, user, system=None, use_cache=True, **kw):
    ids, mask = chat_ids(tok, user, system)
    out = model.generate(input_ids=ids, attention_mask=mask, use_cache=use_cache,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id, **{**GEN_KW, **kw})
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_batch(model, tok, users, batch_size=16, **kw):
    """Greedy cached generation for a list of user turns, left-padded batches.

    Each prompt is tokenised exactly as in `chat_ids`, then left-padded by hand, so the unpadded token ids are
    identical to the single-prompt path. Equality of batched and unbatched outputs is checked in Stage 2.
    """
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    outs = []
    for i in range(0, len(users), batch_size):
        encs = [chat_ids(tok, u)[0][0] for u in users[i:i + batch_size]]
        L = max(len(e) for e in encs)
        ids = torch.full((len(encs), L), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros((len(encs), L), dtype=torch.long, device="cuda")
        for j, e in enumerate(encs):
            ids[j, L - len(e):] = e
            mask[j, L - len(e):] = 1
        out = model.generate(input_ids=ids, attention_mask=mask, use_cache=True, pad_token_id=pad,
                             **{**GEN_KW, **kw})
        outs += [tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]
    return outs


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class Ablate:
    """Remove the component along unit direction `d` from the output of decoder block `layer`.

    h <- h - alpha (h . d) d. alpha = 1 (default) is full projection removal, with exactly the Stage 1-2 arithmetic;
    alpha = 2 reflects the component; alpha = 0 is measure-only (the output is left untouched).
    Every call records |h . d| (before the edit) at every processed position: `mean_abs_proj()` is the mean over
    all positions since the last `reset_stats()`; the magnitude actually removed per position is alpha * |h . d|.
    """

    def __init__(self, model, layer, d, alpha=1.0):
        self.d = (d / d.norm()).to("cuda", torch.float32)
        self.alpha = float(alpha)
        self.reset_stats()
        blk = decoder_layers(model)[layer]
        self.h = blk.register_forward_hook(self.hook)

    def reset_stats(self):
        self.abs_sum = torch.zeros((), device="cuda", dtype=torch.float64)
        self.n_pos = 0

    def mean_abs_proj(self):
        return float(self.abs_sum) / self.n_pos if self.n_pos else None

    def hook(self, mod, inp, out):
        h = _hs(out)
        hf = h.float()
        proj = hf @ self.d
        self.abs_sum += proj.abs().sum(dtype=torch.float64)
        self.n_pos += proj.numel()
        if self.alpha == 0.0:
            return None
        if self.alpha == 1.0:
            hf = hf - proj.unsqueeze(-1) * self.d  # identical to the original (hf @ d) expression
        else:
            hf = hf - (self.alpha * proj).unsqueeze(-1) * self.d
        return _replace(out, hf.to(h.dtype))

    def remove(self):
        self.h.remove()


class Capture:
    """Record last-input-token post-block residual at each listed layer (prefill only)."""

    def __init__(self, model, layers=None):
        blocks = decoder_layers(model)
        self.acts = {}
        layers = range(len(blocks)) if layers is None else layers
        self.hs = [blocks[i].register_forward_hook(self.make(i)) for i in layers]

    def make(self, i):
        def f(mod, inp, out):
            self.acts[i] = _hs(out)[0, -1].float().cpu()
        return f

    def remove(self):
        for h in self.hs:
            h.remove()
