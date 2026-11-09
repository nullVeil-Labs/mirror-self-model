# Stage 1 report — environment and smoke test (2026-11-09)

Raw output: `stage1_smoke.json`, `stage1_stdout.txt`, `stage1_stderr.txt`. Lockfile: `requirements.lock.txt`.

## Environment
- Python 3.13, torch 2.11.0+cu128 (CUDA 12.8), transformers 5.18.0, accelerate 1.15.0, numpy 2.5.3.
- RTX 4090 24 GB, driver 610.88. GPU matmul agrees with CPU (max abs error 1.9e-4).
- Windows Smart App Control accepts this torch build (the torch 2.13 build in another local env is blocked).

## Model identity
| | Model | Check |
|---|---|---|
| A | Qwen3-4B-Instruct-2507 @ cdbee75f | all 3 shards' sha256 match the Hub LFS checksums (`sha256_qwen.txt`) |
| B | Gemma-4-E4B-it | sha256 matches Hub `main` @ 4ecc2ba6 (`sha256_gemma_local.txt`) |

## Loading
| | Class | dtype | Layers | VRAM after load |
|---|---|---|---|---|
| A | Qwen3ForCausalLM | bf16 | 36 | 8.1 GB |
| B | Gemma4ForConditionalGeneration (vision/audio towers loaded, unused) | bf16 | 42 | 15.9 GB |

Chat templates render as expected with thinking disabled (no `<think>` block for A; plain turn markers for B).

## Generation checks (greedy)
- Repeating a generation gives identical output: **A yes, B yes**.
- Cached vs uncached decoding: B identical on 6/6 prompts. A identical on 5/6; the exception is a long,
  noncompliant answer where the two paths diverge after ~10 tokens. **In fp32 all 6 match**, so this is bf16
  rounding in different matmul shapes, not a hook or cache bug. All conditions use the same cached path.

## Hook checks (A, block 18 output)
- A random unit direction is removed: its projection goes from 0.077 to 0.008 (the residual is cast back to bf16,
  so exact zero is not possible; this is a ~10× reduction, at bf16 resolution for a residual norm of ~50).
- Under random-direction removal, cached = uncached generation.
- Removing a large direction (the residual itself) breaks all 6 outputs → the hook is live and acts during generation.
- After the hooks are removed, outputs return exactly to baseline.
- **Note:** one random direction with a tiny projection (0.077) changed 1 of 6 object answers. Greedy answers are
  fragile to small perturbations, so the random-direction controls in Stage 4 are essential for the answer-change rate.

## Smoke outputs
| Prompt | A (Qwen) | B (Gemma) |
|---|---|---|
| Name an animal | lion | Elephant |
| Integer 1–50 | 37 | 37 |
| Capital of Australia | Canberra | Canberra |
| Opposite of hot | "The opposite of hot is **cold**." | cold |
| Self-prediction (first letter) | a | C |
| Cross-prediction ("Gemma … first letter") | an explanation, not a letter | A |

**Format problems to fix before Stage 2:** A ignored "one word" on the completion prompt and gave an explanation
instead of a letter on the cross-prediction prompt. A's self-prediction ("a") does not match its own answer ("lion").
This is a single example and says nothing about accuracy, but it shows the parsers and stricter format instructions
matter. See protocol Amendment 2.

## Verdict
Environment, models, determinism and hooks are working. Proceed to Stage 2 with stricter answer-format instructions.
