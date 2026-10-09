# Results

**No model has been benchmarked yet.** The table below is the format the harness will emit, with no values.

| Model | N | Schema % | Gate % | Correct % | TTFT p50 (s) | tok/s | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct Q4_K_M | - | - | - | - | - | - | - |
| Llama-3.2-3B-Instruct Q8_0 | - | - | - | - | - | - | - |
| Mistral-7B-Instruct-v0.3 Q4_K_M | - | - | - | - | - | - | - |
| Qwen2.5-Coder-7B Q4_K_M | - | - | - | - | - | - | - |

## Protocol

Greedy decoding (temperature 0), same 20 prompts, constrained (GBNF) and unconstrained runs reported separately, hardware recorded (CPU model, RAM, GPU). N=20 per cell is small; report counts alongside percentages and do not claim differences below about 2 prompts.

## Verified today

- 17 tests pass: every gate rejection code, injection strings never reach execution, scripted oracle backend scores 100%, corrupted backend scores 0% with 20 logged violations.
- Property test caught a real bug (azimuth rounding to exactly 360.0); fixed.
- `mypy --strict` clean.
