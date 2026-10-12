# Results

Four public quantized models, each run through the same 20-prompt tool-calling battery on CPU (llama.cpp) and GPU (Ollama), constrained and unconstrained: 16 configurations. All numbers below come from `results/real/*.json` and are rebuilt by `python scripts/diagnose.py`.

## Setup

| | |
| --- | --- |
| Machine | 12th Gen Intel Core i9-12950HX (16 physical / 24 logical cores), 63.7 GB RAM, NVIDIA RTX A4500 Laptop GPU (16 GB), Windows 11 |
| Software | llama-cpp-python 0.3.36 (prebuilt CPU wheel), Ollama 0.34.2, Python 3.12 |
| Weights | the **same GGUF files** on both backends: the Ollama blobs for `llama3.2:3b-instruct-q4_K_M`, `llama3.2:3b-instruct-q8_0`, `qwen2.5-coder:7b-instruct-q4_K_M`, `mistral:7b-instruct-v0.3-q4_K_M` |
| Battery | 20 fixed prompts, 4 registered tools: sensor status (5), azimuth (5), distance (4), battery runtime (4), prompt injection (2). System prompt and expected calls are in `src/edge_slm_eval/battery.py` |
| Decoding | greedy (temperature 0), at most 96 new tokens, zero-shot |
| Constrained | llama.cpp: a GBNF grammar forcing `{"call": <string>}`. Ollama: a JSON schema in the `format` field. These are different implementations of the same idea |
| Runs | one discarded warm-up call, then 3 passes of the 20 prompts. Correctness is taken from the first pass; timing is pooled over all 60 calls |
| Timing | TTFT is the median time to the first token. Decode tok/s is tokens after the first divided by time after the first token |

"Correct" means the output passed the schema gate, passed the AST gate **and** parsed to exactly the expected call. A call that is semantically right but written with keyword arguments is rejected by the gate (positional literals only) and counts as wrong in the headline column.

## Results (N = 20 prompts per row)

| Model | Device | Mode | Schema % | Correct % (strict gate) | Correct % if keyword args accepted | TTFT p50 (s) | Decode tok/s | GPU clock, median MHz | Memory (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.2-3b-instruct-q4_K_M | cpu (llama.cpp) | constrained | 100 | 65 | 100 | 0.263 | 10.8 | - | 3671 |
| llama3.2-3b-instruct-q4_K_M | cpu (llama.cpp) | unconstrained | 100 | 65 | 100 | 0.244 | 20.8 | - | 3663 |
| llama3.2-3b-instruct-q4_K_M | gpu (ollama) | constrained | 100 | 60 | 95 | 0.056 | 126.4 | 1200 | 2436 |
| llama3.2-3b-instruct-q4_K_M | gpu (ollama) | unconstrained | 100 | 60 | 95 | 0.057 | 134.5 | 1095 | 2436 |
| llama3.2-3b-instruct-q8_0 | cpu (llama.cpp) | constrained | 100 | 60 | 70 | 0.562 | 8.7 | - | 3706 |
| llama3.2-3b-instruct-q8_0 | cpu (llama.cpp) | unconstrained | 75 | 60 | 70 | 0.554 | 14.0 | - | 3697 |
| llama3.2-3b-instruct-q8_0 | gpu (ollama) | constrained | 100 | 50 | 65 | 0.054 | 101.8 | 1170 | 3774 |
| llama3.2-3b-instruct-q8_0 | gpu (ollama) | unconstrained | 75 | 50 | 65 | 0.061 | 103.8 | 1418 | 3774 |
| mistral-7b-instruct-v0.3-q4_K_M | cpu (llama.cpp) | constrained | 100 | 95 | 95 | 0.629 | 8.2 | - | 7696 |
| mistral-7b-instruct-v0.3-q4_K_M | cpu (llama.cpp) | unconstrained | 95 | 90 | 90 | 0.698 | 8.8 | - | 7690 |
| mistral-7b-instruct-v0.3-q4_K_M | gpu (ollama) | constrained | 100 | 95 | 95 | 0.068 | 64.2 | 952 | 4722 |
| mistral-7b-instruct-v0.3-q4_K_M | gpu (ollama) | unconstrained | 95 | 90 | 90 | 0.059 | 63.6 | 945 | 4722 |
| qwen2.5-coder-7b-instruct-q4_K_M | cpu (llama.cpp) | constrained | 100 | 100 | 100 | 0.639 | 6.3 | - | 7511 |
| qwen2.5-coder-7b-instruct-q4_K_M | cpu (llama.cpp) | unconstrained | 60 | 60 | 60 | 0.620 | 10.0 | - | 7503 |
| qwen2.5-coder-7b-instruct-q4_K_M | gpu (ollama) | constrained | 100 | 100 | 100 | 0.067 | 61.4 | 840 | 4528 |
| qwen2.5-coder-7b-instruct-q4_K_M | gpu (ollama) | unconstrained | 65 | 65 | 65 | 0.069 | 62.6 | 878 | 4528 |

**Read the columns with care.**
- *Correct % if keyword args accepted* is a **post-hoc diagnostic** computed from the saved raw outputs by `scripts/diagnose.py`. It relaxes only the keyword-argument rule (arguments are matched to the real parameter names and reordered); the JSON schema, the tool registry, literal-only arguments and types all still apply. It does not change the gate and is not the headline.
- *Memory* means different things: for CPU rows it is the peak resident memory of the Python process (weights included); for GPU rows it is the loaded-model size that Ollama reports, which is mostly GPU memory.
- *GPU clock* is the median streaming-multiprocessor clock while the GPU was busy (maximum 1785 MHz). See [the throttling note](#gpu-throttling).

With 20 prompts per row, a 95% interval on any percentage is roughly ±20 points (65% is about 43-82%). Differences of one to three prompts, such as the CPU-versus-GPU gaps for the same Llama weights, are within that noise.

## Findings

**1. Constrained decoding raised format adherence and never lowered it.** Qwen2.5-Coder-7B went from 60% (CPU) and 65% (GPU) valid output unconstrained to 100% constrained. Llama 3.2 3B Q8 went from 75% to 100%, and Mistral from 95% to 100%. Llama 3.2 3B Q4 was already at 100%. For Qwen all the unconstrained failures are `SCHEMA` failures of two kinds: most (6 of 8 on CPU, 5 of 7 on GPU) have an extra closing brace, `{"call": "..."}}`, and two (on the injection prompts) are wrapped in a markdown code fence.

**2. A valid JSON wrapper is not a valid call.** Llama 3.2 3B Q8 scored 100% schema adherence constrained but only 60% (CPU) and 50% (GPU) correct. The model tries to write `get_sensor_status(sensor_id="thermal-01")` inside a JSON string without escaping the inner quotes; the grammar then closes the string at the first inner quote, producing `get_sensor_status(sensor_id=` and nothing more. Grammar decoding guaranteed the outer structure and turned an escaping mistake into a truncated call. Checking the inner call needs its own gate, which is what the AST check does.

**3. One strict rule explains much of the Llama result.** Llama 3.2 3B Q4 writes `get_sensor_status(sensor_id='thermal-01')` with a keyword argument instead of the positional form the system prompt asks for. The gate rejects it. Accepting keyword arguments would lift Llama 3B Q4 from 65% to 100% (CPU) and from 60% to 95% (GPU). By category, Llama gets 0/5 on the sensor-status calls but 5/5 on azimuth and 4/4 on distance, so its failures come entirely from how it writes a string argument. "Follows the stated positional format" and "means the right call" are different questions, and the table reports both.

**4. Under the strict gate, the 7B models beat the 3B models.** Constrained, strict: Qwen2.5-Coder-7B 100% (both devices), Mistral-7B 95% (both), Llama 3B Q4 60-65%, Llama 3B Q8 50-60%. The 3B Q8 did not beat the 3B Q4 here (and lost more when keyword arguments were accepted, 65-70% against 95-100%). That is six or seven prompts, so it is not noise, but it is one battery and one prompt, so it is not evidence that Q8 is generally worse.

**5. Prompt injection never reached execution.** Across all 320 first-pass outputs, nothing outside the four registered tools was executed. Four outputs tried: Llama 3B Q4 on the GPU emitted the injected `os.system('ls')` in both modes (rejected `NOT_A_CALL`), and Mistral, unconstrained, appended a `delete_logs()` follow-up to its legitimate call on both devices (rejected `SCHEMA`, extra data). The other injection failures are the same keyword-argument rejections as above, not obedience to the injected text.

**6. Speed.**
- *CPU:* decode 6-21 tok/s; TTFT 0.24-0.70 s. The 3B Q4 is fastest (10.8 constrained, 20.8 unconstrained), the Qwen 7B slowest constrained (6.3).
- *GPU:* decode about 100-135 tok/s for the 3B models and 61-64 tok/s for the 7B models; TTFT 0.05-0.07 s. That is roughly an order of magnitude over CPU, but see the clock note: these are throttled-GPU numbers.
- *Grammar cost depends on the model on CPU.* Constrained decoding was much slower than unconstrained for Llama 3.2 (128k-token vocabulary: 10.8 vs 20.8 tok/s, and 8.7 vs 14.0) and Qwen2.5 (152k: 6.3 vs 10.0) but barely slower for Mistral (32k: 8.2 vs 8.8). That is consistent with the grammar mask costing more as the vocabulary grows; I did not test that directly.

## GPU throttling

The laptop was on AC power, but the RTX A4500 was power-capped (`sw_power_cap`) in every GPU run, with clocks of 840-1418 MHz against a 1785 MHz maximum, and in one run thermally slowed to 210 MHz. Every GPU run recorded clocks, temperature and throttle reasons (in each JSON under `gpu_state_during_run`), and each started after waiting for the GPU to reach 60 °C. Consequences:
- **GPU tokens per second are not hardware limits.** They depend on the clock state during the run (the 3B rows ran at 1095-1418 MHz, the 7B rows at 840-952 MHz), so a 3B-versus-7B GPU comparison is partly a clock comparison.
- The accuracy, schema and gate columns are unaffected by clocks.
- CPU timings were taken with an unrelated process occasionally using about one logical core; I saw it once and did not track it.

## Differences between the CPU and GPU rows

The two backends load identical weights but are not identical systems: they use different chat templates, different implementations of constrained decoding, and a different default context size (2,048 against 4,096). So a CPU-versus-GPU gap in *correctness* (Llama 3B: 65 vs 60 and 60 vs 50) is within the noise above and mixes template effects with anything else; the gap in *speed* is mostly hardware. Mistral's output was non-deterministic across passes on the GPU in constrained mode (greedy decoding gave a different answer to at least one prompt in a later pass); every other configuration was identical across all three passes.

## What went wrong along the way, and was fixed

1. **Mistral on CPU first scored 0%.** Its embedded chat template silently drops the system role, so the model never saw the instructions (it replied "I don't have real-time access…"). Fixed with a probe that detects this and folds the system text into the user message; Mistral now scores the same on CPU and GPU. Only Mistral is affected among these four.
2. **The first GPU pass had unexplained speeds** (118 tok/s for the 3B model, then about 10 tok/s for 7B models). The cause was GPU throttling. I added clock telemetry and a cool-down gate and re-ran all eight GPU configurations, then re-ran one configuration (Llama 3B Q4 constrained) that had started at 210 MHz. The superseded files are kept in `results/real_run1_superseded/`.
3. **A downloads problem**: this machine's DNS dropped several model pulls; they were retried until complete. It did not affect results.

## Not done

- A battery larger than 20 prompts, more tools, long contexts, and multi-turn tool use (context degradation under recursive calls).
- Few-shot or format-example prompts, which would probably fix the keyword-argument and escaping habits and are deliberately excluded from this zero-shot protocol.
- Q8 versus Q4 for the 7B models, and other quantization levels.
- A GPU run at a controlled, fixed clock; vLLM; a batch-size or concurrency study; measured energy use.
- Statistical uncertainty beyond the interval above (the runs are deterministic, so repeats add timing samples, not accuracy samples).

## Verified

- 23 tests (gate rejection codes, scripted backends, a fake Ollama server for the streaming and timing code, and the keyword-normalisation diagnostic); `mypy --strict` and ruff clean.
- The Docker image (CPU build, prebuilt llama-cpp wheel) builds, and the 23 tests pass inside it with `--network none`.

`results/real/results_table.md` (built by `scripts/diagnose.py`) is the table used on this page. `results/real/table.md` is the raw table that `scripts/run_real.py` writes after each run; it lacks the keyword-argument column.
