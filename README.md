# edge-slm-eval

Air-gapped evaluation harness for quantized small language models on disconnected or tactical edge nodes. It measures structured-output adherence, tool-call correctness, time-to-first-token, throughput and peak memory, with a deterministic AST gate between model text and anything that executes.

**Status:** v0.1 scaffold. The gate, 20-prompt battery, harness and report writer are tested against scripted backends. **No real GGUF model has been run yet**, so there are no model numbers (see [docs/RESULTS.md](docs/RESULTS.md)).

## Design rule

Model output is data. It reaches execution only if it (1) is JSON matching a strict schema (`{"call": "..."}`, extra keys forbidden) and (2) parses with `ast` as exactly one call to a registered function with positional literal arguments of the registered types. No `eval`/`exec`; keyword args, attribute access, nested calls and non-literals are rejected. Every rejection is logged with a code (`SCHEMA`, `NOT_A_CALL`, `UNKNOWN_TOOL`, `NON_LITERAL`, `ARITY`, `ARG_TYPE`, ...).

## Use

```bash
uv pip install -e ".[llama,dev]"
edge-slm-eval --model models/llama-3.2-3b-instruct-q4_k_m.gguf --out results/
edge-slm-eval --model ... --no-grammar     # compare constrained (GBNF) vs unconstrained decoding
```

Offline container (weights mounted read-only, no network):

```bash
docker build -t edge-slm-eval .
docker run --rm --network none -v "$PWD/models:/models:ro" edge-slm-eval \
    edge-slm-eval --model /models/model.gguf --out /tmp/results
```

## Battery

20 fixed prompts: sensor status (5), azimuth (5), distance (4), battery runtime (4) and 2 prompt-injection cases whose correct behaviour is still a single registered call. Expected calls are checked after parsing, so `1.0` and `1` compare equal.

## Checks

```bash
pytest          # unit + hypothesis (arbitrary call text never executes; azimuth always in [0, 360))
mypy src tests  # strict
```

## Not yet done

Context-window degradation under recursive multi-turn tool calls, multi-quantization matrix (Q4_K_M vs Q8), vLLM backend, CPU vs single-GPU comparison.
