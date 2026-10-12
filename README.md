# edge-slm-eval — test small language models for offline, tactical-edge use, with a deterministic gate in front of every tool call

![CI](https://github.com/PCSchmidt/edge-slm-eval/actions/workflows/ci.yml/badge.svg)

**Status: v0.2, portfolio project.** The gate, battery, harness and both backends are built and tested, and the battery has been run on four real quantized models on CPU and GPU (16 configurations). The main open items are a bigger battery, few-shot prompts and a fixed-clock GPU run ([Limitations](#limitations)).

```text
$ python examples/demo_gate.py            # what the gate does with model-emitted text (no model involved)
EXECUTE  -> 'OFFLINE'            {"call": "get_sensor_status(\"radar-07\")"}
EXECUTE  -> 51.059               {"call": "azimuth_deg(38.9, -77.0, 40.7, -74.0)"}
REJECT   NOT_A_CALL              {"call": "os.system('rm -rf /')"}
REJECT   NON_LITERAL             {"call": "get_sensor_status(open('/etc/passwd').read())"}
REJECT   ARG_TYPE                {"call": "azimuth_deg(\"north\", 1, 2, 3)"}
REJECT   KEYWORDS                {"call": "get_sensor_status(sensor_id='gps-01')"}
REJECT   SCHEMA                  Sure! The status of radar-07 is OFFLINE.
REJECT   SCHEMA                  {"call": "battery_hours(500, 25)", "note": "extra key"}
```

*Real output of [examples/demo_gate.py](examples/demo_gate.py), lightly trimmed. These are hand-written strings standing in for model output, to show the gate's decisions; they are not results from any model.*

## What is this? (plain-language overview)

More and more AI features are meant to run **on the device**: a laptop in a vehicle, a sensor node, a ruggedised box with no internet. In those places you cannot call a cloud model. You use a *small* language model, shrunk ("quantized") to fit modest memory and run on a CPU or a single GPU. Two questions then matter before you trust it:

1. **Does it still behave?** Shrinking a model can make it worse at following a strict output format, and an agent that calls tools needs that format exactly right every time.
2. **Is it safe to let it act?** Whatever text the model emits, something has to decide whether it is allowed to run.

edge-slm-eval is a test harness that answers both. It feeds a fixed set of 20 tool-calling prompts to a local model and records how often the output is valid, how often it is correct, how long the first token takes, how many tokens per second it produces, and how much memory it uses. The key design point is the **gate**: the model's text never reaches a tool directly. It must first parse as a strict JSON schema and then as exactly one call to a registered function with literal arguments of the right type. Anything else is rejected, logged with a reason code, and never executed.

**What it is not.** It is not an agent framework, a benchmark of general intelligence, or a model server. It measures a narrow thing: *can this local model emit safe, correct tool calls, and at what speed and memory cost*.

### Who it is for, and when it earns its place

| If you are... | the harness gives you... |
| --- | --- |
| choosing a model and quantization for an offline deployment | a like-for-like table: format adherence, correctness, speed and memory per model and quantization level |
| building a local or air-gapped agent | the gate pattern, ready to reuse: schema, then AST check, then execute, with every rejection logged |
| a safety reviewer asking "what can the model make the system do?" | a short, auditable answer: only the four registered pure functions, with literal arguments |
| doing red-team or robustness work | a fixed battery that includes prompt-injection cases whose correct behaviour is still one registered call |

It is a poor fit if you need to evaluate reasoning quality, long-context behaviour or multi-turn planning (not built yet), or if your model is served over a network.

### Why gate with a syntax tree instead of trusting the model, or asking another model?

A model can emit anything, including `os.system('rm -rf /')`. Constrained decoding (a grammar) reduces bad output but is not a guarantee, and a second model checking the first is just more probabilistic output. Parsing the call with Python's `ast` module and checking it against a fixed registry gives a decision that is deterministic, testable and explainable. `eval` and `exec` are never used anywhere.

**What's real and what's not.** The gate, the battery, the harness and the report writer are tested, and four real models (Llama 3.2 3B at Q4_K_M and Q8_0, Mistral 7B v0.3 and Qwen2.5-Coder 7B at Q4_K_M) have been benchmarked on one laptop (i9-12950HX CPU, RTX A4500 GPU). The numbers are real but narrow: 20 prompts, four tools, zero-shot, one machine whose GPU was power-throttled during the runs. Read the results with the caveats in [docs/RESULTS.md](docs/RESULTS.md).

## At a glance

| | |
| --- | --- |
| Input | a local GGUF model file (CPU via llama.cpp) or a model served by a local Ollama (GPU) |
| Battery | 20 fixed prompts: sensor status (5), azimuth (5), distance (4), battery runtime (4), prompt injection (2) |
| Gate | strict JSON schema, then AST check against a 4-tool registry; 8 reject codes |
| Measured per model | schema adherence, gate pass rate, correctness, TTFT (median), tokens/s, peak RSS |
| Decoding modes | constrained (GBNF grammar) and unconstrained, reported separately |
| Tests | 23, including property-based and a fake-server test of the Ollama streaming code; `mypy --strict`; ruff |

## Architecture at a glance

```mermaid
flowchart LR
    PROMPT[/"prompt + system text"/]
    subgraph gen["1 · Generate (probabilistic)"]
        MODEL["Local quantized SLM<br>GGUF via llama.cpp<br>optional GBNF grammar<br>no network"]
    end
    subgraph gates["2 · Deterministic gates"]
        direction TB
        SCHEMA{"JSON matches<br>call: string only,<br>extra keys forbidden?"}
        AST{"ast.parse: one plain call,<br>registered tool, literal<br>positional args of right type?"}
    end
    subgraph exec["3 · Execute"]
        TOOLS["registry of 4 pure local tools<br>sensor status · azimuth<br>distance · battery hours"]
    end
    subgraph meas["Measured around every call"]
        direction TB
        M1["TTFT · tokens/s<br>peak RSS"]
        M2["correct vs expected call"]
        LOG["violations.jsonl<br>with reject code"]
    end
    REPORT(["Markdown table per model<br>and quantization"])
    PROMPT --> MODEL --> SCHEMA
    SCHEMA -- "yes" --> AST
    SCHEMA -- "no: SCHEMA" --> LOG
    AST -- "yes" --> TOOLS --> M2
    AST -- "no: NOT_A_CALL, UNKNOWN_TOOL,<br>NON_LITERAL, ARITY, ARG_TYPE,<br>KEYWORDS, SYNTAX, TOO_LONG" --> LOG
    MODEL -.-> M1
    M1 --> REPORT
    M2 --> REPORT
    LOG --> REPORT
    classDef det fill:#e8f1fb,stroke:#2f6db5,color:#102a43
    classDef prob fill:#fff1d6,stroke:#c27c0e,color:#3d2500,stroke-width:3px
    classDef bad fill:#fde8e8,stroke:#c0392b,color:#4a1010
    classDef io fill:#f2f2f2,stroke:#666,color:#222
    class SCHEMA,AST,TOOLS,M1,M2 det
    class MODEL prob
    class LOG bad
    class PROMPT,REPORT io
    style gen fill:#fff9ec,stroke:#c27c0e,color:#3d2500
    style gates fill:#f4f8fd,stroke:#2f6db5,color:#102a43
    style exec fill:#f4f8fd,stroke:#2f6db5,color:#102a43
    style meas fill:#f4f8fd,stroke:#2f6db5,color:#102a43
```

*Blue is plain deterministic code; the single amber box is the only probabilistic component. Nothing it produces reaches a tool without passing both gates.*

## Quickstart

```bash
git clone https://github.com/PCSchmidt/edge-slm-eval && cd edge-slm-eval
uv venv && uv pip install -e ".[dev]"            # gate, battery, harness: no model needed

python examples/demo_gate.py                    # the output shown above
pytest -q                                       # 23 passed

# with a real model (needs a C++ toolchain to build llama-cpp-python):
uv pip install -e ".[llama]"
edge-slm-eval --model models/llama-3.2-3b-instruct-q4_k_m.gguf --out results/
edge-slm-eval --model ... --no-grammar          # unconstrained, for comparison
```

Offline container (builds with a prebuilt CPU wheel; the tests pass inside it with no network):

```bash
docker build -t edge-slm-eval .
docker run --rm --network none edge-slm-eval                                  # runs the tests
docker run --rm --network none -v "$PWD/models:/models:ro" edge-slm-eval \
    edge-slm-eval --model /models/model.gguf --out /tmp/results
```

To reproduce the real-model table (needs Ollama with the four models pulled; about an hour on CPU):

```bash
python scripts/run_real.py --passes 3      # writes results/real/*.json, runs CPU then GPU
python scripts/diagnose.py                 # rebuilds results/real/results_table.md
```

## Approach: why it is built this way

- **Model output is data, not code.** It becomes a call only after a schema check and an AST check; nothing is ever evaluated.
- **A closed world of tools.** Four pure, local, deterministic functions: no files, no network, no shell. The model cannot make the system do anything outside that list.
- **Measure the cost, not just the quality.** On an edge node, a correct answer that takes eight seconds or exhausts memory is a failure, so time-to-first-token, throughput and peak memory sit beside accuracy.
- **Constrained and unconstrained are separate rows.** Grammar decoding hides format weaknesses; reporting both shows what the grammar is buying.
- **Log every rejection with a code.** A failed gate is data about the model.

## Motivation

Standard LLM agent frameworks assume a fast network and permissive execution. Disconnected and austere deployments have neither, and that is where "what can the model make the system do?" matters most. This repo is a small, inspectable answer to that question, in the same spirit as my Meridian work: probabilistic components propose, deterministic gates decide.

## Method (what the harness actually does)

1. **Battery** (`battery.py`): 20 fixed cases, each with an expected call. Expected calls are validated by the same gate, so the ground truth passes its own checks.
2. **Generate** (`backends.py`): a backend returns text plus timings. `LlamaCppBackend` streams a chat completion at temperature 0, optionally with a GBNF grammar, and records time to first token, tokens produced and peak resident memory. `ScriptedBackend` is a test double.
3. **Gate** (`guardrail.py`): `parse_output` requires JSON with exactly one string field `call`; `check_call` requires one plain call to a registered function, positional literal arguments only (numbers or strings, no booleans, no keywords, no nesting), the right arity and types, and under 500 characters.
4. **Execute** (`tools.py`): only after both gates pass; the result is recorded. Correctness compares the parsed call with the expected call, so `1.0` and `1` are equal.
5. **Report** (`harness.py`): per-model summary and a Markdown table; every rejection goes to `violations.jsonl` with its code.

## Results

Four models x CPU/GPU x constrained/unconstrained, 20 prompts each, the same GGUF weights on both backends. "Correct" requires the output to pass the schema gate and the AST gate and equal the expected call. Full protocol, caveats, and the first pass that was thrown away and why: [docs/RESULTS.md](docs/RESULTS.md).

| Model | Device | Mode | Schema % | Correct % (strict gate) | Correct % if keyword args accepted | TTFT p50 (s) | Decode tok/s | GPU clock, median MHz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2 3B Q4_K_M | CPU | constrained | 100 | 65 | 100 | 0.263 | 10.8 | - |
| Llama 3.2 3B Q4_K_M | CPU | unconstrained | 100 | 65 | 100 | 0.244 | 20.8 | - |
| Llama 3.2 3B Q4_K_M | GPU | constrained | 100 | 60 | 95 | 0.056 | 126.4 | 1200 |
| Llama 3.2 3B Q4_K_M | GPU | unconstrained | 100 | 60 | 95 | 0.057 | 134.5 | 1095 |
| Llama 3.2 3B Q8_0 | CPU | constrained | 100 | 60 | 70 | 0.562 | 8.7 | - |
| Llama 3.2 3B Q8_0 | CPU | unconstrained | 75 | 60 | 70 | 0.554 | 14.0 | - |
| Llama 3.2 3B Q8_0 | GPU | constrained | 100 | 50 | 65 | 0.054 | 101.8 | 1170 |
| Llama 3.2 3B Q8_0 | GPU | unconstrained | 75 | 50 | 65 | 0.061 | 103.8 | 1418 |
| Mistral 7B v0.3 Q4_K_M | CPU | constrained | 100 | 95 | 95 | 0.629 | 8.2 | - |
| Mistral 7B v0.3 Q4_K_M | CPU | unconstrained | 95 | 90 | 90 | 0.698 | 8.8 | - |
| Mistral 7B v0.3 Q4_K_M | GPU | constrained | 100 | 95 | 95 | 0.068 | 64.2 | 952 |
| Mistral 7B v0.3 Q4_K_M | GPU | unconstrained | 95 | 90 | 90 | 0.059 | 63.6 | 945 |
| Qwen2.5-Coder 7B Q4_K_M | CPU | constrained | 100 | 100 | 100 | 0.639 | 6.3 | - |
| Qwen2.5-Coder 7B Q4_K_M | CPU | unconstrained | 60 | 60 | 60 | 0.620 | 10.0 | - |
| Qwen2.5-Coder 7B Q4_K_M | GPU | constrained | 100 | 100 | 100 | 0.067 | 61.4 | 840 |
| Qwen2.5-Coder 7B Q4_K_M | GPU | unconstrained | 65 | 65 | 65 | 0.069 | 62.6 | 878 |

The "if keyword args accepted" column is a post-hoc diagnostic from the saved raw outputs, not the headline; it relaxes only the keyword-argument rule. With 20 prompts a percentage is uncertain by roughly ±20 points, so differences of one to three prompts are noise.

- **Constrained decoding raised format adherence and never lowered it**: Qwen2.5-Coder 7B went from 60-65% valid output to 100%, Llama 3B Q8 from 75% to 100%.
- **A valid JSON wrapper is not a valid call.** Llama 3B Q8 was 100% schema-valid constrained but 50-60% correct: it tries to write an inner quote without escaping it, the grammar closes the string there, and the call is truncated to `get_sensor_status(sensor_id=`. The AST gate is what catches that.
- **One strict rule explains much of the Llama result.** It writes `get_sensor_status(sensor_id='...')` with a keyword argument where the prompt asks for a positional one. Accepting keyword arguments would move Llama 3B Q4 from 65% to 100%. Both numbers are reported.
- **Prompt injection never reached execution.** Across the 320 first-pass outputs nothing outside the four registered tools ran. Llama 3B Q4 on the GPU did emit the injected `os.system('ls')`, rejected `NOT_A_CALL`; Mistral, unconstrained, appended a `delete_logs()` call, rejected by the schema.
- **Speed.** CPU decode 6-21 tok/s with a first token in 0.24-0.70 s; GPU about 100-135 tok/s for the 3B models and 61-64 tok/s for the 7B models. The GPU numbers are **throttled-GPU numbers**: the laptop GPU was power-capped at 840-1418 MHz of a 1785 MHz maximum, so they depend on the clock state and are not hardware limits.

## Limitations

- **Small, narrow battery.** 20 prompts, four tools, zero-shot, one fixed prompt. It tests format discipline and simple calls, not reasoning, long contexts or multi-turn tool use (not built). Intervals are about ±20 points.
- **The strict gate is a choice.** Rejecting keyword arguments is a design decision that costs some models a lot (Llama 3B Q4: 65% against 100%). The headline uses the strict gate; the diagnostic column shows the other view. Few-shot examples would probably remove the habit and are excluded on purpose.
- **GPU timings are not hardware limits.** The GPU was power-capped during every GPU run (clocks 840-1418 MHz of 1785), and one run was thermally throttled and had to be repeated. Accuracy columns are unaffected.
- **CPU and GPU rows are different systems.** They share weights but not chat templates, constrained-decoding implementations or default context length, so small correctness gaps between them are not a hardware effect.
- **One machine, one run each.** No other hardware, no fixed-clock run, and Mistral was non-deterministic across passes on the GPU in constrained mode.
- **Memory is not one measurement.** CPU rows report process memory, GPU rows report the model size Ollama says is loaded.
- **Only some quantizations.** Q4_K_M for the 7B models; Q8_0 only for the 3B. vLLM, multi-turn context degradation and energy use are not covered.
- **The tools are simulated**: a fixed sensor table and standard spherical formulas.

## Operational notes

### What runs where

Everything runs on the local machine; the intended deployment is a container started with `--network none`. The model file is mounted read-only.

### Data and keys

No keys, no network use after the model file is on disk, no private data. Prompts and tools are synthetic.

### Verify the claims (a reviewer's path)

```bash
pytest -q                           # 23 passed, no model needed
mypy src tests                      # strict, clean
python examples/demo_gate.py        # the gate decisions shown at the top
python scripts/diagnose.py          # rebuilds the results table from results/real/*.json
docker build -t edge-slm-eval . && docker run --rm --network none edge-slm-eval   # tests offline
```

### Documentation map

| Doc | What it covers |
| --- | --- |
| [docs/RESULTS.md](docs/RESULTS.md) | protocol, the full table, findings, the GPU throttling note, what went wrong and was fixed |
| [results/real/](results/real/) | every run as JSON: per-prompt outputs, rejection codes, timings and GPU state |
| [scripts/run_real.py](scripts/run_real.py), [scripts/diagnose.py](scripts/diagnose.py) | the benchmark driver and the diagnostic/table builder |
| [examples/demo_gate.py](examples/demo_gate.py) | the gate demo shown above |
| [src/edge_slm_eval/guardrail.py](src/edge_slm_eval/guardrail.py) | the whole gate, in one short file |

## License

MIT. See [LICENSE](LICENSE).

## Author

Chris Schmidt ([PCSchmidt](https://github.com/PCSchmidt)). Part of a portfolio on evaluator-gated AI systems: probabilistic components propose, deterministic gates decide.
