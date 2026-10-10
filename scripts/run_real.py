"""Run the 20-prompt battery on real quantized models, on CPU (llama.cpp) and GPU (Ollama).

Both backends load the SAME GGUF weights: the Ollama blob is passed to llama.cpp directly.
Each configuration is warmed up once, then the battery is run `--passes` times; decoding is
greedy, so correctness should be identical across passes (this is checked and recorded) and the
passes only add timing samples. Results are resumable: finished runs in results/real are skipped.

    python scripts/run_real.py --devices cpu,gpu --passes 3
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

from edge_slm_eval.backends import Backend, LlamaCppBackend, OllamaBackend
from edge_slm_eval.battery import SYSTEM, build_battery
from edge_slm_eval.harness import CaseResult, markdown_table, run_case, summarise

OUT = Path("results/real")
MODELS = {
    "llama3.2-3b-instruct-q4_K_M": "llama3.2:3b-instruct-q4_K_M",
    "llama3.2-3b-instruct-q8_0": "llama3.2:3b-instruct-q8_0",
    "qwen2.5-coder-7b-instruct-q4_K_M": "qwen2.5-coder:7b-instruct-q4_K_M",
    "mistral-7b-instruct-v0.3-q4_K_M": "mistral:7b-instruct-v0.3-q4_K_M",
}


def sh(cmd: list[str]) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return (done.stdout or "").strip()


THROTTLE_BITS = {
    0x1: "gpu_idle", 0x2: "app_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake",
}


class GpuSampler(threading.Thread):
    """Samples GPU clocks, temperature and throttle reasons every 2 s while a config runs.

    Laptop GPUs throttle under sustained load; without this record a throughput number cannot be
    interpreted. 'sm_clock_min/median' are in MHz while the sampler ran.
    """

    QUERY = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active,utilization.gpu"

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self.rows: list[tuple[float, ...]] = []

    def run(self) -> None:
        while not self._stop_evt.is_set():
            out = sh(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"])
            try:
                f = [x.strip() for x in out.split(",")]
                self.rows.append((float(f[0]), float(f[1]), float(f[2]), float(f[3]), int(f[4], 16), float(f[5])))
            except (ValueError, IndexError):
                pass
            self._stop_evt.wait(2.0)

    def finish(self) -> dict[str, Any]:
        self._stop_evt.set()
        self.join(timeout=5)
        busy = [r for r in self.rows if r[5] > 50] or self.rows  # samples where the GPU was working
        if not busy:
            return {}
        reasons = 0
        for r in busy:
            reasons |= int(r[4])
        return {
            "samples": len(self.rows),
            "sm_clock_mhz_median_when_busy": statistics.median(r[0] for r in busy),
            "sm_clock_mhz_min_when_busy": min(r[0] for r in busy),
            "sm_clock_mhz_max": busy[0][1],
            "temp_c_max": max(r[2] for r in busy),
            "power_w_median_when_busy": statistics.median(r[3] for r in busy),
            "throttle_reasons_seen": [n for bit, n in THROTTLE_BITS.items() if reasons & bit and bit != 0x1],
        }



def cool_down(target_c: float = 60.0, max_wait_s: float = 300.0) -> float:
    """Wait for the GPU to cool so each GPU config starts from a comparable thermal state."""
    start = time.monotonic()
    while True:
        try:
            temp = float(sh(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]))
        except ValueError:
            return -1.0
        if temp <= target_c or time.monotonic() - start > max_wait_s:
            return temp
        time.sleep(10)


def blob_path(tag: str) -> str:
    m = re.search(r"^FROM (.+)$", sh(["ollama", "show", tag, "--modelfile"]), re.MULTILINE)
    if not m:
        raise SystemExit(f"cannot resolve a GGUF blob for {tag}; is it pulled?")
    return m.group(1).strip()


def environment() -> dict[str, Any]:
    import llama_cpp

    return {
        "platform": platform.platform(),
        "cpu": sh(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "llama_cpp_python": llama_cpp.__version__,
        "ollama": sh(["ollama", "--version"]),
        "python": platform.python_version(),
    }


def run_config(backend: Backend, passes: int) -> dict[str, Any]:
    cases = build_battery()
    run_case(backend, cases[0])  # warm-up: load weights, fill caches; discarded
    sampler = GpuSampler()
    sampler.start()
    all_passes: list[list[CaseResult]] = [[run_case(backend, c) for c in cases] for _ in range(passes)]
    first = all_passes[0]
    deterministic = all(
        [(r.raw, r.correct) for r in p] == [(r.raw, r.correct) for r in first] for p in all_passes
    )
    gpu_state = sampler.finish()
    pooled = [r for p in all_passes for r in p]
    summary = summarise(backend.name, pooled, backend.device, backend.mode)
    summary = type(summary)(**{**asdict(summary), "n": len(cases)})  # N = distinct prompts
    return {
        "summary": asdict(summary),
        "passes": passes,
        "deterministic_across_passes": deterministic,
        "gpu_state_during_run": gpu_state,
        "system_role_supported": getattr(backend, "system_role_supported", None),
        "ttft_s_per_pass_median": [statistics.median(r.ttft_s for r in p) for p in all_passes],
        "cases": [asdict(r) for r in first],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--devices", default="cpu,gpu")
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = environment()
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1), flush=True)

    for label in args.models.split(","):
        tag = MODELS[label]
        for device in args.devices.split(","):
            for constrained in (True, False):
                mode = "constrained" if constrained else "unconstrained"
                path = OUT / f"{label}__{device}__{mode}.json"
                if path.exists():
                    continue
                print(f"-> {label} {device} {mode}", flush=True)
                if device == "gpu":
                    print(f"   GPU temperature before start: {cool_down():.0f} C", flush=True)
                backend: Backend
                if device == "cpu":
                    backend = LlamaCppBackend(blob_path(tag), use_grammar=constrained, name=label)
                else:
                    backend = OllamaBackend(tag, constrained=constrained, name=label)
                result = run_config(backend, args.passes)
                path.write_text(json.dumps(result, indent=1))
                s = result["summary"]
                print(
                    f"   schema {s['schema_adherence']:.0%} correct {s['accuracy']:.0%} "
                    f"ttft {s['ttft_p50_s']:.2f}s decode {s['decode_tokens_per_s']:.1f} tok/s "
                    f"deterministic={result['deterministic_across_passes']}",
                    flush=True,
                )
                del backend

    from edge_slm_eval.harness import Summary

    summaries = [
        Summary(**json.loads(p.read_text())["summary"]) for p in sorted(OUT.glob("*__*__*.json"))
    ]
    (OUT / "table.md").write_text(markdown_table(summaries), encoding="utf-8")
    print((OUT / "table.md").read_text())
    _ = SYSTEM  # the system prompt used is the one in battery.py


if __name__ == "__main__":
    main()
