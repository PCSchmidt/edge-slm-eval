"""Runs the battery, records every violation, and renders a Markdown results table."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass

from .backends import Backend
from .battery import SYSTEM, Case, build_battery
from .guardrail import GuardrailViolation, ToolCall, check_call, execute, parse_output


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    raw: str
    schema_ok: bool
    gate_ok: bool
    correct: bool
    violation: str | None
    executed: str | None  # repr of the tool result; only set when the gate passed
    ttft_s: float
    total_s: float
    n_tokens: int
    peak_rss_mb: float | None


@dataclass(frozen=True)
class Summary:
    backend: str
    device: str
    mode: str
    n: int
    schema_adherence: float
    gate_pass: float
    accuracy: float
    ttft_p50_s: float
    tokens_per_s: float  # end to end: tokens / total time, includes prompt processing
    decode_tokens_per_s: float  # tokens after the first / time after the first token
    peak_rss_mb: float | None


def _expected(case: Case) -> ToolCall:
    return check_call(case.expected_call)


def run_case(backend: Backend, case: Case) -> CaseResult:
    gen = backend.generate(SYSTEM, case.prompt)
    schema_ok = gate_ok = correct = False
    violation: str | None = None
    executed: str | None = None
    try:
        out = parse_output(gen.text)
        schema_ok = True
        call = check_call(out.call)
        gate_ok = True
        correct = call == _expected(case)
        executed = repr(execute(call))
    except GuardrailViolation as exc:
        violation = str(exc)
    return CaseResult(
        case.id, gen.text, schema_ok, gate_ok, correct, violation, executed,
        gen.ttft_s, gen.total_s, gen.n_tokens, gen.peak_rss_mb,
    )  # fmt: skip


def summarise(backend: str, results: list[CaseResult], device: str = "", mode: str = "") -> Summary:
    n = len(results)
    total_tokens = sum(r.n_tokens for r in results)
    total_time = sum(r.total_s for r in results)
    decode_tokens = sum(max(r.n_tokens - 1, 0) for r in results)
    decode_time = sum(max(r.total_s - r.ttft_s, 0.0) for r in results)
    rss = [r.peak_rss_mb for r in results if r.peak_rss_mb is not None]
    return Summary(
        backend=backend,
        device=device,
        mode=mode,
        n=n,
        schema_adherence=sum(r.schema_ok for r in results) / n,
        gate_pass=sum(r.gate_ok for r in results) / n,
        accuracy=sum(r.correct for r in results) / n,
        ttft_p50_s=statistics.median(r.ttft_s for r in results),
        tokens_per_s=total_tokens / total_time if total_time > 0 else 0.0,
        decode_tokens_per_s=decode_tokens / decode_time if decode_time > 0 else 0.0,
        peak_rss_mb=max(rss) if rss else None,
    )


def run_battery(backend: Backend) -> tuple[Summary, list[CaseResult]]:
    results = [run_case(backend, c) for c in build_battery()]
    return summarise(backend.name, results, backend.device, backend.mode), results


def violations_jsonl(results: list[CaseResult]) -> str:
    return "\n".join(json.dumps(asdict(r)) for r in results if r.violation is not None)


def markdown_table(summaries: list[Summary]) -> str:
    head = (
        "| Model | Device | Mode | N | Schema % | Gate % | Correct % | TTFT p50 (s) | Decode tok/s | Memory (MB) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {s.backend} | {s.device} | {s.mode} | {s.n} | {100 * s.schema_adherence:.0f} "
        f"| {100 * s.gate_pass:.0f} | {100 * s.accuracy:.0f} | {s.ttft_p50_s:.3f} "
        f"| {s.decode_tokens_per_s:.1f} | {'n/a' if s.peak_rss_mb is None else f'{s.peak_rss_mb:.0f}'} |"
        for s in summaries
    ]
    return head + "\n".join(rows) + "\n"
