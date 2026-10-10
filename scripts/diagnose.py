"""Post-hoc diagnostic on saved raw outputs: how much accuracy does one strict rule cost?

The gate rejects keyword arguments, e.g. get_sensor_status(sensor_id='thermal-01'). That is a
design decision (positional literals only), and the headline numbers keep it. This script asks a
separate question of the saved outputs: if keyword arguments matching the real parameter names
were accepted and reordered, how many calls would then be correct? Nothing else is relaxed:
the JSON schema, the registry, literal-only arguments and types are all still enforced.

This is a diagnostic, never a replacement for the gate. It never executes anything.

    python scripts/diagnose.py   -> results/real/diagnostics.json
"""

from __future__ import annotations

import ast
import collections
import inspect
import json
from pathlib import Path
from typing import Any

from edge_slm_eval.battery import build_battery
from edge_slm_eval.guardrail import GuardrailViolation, check_call, parse_output
from edge_slm_eval.tools import REGISTRY

OUT = Path("results/real")


def positionalise(call: str) -> str | None:
    """Rewrite name(k=v, ...) as name(v, ...) in parameter order; None if it is not that shape."""
    try:
        node = ast.parse(call.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)) or node.args:
        return None
    spec = REGISTRY.get(node.func.id)
    if spec is None or not node.keywords:
        return None
    names = list(inspect.signature(spec.fn).parameters)
    given = {k.arg: k.value for k in node.keywords if k.arg}
    if sorted(given) != sorted(names):
        return None
    ordered = ", ".join(ast.unparse(given[n]) for n in names)
    return f"{node.func.id}({ordered})"


def write_table(rows: dict[str, dict[str, Any]]) -> None:
    """results_table.md: one row per configuration, built only from the saved JSON."""
    head = (
        "| Model | Device | Mode | Schema % | Correct % (strict gate) | Correct % if keyword args accepted "
        "| TTFT p50 (s) | Decode tok/s | GPU clock, median MHz | Memory (MB) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for f in sorted(OUT.glob("*__*__*.json")):
        run = json.loads(f.read_text())
        sm, d = run["summary"], rows[f.stem]
        clock = run.get("gpu_state_during_run", {}).get("sm_clock_mhz_median_when_busy")
        gpu_clock = f"{clock:.0f}" if clock and sm["device"].startswith("gpu") else "-"
        mem = "n/a" if sm["peak_rss_mb"] is None else f"{sm['peak_rss_mb']:.0f}"
        lenient = 100 * d["correct_if_keywords_accepted"] / d["of"]
        lines.append(
            f"| {sm['backend']} | {sm['device']} | {sm['mode']} | {100 * sm['schema_adherence']:.0f} "
            f"| {100 * sm['accuracy']:.0f} | {lenient:.0f} "
            f"| {sm['ttft_p50_s']:.3f} | {sm['decode_tokens_per_s']:.1f} | {gpu_clock} | {mem} |"
        )
    (OUT / "results_table.md").write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    expected = {c.id: check_call(c.expected_call) for c in build_battery()}
    rows = {}
    for f in sorted(OUT.glob("*__*__*.json")):
        run = json.loads(f.read_text())
        strict = lenient = 0
        reasons: collections.Counter[str] = collections.Counter()
        for case in run["cases"]:
            strict += bool(case["correct"])
            ok = bool(case["correct"])
            if not ok and (case["violation"] or "").startswith("KEYWORDS"):
                try:
                    fixed = positionalise(parse_output(case["raw"]).call)
                    if fixed and check_call(fixed) == expected[case["case_id"]]:
                        ok = True
                        reasons["recovered_by_keyword_normalisation"] += 1
                except GuardrailViolation:
                    pass
            lenient += ok
        rows[f.stem] = {
            "strict_correct": strict,
            "correct_if_keywords_accepted": lenient,
            "of": len(run["cases"]),
            "recovered": dict(reasons),
        }
    (OUT / "diagnostics.json").write_text(json.dumps(rows, indent=1))
    write_table(rows)
    print(f"{'config':<62}{'strict':>8}{'kw-ok':>8}")
    for k, v in rows.items():
        print(f"{k:<62}{v['strict_correct']:>5}/20{v['correct_if_keywords_accepted']:>5}/20")


if __name__ == "__main__":
    main()
