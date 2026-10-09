from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edge_slm_eval import GuardrailViolation, check_call, execute, parse_output, run_battery
from edge_slm_eval.backends import ScriptedBackend
from edge_slm_eval.battery import build_battery
from edge_slm_eval.harness import markdown_table, violations_jsonl
from edge_slm_eval.tools import azimuth_deg


def test_battery_is_twenty_valid_cases() -> None:
    cases = build_battery()
    assert len(cases) == 20
    for c in cases:
        check_call(c.expected_call)  # ground truth passes its own gate


@pytest.mark.parametrize(
    ("call", "code"),
    [
        ("os.system('ls')", "NOT_A_CALL"),
        ("__import__('os').system('ls')", "NOT_A_CALL"),
        ("eval('1')", "UNKNOWN_TOOL"),
        ("get_sensor_status(open('/etc/passwd').read())", "NON_LITERAL"),
        ("get_sensor_status(sensor_id='a')", "KEYWORDS"),
        ("get_sensor_status('a', 'b')", "ARITY"),
        ("azimuth_deg('x', 1, 2, 3)", "ARG_TYPE"),
        ("get_sensor_status(1)", "ARG_TYPE"),
        ("get_sensor_status(True)", "ARG_TYPE"),
        ("get_sensor_status('a'); x", "SYNTAX"),
        ("get_sensor_status('" + "a" * 600 + "')", "TOO_LONG"),
    ],
)
def test_guardrail_rejects(call: str, code: str) -> None:
    with pytest.raises(GuardrailViolation) as ei:
        check_call(call)
    assert ei.value.code == code


def test_valid_call_executes() -> None:
    assert execute(check_call("get_sensor_status('radar-07')")) == "OFFLINE"
    assert execute(check_call("azimuth_deg(0, 0, 10, 0)")) == 0.0


def test_schema_rejects_extra_keys_and_non_json() -> None:
    for raw in ["not json", '{"call": 1}', '{"call": "x", "extra": 1}', "[]"]:
        with pytest.raises(GuardrailViolation) as ei:
            parse_output(raw)
        assert ei.value.code == "SCHEMA"


def test_harness_perfect_and_corrupted_backends() -> None:
    by_prompt = {c.prompt: c.expected_call for c in build_battery()}
    good = ScriptedBackend(lambda p: json.dumps({"call": by_prompt[p]}), "oracle")
    summary, _ = run_battery(good)
    assert summary.schema_adherence == summary.gate_pass == summary.accuracy == 1.0

    def flaky(p: str) -> str:
        return "sure! here you go" if "radar" in p else json.dumps({"call": "os.system('x')"})

    summary, results = run_battery(ScriptedBackend(flaky, "bad"))
    assert summary.accuracy == 0.0 and summary.gate_pass == 0.0
    assert all(r.executed is None for r in results)  # nothing ran behind a failed gate
    assert len(violations_jsonl(results).splitlines()) == 20
    assert "| bad |" in markdown_table([summary])


@given(st.text(max_size=200))
def test_arbitrary_call_text_never_crashes_or_executes(text: str) -> None:
    try:
        check_call(text)
    except GuardrailViolation:
        pass


@given(
    st.floats(-90, 90), st.floats(-180, 180), st.floats(-90, 90), st.floats(-180, 180)
)
def test_azimuth_in_range(a: float, b: float, c: float, d: float) -> None:
    assert 0.0 <= azimuth_deg(a, b, c, d) < 360.0
