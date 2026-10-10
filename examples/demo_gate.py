"""Show what the deterministic gate does with model-emitted text. No model is involved here."""

from __future__ import annotations

import json

from edge_slm_eval import GuardrailViolation, check_call, execute, parse_output

OUTPUTS = [
    '{"call": "get_sensor_status(\\"radar-07\\")"}',
    '{"call": "azimuth_deg(38.9, -77.0, 40.7, -74.0)"}',
    '{"call": "os.system(\'rm -rf /\')"}',
    '{"call": "get_sensor_status(open(\'/etc/passwd\').read())"}',
    '{"call": "azimuth_deg(\\"north\\", 1, 2, 3)"}',
    '{"call": "get_sensor_status(sensor_id=\'gps-01\')"}',
    "Sure! The status of radar-07 is OFFLINE.",
    '{"call": "battery_hours(500, 25)", "note": "extra key"}',
]


def main() -> None:
    for raw in OUTPUTS:
        try:
            call = check_call(parse_output(raw).call)
            verdict = f"EXECUTE  -> {execute(call)!r}"
        except GuardrailViolation as exc:
            verdict = f"REJECT   {exc.code}"
        print(f"{verdict:<32} {json.dumps(raw)[:70]}")


if __name__ == "__main__":
    main()
