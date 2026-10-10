from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("diagnose", Path(__file__).parent.parent / "scripts" / "diagnose.py")
assert spec and spec.loader
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def test_keyword_call_is_reordered_to_parameter_order() -> None:
    assert diagnose.positionalise("get_sensor_status(sensor_id='gps-01')") == "get_sensor_status('gps-01')"
    # arguments given out of order are put back in signature order
    assert (
        diagnose.positionalise("battery_hours(load_w=25, capacity_wh=500)") == "battery_hours(500, 25)"
    )


def test_only_the_exact_keyword_shape_is_rewritten() -> None:
    assert diagnose.positionalise("get_sensor_status('gps-01')") is None  # already positional
    assert diagnose.positionalise("get_sensor_status(sensor='gps-01')") is None  # wrong name
    assert diagnose.positionalise("battery_hours(500, load_w=25)") is None  # mixed
    assert diagnose.positionalise("os.system(cmd='x')") is None  # not a registered tool
    assert diagnose.positionalise("get_sensor_status(") is None  # does not parse
