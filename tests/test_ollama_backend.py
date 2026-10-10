from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from edge_slm_eval.backends import JSON_CALL_SCHEMA, OllamaBackend
from edge_slm_eval.battery import build_battery
from edge_slm_eval.harness import Summary, markdown_table, run_battery

REQUESTS: list[dict[str, Any]] = []
REPLY = ['{"call": ', '"get_sensor_status(', '\\"gps-01\\")"}']


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # silence
        pass

    def do_GET(self) -> None:
        body = json.dumps({"models": [{"name": "fake:1b", "model": "fake:1b", "size": 3 * 2**30}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        REQUESTS.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()
        for i, piece in enumerate(REPLY):
            time.sleep(0.01)
            self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
            self.wfile.flush()
        self.wfile.write((json.dumps({"message": {"content": ""}, "done": True, "eval_count": 17}) + "\n").encode())


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), FakeOllama)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_streaming_parse_tokens_ttft_and_memory(server: str) -> None:
    b = OllamaBackend("fake:1b", host=server)
    g = b.generate("sys", "prompt")
    assert g.text == '{"call": "get_sensor_status(\\"gps-01\\")"}'
    assert g.n_tokens == 17  # taken from the server's eval_count, not from the chunk count
    assert 0.0 < g.ttft_s <= g.total_s
    assert g.peak_rss_mb == pytest.approx(3072.0)


def test_constrained_sends_schema_and_unconstrained_does_not(server: str) -> None:
    REQUESTS.clear()
    OllamaBackend("fake:1b", constrained=True, host=server).generate("s", "p")
    OllamaBackend("fake:1b", constrained=False, host=server).generate("s", "p")
    assert REQUESTS[0]["format"] == JSON_CALL_SCHEMA
    assert "format" not in REQUESTS[1]
    assert REQUESTS[0]["options"]["temperature"] == 0  # greedy decoding for reproducibility


def test_backend_describes_itself_for_the_results_table(server: str) -> None:
    b = OllamaBackend("fake:1b", constrained=False, host=server, name="label")
    assert (b.name, b.device, b.mode) == ("label", "gpu (ollama)", "unconstrained")


def test_battery_through_the_fake_server_and_table_columns(server: str) -> None:
    summary, results = run_battery(OllamaBackend("fake:1b", host=server))
    assert summary.n == len(build_battery()) == 20
    assert summary.schema_adherence == 1.0  # the fake always returns valid JSON
    assert summary.decode_tokens_per_s > 0
    assert summary.device == "gpu (ollama)" and summary.mode == "constrained"
    # the fake answers get_sensor_status("gps-01") to everything: right only where that is expected
    assert 0.0 < summary.accuracy < 1.0
    table = markdown_table([summary])
    assert "| Device | Mode |" in table and "gpu (ollama)" in table
    assert isinstance(summary, Summary) and all(r.executed is not None for r in results if r.correct)
