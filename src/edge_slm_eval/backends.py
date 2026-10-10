"""Generation backends. The harness only sees `Backend`."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Generation:
    text: str
    ttft_s: float
    total_s: float
    n_tokens: int
    peak_rss_mb: float | None = None


class Backend(Protocol):
    name: str  # model identifier
    device: str  # e.g. "cpu (llama.cpp)" or "gpu (ollama)"
    mode: str  # "constrained" (grammar / JSON schema) or "unconstrained"

    def generate(self, system: str, prompt: str) -> Generation: ...


class ScriptedBackend:
    """Deterministic test double: `responder(prompt) -> raw text`. Timings are fake (0)."""

    def __init__(self, responder: Callable[[str], str], name: str = "scripted") -> None:
        self.name = name
        self.device = "none (scripted)"
        self.mode = "n/a"
        self._responder = responder

    def generate(self, system: str, prompt: str) -> Generation:
        text = self._responder(prompt)
        return Generation(text=text, ttft_s=0.0, total_s=0.0, n_tokens=len(text.split()))


# GBNF grammar constraining output to {"call": "<string>"}; the AST gate still validates the call.
JSON_CALL_GBNF = r"""
root ::= "{" ws "\"call\"" ws ":" ws string ws "}"
string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
ws ::= [ \t\n]*
"""


class LlamaCppBackend:
    """Local GGUF model via llama-cpp-python (extra: `llama`). No network access is used."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        use_grammar: bool = True,
        max_tokens: int = 96,
        n_gpu_layers: int = 0,
        name: str | None = None,
    ) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_grammar import LlamaGrammar

        self.name = name or model_path.replace("\\", "/").rsplit("/", 1)[-1]
        self.device = "cpu (llama.cpp)" if n_gpu_layers == 0 else "gpu (llama.cpp)"
        self.mode = "constrained" if use_grammar else "unconstrained"
        self._llm: Any = Llama(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers, verbose=False
        )
        self._grammar: Any = LlamaGrammar.from_string(JSON_CALL_GBNF) if use_grammar else None
        self._max_tokens = max_tokens
        self.system_role_supported = self._template_keeps_system_role()

    def _template_keeps_system_role(self) -> bool:
        """Probe the model's own chat template: does a system message survive into the prompt?

        Some templates (e.g. Mistral v0.3) silently drop the system role. Without this check the
        model never sees its instructions and every result is meaningless. When the role is not
        kept, `generate` folds the system text into the user message instead.
        """
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        template = self._llm.metadata.get("tokenizer.chat_template")
        if not template:
            return False
        try:
            bos = self._llm._model.token_get_text(self._llm.token_bos())
            eos = self._llm._model.token_get_text(self._llm.token_eos())
            probe = Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)(
                messages=[{"role": "system", "content": "SYSPROBE-91"}, {"role": "user", "content": "hi"}]
            ).prompt
        except Exception:  # noqa: BLE001 - any template failure means the system role is not usable
            return False
        return "SYSPROBE-91" in probe

    def generate(self, system: str, prompt: str) -> Generation:
        import psutil

        proc = psutil.Process()
        peak = proc.memory_info().rss
        start = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []
        n = 0
        for chunk in self._llm.create_chat_completion(
            messages=(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                if self.system_role_supported
                else [{"role": "user", "content": system + "\n\n" + prompt}]
            ),
            temperature=0.0,
            max_tokens=self._max_tokens,
            grammar=self._grammar,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                if first is None:
                    first = time.perf_counter() - start
                pieces.append(delta)
                n += 1
                peak = max(peak, proc.memory_info().rss)
        total = time.perf_counter() - start
        return Generation("".join(pieces), first if first is not None else total, total, n, peak / 2**20)


JSON_CALL_SCHEMA = {
    "type": "object",
    "properties": {"call": {"type": "string"}},
    "required": ["call"],
    "additionalProperties": False,
}


class OllamaBackend:
    """A local Ollama server over localhost (stdlib HTTP only; no model data leaves the machine).

    `constrained` sends a JSON schema as the `format` field (server-side grammar decoding).
    TTFT is wall-clock to the first non-empty content chunk, so the model must be warm. Memory is
    the model size the server reports for the loaded model (VRAM + RAM), not process RSS.
    """

    def __init__(
        self,
        model: str,
        constrained: bool = True,
        host: str = "http://127.0.0.1:11434",
        max_tokens: int = 96,
        name: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.name = name or model
        self.device = "gpu (ollama)"
        self.mode = "constrained" if constrained else "unconstrained"
        self._model, self._host, self._max = model, host.rstrip("/"), max_tokens
        self._constrained, self._timeout = constrained, timeout_s

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        import urllib.request

        req = urllib.request.Request(
            self._host + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(req, timeout=self._timeout)

    def loaded_mb(self) -> float | None:
        """Size of this model as loaded, from /api/ps, in MB."""
        import urllib.request

        with urllib.request.urlopen(self._host + "/api/ps", timeout=10) as r:
            for m in json.load(r).get("models", []):
                if m.get("model") == self._model or m.get("name") == self._model:
                    return float(m["size"]) / 2**20
        return None

    def generate(self, system: str, prompt: str) -> Generation:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0, "num_predict": self._max, "seed": 0},
            "keep_alive": "10m",
        }
        if self._constrained:
            body["format"] = JSON_CALL_SCHEMA
        start = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []
        n_tokens = 0
        with self._post("/api/chat", body) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if first is None:
                        first = time.perf_counter() - start
                    pieces.append(piece)
                if chunk.get("done"):
                    n_tokens = int(chunk.get("eval_count", len(pieces)))
        total = time.perf_counter() - start
        return Generation("".join(pieces), first if first is not None else total, total, n_tokens, self.loaded_mb())
