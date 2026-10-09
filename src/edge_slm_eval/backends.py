"""Generation backends. The harness only sees `Backend`."""

from __future__ import annotations

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
    name: str

    def generate(self, system: str, prompt: str) -> Generation: ...


class ScriptedBackend:
    """Deterministic test double: `responder(prompt) -> raw text`. Timings are fake (0)."""

    def __init__(self, responder: Callable[[str], str], name: str = "scripted") -> None:
        self.name = name
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
    ) -> None:
        from llama_cpp import Llama, LlamaGrammar

        self.name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
        self._llm: Any = Llama(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False
        )
        self._grammar: Any = LlamaGrammar.from_string(JSON_CALL_GBNF) if use_grammar else None
        self._max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> Generation:
        import psutil

        proc = psutil.Process()
        peak = proc.memory_info().rss
        start = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []
        n = 0
        for chunk in self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
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
