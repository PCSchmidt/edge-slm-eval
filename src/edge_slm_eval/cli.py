"""`edge-slm-eval --model path/to/model.gguf [--no-grammar] [--out results/]`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .backends import LlamaCppBackend
from .harness import markdown_table, run_battery, violations_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edge-slm-eval", description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="local GGUF file")
    p.add_argument("--no-grammar", action="store_true", help="disable GBNF constrained decoding")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    backend = LlamaCppBackend(str(args.model), use_grammar=not args.no_grammar)
    summary, results = run_battery(backend)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "violations.jsonl").write_text(violations_jsonl(results), encoding="utf-8")
    table = markdown_table([summary])
    (args.out / f"{backend.name}.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
