#!/usr/bin/env python3
"""Build a self-supervised dataset from local code/doc corpora.

This script extracts prompt/completion style examples (and code<->doc pairs)
from local repositories to support offline self-supervised learning.

Typical workflow:
1) Ingest repos into a local folder (e.g. via ToolEnvironmentBridge action
   `github_repo_ingest`, or by cloning manually).
2) Run this script to build a JSONL dataset for fine-tuning or contrastive training.

The output is JSONL, one example per line:
  {"task": "...", "input": "...", "output": "...", "meta": {...}}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modules.learning.code_doc_self_supervised import CodeDocDatasetConfig, build_self_supervised_examples, write_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=None,
        help="Corpus root to scan (repeatable). Defaults to data/external_repos if it exists.",
    )
    parser.add_argument(
        "--output",
        default="data/training/code_doc_self_supervised.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--max-files", type=int, default=1200, help="Maximum number of files to scan.")
    parser.add_argument("--max-examples", type=int, default=10000, help="Maximum number of examples to emit.")
    parser.add_argument("--max-chars-per-file", type=int, default=180000, help="Max chars read per file.")
    parser.add_argument("--max-input-chars", type=int, default=2800, help="Max chars per input field.")
    parser.add_argument("--max-output-chars", type=int, default=1400, help="Max chars per output field.")
    parser.add_argument(
        "--include-suffix",
        action="append",
        dest="suffixes",
        default=None,
        help="File suffix to include (repeatable), e.g. --include-suffix .py",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=None,
        help="Task name to include (repeatable).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    roots = list(args.roots or [])
    default_root = Path("data/external_repos")
    if not roots and default_root.exists():
        roots = [str(default_root)]
    if not roots:
        roots = ["."]

    suffixes = args.suffixes or None
    tasks = args.tasks or None

    cfg = CodeDocDatasetConfig(
        include_suffixes=tuple(suffixes) if suffixes else CodeDocDatasetConfig.include_suffixes,
        max_files=int(args.max_files),
        max_examples=int(args.max_examples),
        max_chars_per_file=int(args.max_chars_per_file),
        max_input_chars=int(args.max_input_chars),
        max_output_chars=int(args.max_output_chars),
        tasks=tuple(tasks) if tasks else CodeDocDatasetConfig.tasks,
    )

    result = build_self_supervised_examples(roots, config=cfg)
    examples = result.get("examples") or []
    stats = result.get("stats") or {}

    write = write_jsonl(examples, args.output)

    print("[code-doc-self-supervised] done")
    print(f"- roots: {stats.get('roots')}")
    print(f"- files_scanned: {stats.get('files_scanned')} skipped_files: {stats.get('skipped_files')}")
    print(f"- examples: {stats.get('examples')} tasks: {stats.get('tasks')}")
    print(f"- output: {write.get('output_path')} written: {write.get('written')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

