#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.steps import STEP_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic data preparation steps sequentially."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show available steps and exit.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP_NAME",
        help="Optional subset of steps to run (keeps registry order).",
    )
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Request overwriting outputs when the step supports it.",
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Request skipping overwrites when the step supports it.",
    )
    parser.set_defaults(force=None)
    return parser.parse_args()


def _resolve_steps(requested: Iterable[str] | None) -> List[str]:
    if not requested:
        return list(STEP_REGISTRY.keys())

    unknown = [name for name in requested if name not in STEP_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(unknown)}")

    # Preserve the registry ordering to avoid subtle dependency issues.
    ordered = [name for name in STEP_REGISTRY.keys() if name in requested]
    return ordered


def _call_step(func: Callable, *, force: bool | None):
    kwargs = {}
    signature = inspect.signature(func)
    if force is not None and "force" in signature.parameters:
        kwargs["force"] = force
    return func(**kwargs)


def main() -> None:
    args = parse_args()
    if args.list:
        print("Available steps:")
        for name in STEP_REGISTRY.keys():
            print(f"  - {name}")
        return

    steps_to_run = _resolve_steps(args.steps)
    if not steps_to_run:
        print("No steps to run.")
        return

    for name in steps_to_run:
        func = STEP_REGISTRY[name]
        print(f"\n[STEP] {name}")
        result = _call_step(func, force=args.force)
        if result is not None:
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

