"""Command-line interface: ``beladymem {gate,score,version}``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .gates import run_all_gates
from .report import NON_CLAIMS
from .score import score
from .trace import MemoryTrace


def _cmd_gate(args: argparse.Namespace) -> int:
    ok = run_all_gates(verbose=True)
    return 0 if ok else 1


def _cmd_score(args: argparse.Namespace) -> int:
    trace = MemoryTrace.from_jsonl(args.trace)
    policies = [p.strip() for p in args.policy.split(",") if p.strip()]
    reports = [score(trace, args.budget, p, budget_mode=args.budget_mode) for p in policies]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        for r in reports:
            print(r.summary())
        print("\nNON-CLAIMS:")
        for nc in NON_CLAIMS:
            print(f"  - {nc}")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beladymem",
        description=(
            "Score agent-memory eviction policies by competitive ratio against "
            "the Belady MIN offline-optimal oracle (synthetic-validated)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    pg = sub.add_parser("gate", help="run the pre-registered sensitivity gates G1-G9")
    pg.set_defaults(func=_cmd_gate)

    ps = sub.add_parser("score", help="score eviction policies on a JSONL trace")
    ps.add_argument("trace", help="path to a MemoryTrace JSONL file")
    ps.add_argument("--budget", type=int, required=True, help="cache budget B")
    ps.add_argument("--policy", default="lru,lfu,fifo", help="comma-separated reference policies")
    ps.add_argument(
        "--budget-mode", default="count", choices=["count", "bytes"], dest="budget_mode"
    )
    ps.add_argument("--json", action="store_true", help="emit JSON reports")
    ps.set_defaults(func=_cmd_score)

    pv = sub.add_parser("version", help="print the installed version")
    pv.set_defaults(func=_cmd_version)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
