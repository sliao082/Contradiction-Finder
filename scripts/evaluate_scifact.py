from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NLI_MODEL, SCIFACT_EVALUATION_DIR
from src.evaluation.scifact import DEFAULT_THRESHOLDS, run_scifact_evaluation


def parse_thresholds(raw: str) -> list[float]:
    if not raw.strip():
        return DEFAULT_THRESHOLDS
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local NLI contradiction scoring on SciFact-derived pairs."
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Maximum examples to evaluate. Use 0 for all available pairs.",
    )
    parser.add_argument("--model-name", default=NLI_MODEL)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated contradiction thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation outputs. Defaults to data/evaluation/scifact/{split}.",
    )
    parser.add_argument(
        "--exclude-neutral",
        action="store_true",
        help="Evaluate only SUPPORT/CONTRADICT examples with rationale evidence.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip optional PNG chart generation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCIFACT_EVALUATION_DIR / args.split
    result = run_scifact_evaluation(
        split=args.split,
        max_examples=None if args.max_examples <= 0 else args.max_examples,
        thresholds=parse_thresholds(args.thresholds),
        model_name=args.model_name,
        output_dir=output_dir,
        include_neutral=not args.exclude_neutral,
        save_charts=not args.no_charts,
    )
    print(json.dumps(result["paths"], indent=2))
    print(json.dumps(result["metrics"]["prediction_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

