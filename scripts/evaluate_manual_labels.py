from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.manual import compute_manual_metrics, save_manual_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute manual precision-style metrics from a labeled conflict-card CSV."
    )
    parser.add_argument("label_csv", help="CSV produced by export_manual_labels.py.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    metrics = compute_manual_metrics(args.label_csv)
    output_path = save_manual_metrics(args.label_csv, output_path=args.output_json)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

