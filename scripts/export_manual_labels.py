from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.manual import MANUAL_LABELS, export_conflict_cards_for_labeling


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export conflict cards to a CSV template for manual labeling."
    )
    parser.add_argument(
        "run_dir_or_cards_path",
        help="A processed run directory or a cards.jsonl path.",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    output_path = export_conflict_cards_for_labeling(
        args.run_dir_or_cards_path,
        output_csv=args.output_csv,
    )
    print(f"Wrote manual labeling CSV: {output_path}")
    print("Allowed manual_label values:")
    for label in MANUAL_LABELS:
        print(f"- {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

