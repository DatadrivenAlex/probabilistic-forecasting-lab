"""CLI entry point for the reproducible synthetic backtest."""

from __future__ import annotations

import argparse
import json

from .data import make_synthetic_series
from .forecast import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline probabilistic forecast lab")
    parser.add_argument("--hours", type=int, default=24 * 120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_backtest(make_synthetic_series(args.hours, seed=args.seed))
    print(json.dumps(result.to_dict(), indent=2))
    print("Synthetic offline benchmark; results do not establish real-world forecast performance.")


if __name__ == "__main__":
    main()
