"""Run with: python -m reclab.competition prepare|run|results."""
import argparse
import pandas as pd
from .data import prepare
from .experiment import run_experiment, latest_run
from .graph import GraphConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "results"])
    parser.add_argument("--source", help="Local official DATA folder; otherwise fetch pinned source")
    parser.add_argument("--tag", default="champion-core")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.source))
    elif args.action == "run":
        prepare()
        print(run_experiment(GraphConfig(dim=args.dim, layers=args.layers, epochs=args.epochs), args.tag))
    else:
        table = pd.read_csv(latest_run() / "metrics.csv")
        print(table[(table.partition == "audit") & (table.market == "combined")].to_string(index=False))


if __name__ == "__main__":
    main()
