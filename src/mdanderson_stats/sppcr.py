"""Command-line entry: python -m mdanderson_stats.sppcr --seed INTEGER."""

import argparse
import sys

import numpy as np

from .cdflib_console import CDFConsoleError
from .randlib import RandlibGenerator
from .sppcr_console import run_sppcr


def main(argv: list[str] | None = None) -> int:
    """Run the menu on standard streams with a reproducible, explicit seed."""
    parser = argparse.ArgumentParser(description="SPPCR allele-frequency analysis menu")
    parser.add_argument("--seed", type=int, required=True, help="explicit RNG seed")
    parser.add_argument(
        "--save-reports", action="store_true", help="ask for output files after each analysis"
    )
    parser.add_argument("--legacy", action="store_true", help="use historical RANDLIB sampling")
    parser.add_argument("--seed2", type=int, default=123456789, help="second seed for legacy RNG")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--unseen-alleles", choices=("drop", "retain"), default="drop")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write-simulations", dest="simulations", action="store_const", const=True)
    output.add_argument("--no-simulations", dest="simulations", action="store_const", const=False)
    args = parser.parse_args(argv)
    try:
        rng = (
            RandlibGenerator(seed=(args.seed, args.seed2))
            if args.legacy
            else np.random.default_rng(args.seed)
        )
        result = run_sppcr(
            rng=rng,
            replicates=args.replicates,
            max_steps=args.max_steps,
            unseen_alleles=args.unseen_alleles,
            write_simulations=args.simulations,
            ask_save=args.save_reports,
        )
    except (ValueError, ArithmeticError, CDFConsoleError, OSError) as error:
        print("SPPCR: " + str(error), file=sys.stderr)
        return 2
    return 1 if result.rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
