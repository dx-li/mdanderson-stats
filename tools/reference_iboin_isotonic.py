"""Exact small weighted-isotonic references, independent of the package/SciPy.

Enumerate all contiguous partitions. For each partition, fit its blocks by
their exact weighted means and retain only nondecreasing fits. The least
weighted squared-error fit is the isotonic projection. Fractions preserve
exact decimal inputs and reveal any pooling/tie differences without a second
floating-point PAVA implementation. This verifies declared weights and rates;
it does not establish undocumented native iBOIN selection conventions.
"""

import csv
import itertools
from fractions import Fraction
from pathlib import Path


def exact_projection(values: list[Fraction], weights: list[Fraction]) -> list[Fraction]:
    size = len(values)
    best_loss: Fraction | None = None
    best_fit: list[Fraction] | None = None
    for breaks in itertools.product((False, True), repeat=size - 1):
        edges = [0, *(i + 1 for i, split in enumerate(breaks) if split), size]
        means = [
            sum(weights[i] * values[i] for i in range(a, b)) / sum(weights[a:b])
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
        if any(left > right for left, right in zip(means[:-1], means[1:], strict=True)):
            continue
        fitted = [
            mean
            for a, b, mean in zip(edges[:-1], edges[1:], means, strict=True)
            for _ in range(a, b)
        ]
        loss = sum(w * (y - fit) ** 2 for w, y, fit in zip(weights, values, fitted, strict=True))
        if best_loss is None or loss < best_loss:
            best_loss, best_fit = loss, fitted
        elif loss == best_loss:
            assert fitted == best_fit, "positive weights imply a unique projection"
    assert best_fit is not None
    return best_fit


def build_rows() -> list[dict[str, str | int]]:
    # name, patients, toxicities, original prior ESS, skeleton, custom weights
    scenarios = [
        ("monotone", [10, 10, 10], [1, 2, 4], [2, 2, 2], [".1", ".25", ".5"], [1, 2, 3]),
        ("valley", [10, 5, 10], [1, 3, 2], [2, 4, 2], [".1", ".25", ".5"], [4, 1, 2]),
        ("decreasing", [10, 10, 10], [7, 4, 1], [1, 3, 5], [".1", ".25", ".5"], [1, 4, 1]),
        (
            "separate_blocks",
            [10, 10, 20, 10],
            [6, 1, 12, 2],
            [2, 4, 2, 4],
            [".05", ".15", ".3", ".5"],
            [1, 3, 2, 5],
        ),
        ("plateau", [8, 4, 12], [2, 1, 3], [2, 4, 2], [".1", ".25", ".5"], [3, 1, 4]),
        ("borrowing_changes_pool", [4, 8, 4], [3, 0, 2], [2, 4, 2], [".1", ".25", ".5"], [1, 2, 1]),
    ]
    rows = []
    for name, patients, toxicities, prior_ess, skeleton, custom in scenarios:
        q = [Fraction(x) for x in skeleton]
        for borrow, weight_policy in itertools.product(
            (False, True), ("patients", "effective", "custom")
        ):
            denominator = [
                n + (m if borrow else 0) for n, m in zip(patients, prior_ess, strict=True)
            ]
            values = [
                (Fraction(y) + (m * prior if borrow else 0)) / total
                for y, m, prior, total in zip(toxicities, prior_ess, q, denominator, strict=True)
            ]
            supplied = (
                patients
                if weight_policy == "patients"
                else denominator
                if weight_policy == "effective"
                else custom
            )
            weights = [Fraction(w) for w in supplied]
            fitted = exact_projection(values, weights)
            case = f"{name}_{'borrow' if borrow else 'observed'}_{weight_policy}"
            for i in range(len(values)):
                rows.append(
                    {
                        "case": case,
                        "dose": i + 1,
                        "patients": patients[i],
                        "toxicities": toxicities[i],
                        "prior_ess": prior_ess[i],
                        "skeleton": skeleton[i],
                        "borrow": int(borrow),
                        "weight_policy": weight_policy,
                        "weight": str(weights[i]),
                        "rate_exact": str(values[i]),
                        "fit_exact": str(fitted[i]),
                    }
                )
    return rows


if __name__ == "__main__":
    rows = build_rows()
    destination = Path(__file__).resolve().parents[1] / "tests/fixtures/iboin-isotonic-exact.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} exact reference rows across {len({r['case'] for r in rows})} cases.")
