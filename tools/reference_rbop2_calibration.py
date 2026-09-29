"""Independent rational enumeration of all 16 two-look randomized trial paths.

No package code or numerical probability libraries are used. Integer-shape
Beta comparisons are integrated as finite polynomials with Fraction arithmetic.
"""

import json
from fractions import Fraction as F
from itertools import product
from math import comb, factorial
from pathlib import Path


def beta(a: int, b: int) -> F:
    return F(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))


def larger(a: int, b: int, c: int, d: int) -> F:
    # Integrate f_X(x) * F_Y(x) using Y's finite binomial-CDF identity.
    degree = c + d - 1
    return sum(
        (F(comb(degree, j)) * beta(a + j, b + degree - j) for j in range(c, degree + 1)), F(0)
    ) / beta(a, b)


CURVES = [
    [[".25", ".8"], [".8", ".8"]],
    [[".25", ".9"], [".9", ".9"]],
    [[".05", ".95"], [".8", ".8"]],
    [[".6", ".9"], [".8", ".8"]],
    [[".25", ".9"], [".9", ".9"]],
]
PRIORS = {"calibration": [[1, 1], [1, 1]], "analysis": [[2, 1], [1, 2]]}


def enumerate_trial(endpoint: str, prior: list, curve: list, pe: F, pc: F) -> tuple:
    positive = expected = total = F(0)
    for e1, e2, c1, c2 in product((0, 1), repeat=4):
        weight = (
            pe ** (e1 + e2)
            * (1 - pe) ** (2 - e1 - e2)
            * pc ** (c1 + c2)
            * (1 - pc) ** (2 - c1 - c2)
        )
        for i, (e, c) in enumerate(((e1, c1), (e1 + e2, c1 + c2))):
            n = i + 1
            q = larger(prior[0][0] + e, prior[0][1] + n - e, prior[1][0] + c, prior[1][1] + n - c)
            if endpoint == "toxicity":
                q = 1 - q
            lower, upper = map(F, curve[i])
            if q < lower or q >= upper:
                total += weight
                expected += 2 * n * weight
                positive += weight * (q >= upper)
                break
        else:
            raise AssertionError("Final cutoffs must resolve every outcome path")
    assert total == 1
    return positive, expected


def main() -> None:
    alpha = F(1, 5)
    reference = {
        "looks": [[1, 1], [2, 2]],
        "cutoff_candidates": CURVES,
        "alpha": float(alpha),
        "priors": PRIORS,
        "endpoints": {},
    }
    for endpoint, alternative in (
        ("efficacy", (F(3, 5), F(3, 10))),
        ("toxicity", (F(1, 10), F(3, 10))),
    ):
        results = {}
        for name, prior in PRIORS.items():
            candidates = []
            for index, curve in enumerate(CURVES):
                null = enumerate_trial(endpoint, prior, curve, F(3, 10), F(3, 10))
                alt = enumerate_trial(endpoint, prior, curve, *alternative)
                values = dict(
                    null_positive=null[0],
                    null_expected_total=null[1],
                    alternative_positive=alt[0],
                    alternative_expected_total=alt[1],
                )
                candidates.append(
                    {
                        "index": index,
                        **{k: float(v) for k, v in values.items()},
                        "rational": {k: str(v) for k, v in values.items()},
                    }
                )
            results[name] = candidates
        feasible = [
            row for row in results["calibration"] if F(row["rational"]["null_positive"]) <= alpha
        ]
        selected = (
            min(
                feasible,
                key=lambda row: (
                    -F(row["rational"]["alternative_positive"]),
                    F(row["rational"]["null_expected_total"]),
                    row["index"],
                ),
            )["index"]
            if feasible
            else None
        )
        reference["endpoints"][endpoint] = {
            "null_rates": [0.3, 0.3],
            "alternative_rates": [float(v) for v in alternative],
            "expected_selected_index": selected,
            "candidates": results,
        }
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/rbop2-calibration-reference.json"
    path.write_text(json.dumps(reference, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: value["expected_selected_index"]
                for name, value in reference["endpoints"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
