"""High-precision integer-shape Beta references without NumPy or SciPy.

For positive integer a,b, the Beta(a,b) CDF is the upper binomial sum
with n=a+b-1 and success probability x. Decimal arithmetic also computes
the posterior-SD interval boundaries, preserving 70-digit intermediate
precision. This verifies the explicit conjugate generalization, not native
informative-prior defaults.
"""

import csv
from decimal import Decimal, localcontext
from math import comb
from pathlib import Path


def beta_cdf(a: int, b: int, x: Decimal) -> Decimal:
    if x <= 0:
        return Decimal(0)
    if x >= 1:
        return Decimal(1)
    n = a + b - 1
    return sum(
        (Decimal(comb(n, j)) * x**j * (1 - x) ** (n - j) for j in range(a, n + 1)),
        start=Decimal(0),
    )


def rows() -> list[dict[str, int | str]]:
    result = []
    with localcontext() as context:
        context.prec = 70
        target = Decimal("0.3")
        for dose, (prior_a, prior_b) in enumerate(((1, 9), (2, 8), (3, 7)), start=1):
            for patients in (0, 2, 4, 8):
                for toxicities in sorted({0, patients // 2, patients}):
                    a = prior_a + toxicities
                    b = prior_b + patients - toxicities
                    total = a + b
                    mean = Decimal(a) / total
                    sd = (Decimal(a * b) / (total * total * (total + 1))).sqrt()
                    lower = max(Decimal(0), target - Decimal("1.5") * sd)
                    upper = min(Decimal(1), target + sd)
                    left = beta_cdf(a, b, lower)
                    right = beta_cdf(b, a, 1 - upper)
                    middle = beta_cdf(a, b, upper) - left
                    overdose = beta_cdf(b, a, 1 - target)
                    result.append(
                        {
                            "dose": dose,
                            "prior_a": prior_a,
                            "prior_b": prior_b,
                            "patients": patients,
                            "toxicities": toxicities,
                            "mean": str(mean),
                            "standard_deviation": str(sd),
                            "lower": str(lower),
                            "upper": str(upper),
                            "escalate_probability": str(left),
                            "stay_probability": str(middle),
                            "deescalate_probability": str(right),
                            "overdose_probability": str(overdose),
                            "unsafe": int(patients >= 2 and overdose > Decimal("0.95")),
                        }
                    )
    return result


if __name__ == "__main__":
    records = rows()
    destination = Path(__file__).resolve().parents[1] / "tests/fixtures/tpi-informative-beta.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} independently calculated posterior references.")
