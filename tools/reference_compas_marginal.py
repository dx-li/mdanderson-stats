"""Independent two-arm marginal-prior integration (group mean integrated analytically)."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, logit


def reference(ns, ys, var, p0):
    sd = math.sqrt(var / 2)
    avar = 1e6 + var / 2
    threshold = logit(p0)
    const = sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) for n, y in zip(ns, ys))

    def inner(a, which):
        def f(z):
            d = z * sd
            theta = (a + d, a - d)
            lp = sum(
                -y * np.logaddexp(0.0, -t) - (n - y) * np.logaddexp(0.0, t)
                for n, y, t in zip(ns, ys, theta)
            )
            return math.exp(lp + const - 0.5 * z * z) / math.sqrt(2 * math.pi)

        if which is None:
            lo, hi = -12, 12
        elif which == 0:
            lo, hi = max(-12, (threshold - a) / sd), 12
        else:
            lo, hi = -12, min(12, (a - threshold) / sd)
        if lo >= hi:
            return 0.0
        points = sorted(set([lo, hi] + ([0.0] if lo < 0 < hi else [])))
        return sum(
            quad(f, left, right, epsabs=1e-14, epsrel=1e-11, limit=200)[0]
            for left, right in zip(points, points[1:])
        )

    def integrate(which):
        def f(a):
            return inner(a, which) * math.exp(-a * a / (2 * avar)) / math.sqrt(2 * math.pi * avar)

        points = sorted(set([-60.0, -10.0, 0.0, float(threshold), 10.0, 60.0]))
        return sum(
            quad(f, a, b, epsabs=2e-16, epsrel=1e-10, limit=200)[0]
            for a, b in zip(points, points[1:])
        )

    den = integrate(None)
    return math.log(den), [integrate(j) / den for j in range(2)]


if __name__ == "__main__":
    cases = []
    for ns, ys, p0 in [([6, 7], [2, 5], 0.4), ([3, 3], [1, 2], 0.5), ([11, 11], [4, 8], 0.4)]:
        for v in [0.00012, 0.042, 0.126, 0.379, 126.1]:
            logz, tails = reference(ns, ys, v, p0)
            case = dict(n=ns, y=ys, p0=p0, variance=v, log_evidence=logz, tails=tails)
            cases.append(case)
            print(case, flush=True)
    Path("tests/fixtures/compas-marginal-reference.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
