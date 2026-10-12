"""Independent enumeration checks separate legacy arithmetic from paper criteria."""

import itertools
import json
import math
from pathlib import Path

import pytest
from scipy.special import betaincc

ROOT = Path(__file__).parent / "fixtures"
PREDICTIVE = json.loads((ROOT / "prt-desktop-predictive.json").read_text())["cases"]
NEGLIGIBLE = json.loads((ROOT / "prt-desktop-negligible.json").read_text())["cases"]


def risks(p, alpha, beta, target=0.3, lower=0.3, upper=0.9):
    pn = pe = 0.0
    for bits in itertools.product((0, 1), repeat=len(p)):
        k = sum(bits)
        mass = math.prod(v if bit else 1 - v for v, bit in zip(p, bits, strict=True))
        exceedance = betaincc(alpha + k, beta + len(p) - k, target)
        pn += mass * (exceedance < lower)
        pe += mass * (exceedance > upper)
    return pn, pe


@pytest.mark.parametrize("case", PREDICTIVE)
def test_native_predictive_uses_fixed_risk_product_and_strict_upper(case):
    _, pe = risks(
        case["probabilities"], case["alpha"], case["beta"], case["target"], upper=case["upper"]
    )
    assert case["predictive_excessive"] == pytest.approx(pe, abs=1e-12)


@pytest.mark.parametrize("case", NEGLIGIBLE)
def test_native_negligible_predictive_and_controller(case):
    k = case["current"] - 1
    p = case["pending"][k]
    q = case["pending"][k + 1] if k + 1 < len(case["pending"]) else []
    pn, pe = risks(p, case["alpha"], case["beta"])
    _, next_pe = risks(q, case["alpha"], case["beta"])
    if p:
        assert case["pn_current"] == pytest.approx(pn, abs=1e-12)
        assert case["pe_negligible"] == pytest.approx(pe, abs=1e-12)
    if q:
        assert case["pe_next"] == pytest.approx(next_pe, abs=1e-12)
    # Independent transcription of recovered late decision branches, including
    # hard-coded dose10 and complemented predictive threshold defects.
    if case["current"] == 10:
        expected = 6.1 if not p else 6.11 if pe <= 0.95 else 6.12
    elif not p:
        expected = 6.0 if k + 1 == len(case["pending"]) else 6.23 if not q else 6.2
        if q and next_pe > 0.05:
            expected = 6.3
    elif pn < 0.05:
        expected = 6.4
    elif k + 1 == len(case["pending"]):
        expected = 6.21
    elif not q:
        expected = 6.22
    else:
        expected = 6.2 if next_pe <= 0.95 else 6.3
    assert case["code"] == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "prt-desktop-pending.json").read_text())["cases"]
)
def test_pending_risks_are_unprojected_posterior_means(case):
    import numpy as np
    from scipy.special import ndtr

    beta = np.asarray(case["coefficients"])
    C, D = beta.shape[1:]
    for k in range(1, D + 1):
        expected = [
            float(np.mean(1 - np.prod(ndtr(-beta[:, h:, k - 1]), axis=1)))
            for d, h, z in case["rows"]
            if d == k and z == 1 and h < C
        ]
        # Native normal-CDF approximation: measured grid maximum3.50e-7.
        # This tolerance is not a general numerical-error guarantee.
        np.testing.assert_allclose(expected, case["mean_pending_risks"][k - 1], rtol=0, atol=5e-7)
    # Deliberately decreasing dose risks survive, proving no isotonic transform
    # is applied to these patient-level posterior means.
    first = [values[0] for values in case["mean_pending_risks"]]
    assert np.all(np.diff(first) < 0)


@pytest.mark.parametrize(
    "c", json.loads((ROOT / "prt-desktop-endpoints.json").read_text())["cases"]
)
def test_native_endpoint_jitter_can_make_predictive_mass_exceed_one(c):
    import numpy as np

    small = c["uniform"] * float(np.float32(1e-5))
    current_mass = math.prod(1 + small if p == 0 else 2 - small for p in c["pending"][0])
    next_small = c["uniform"] * float(np.float32(1e-7))
    next_mass = math.prod(1 + next_small if p == 0 else 2 for p in c["pending"][1])
    # These beta moments put every updated count in the same strict category.
    assert c["pn_current"] == pytest.approx(current_mass if c["alpha"] < 1 else 0, abs=1e-12)
    assert c["pe_negligible"] == pytest.approx(current_mass if c["alpha"] > 1 else 0, abs=1e-12)
    assert c["pe_next"] == pytest.approx(next_mass if c["alpha"] > 1 else 0, abs=1e-12)
    assert current_mass > 1
