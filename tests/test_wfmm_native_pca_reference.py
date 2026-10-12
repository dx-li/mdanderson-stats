"""Independent matrix checks for native PCA/hybrid reference transforms."""

import json
from pathlib import Path

import numpy as np
import pytest

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-pca.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_native_pca_and_hybrid_matrix_contracts(case):
    if case["exit_code"]:
        assert case["settings"]["alphaPC"] == 1 or case["case"] == "zero_energy"
        assert any("ColQty=0" in e for e in case["errors"])
        return
    y = np.array(case["y"])
    a = np.array(case["analysis"])
    s = np.array(case["synthesis"])
    d = np.array(case["D"])
    typ = case["settings"]["transformtype"]
    k = a.shape[1]
    np.testing.assert_allclose(s, a.T, atol=1e-14)
    np.testing.assert_allclose(s @ a, np.eye(k), atol=2e-14)
    if typ == "PC":
        transformed = y
        np.testing.assert_allclose(d, y @ a, atol=2e-13)
    else:
        # One-level periodic Haar, independently written as pair sums/differences.
        w = np.array([[1, 0, 1, 0], [1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1]]) / np.sqrt(2)
        if typ == "PCw":
            transformed = y
            np.testing.assert_allclose(d[:, :k], y @ a, atol=2e-13)
            np.testing.assert_allclose(d[:, k:], (y - y @ a @ s) @ w, atol=2e-13)
            np.testing.assert_allclose(d[:, :k] @ s + d[:, k:] @ w.T, y, atol=2e-13)
        else:
            transformed = y @ w
            np.testing.assert_allclose(d, transformed @ a, atol=2e-13)
    _, singular, vt = np.linalg.svd(transformed, full_matrices=False)
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)
    retained = np.flatnonzero(cumulative > case["settings"]["alphaPC"])[0] + 1
    assert k == retained
    # PCA sign choices differ by SVD implementation; its retained subspace is
    # the invariant scientific object. No centering is applied by the source.
    np.testing.assert_allclose(a @ s, vt[:k].T @ vt[:k], atol=3e-13)
