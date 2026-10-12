"""Native named-family transforms and independent filter identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    WFMMStudy,
    read_wfmm_study,
    run_wfmm_study,
    wfmm_extended_wavelet_basis,
    wfmm_inverse,
    wfmm_named_wavelet_basis,
    wfmm_transform,
)

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-families.json").read_text())[
    "cases"
]
NAMES = [*(f"sym{i}" for i in range(2, 9)), *(f"coif{i}" for i in range(1, 6))]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['wavelet']}-{c['size']}-{c['boundary']}")
def test_native_named_analysis_and_synthesis(case):
    basis = wfmm_named_wavelet_basis(
        case["size"], wavelet=case["wavelet"], levels=case["levels"], boundary=case["boundary"]
    )
    np.testing.assert_allclose(basis.custom_matrix, case["analysis"], rtol=0, atol=2e-15)
    np.testing.assert_allclose(basis.synthesis_matrix, case["synthesis"], rtol=0, atol=2e-15)
    y = np.random.default_rng(312).normal(size=(3, case["size"]))
    d = wfmm_transform(y, basis).coefficients
    np.testing.assert_allclose(wfmm_inverse(d, basis), y, rtol=0, atol=5e-8)
    assert basis.wavelet_name == case["wavelet"]


@pytest.mark.parametrize("name", NAMES)
def test_quadrature_mirror_orthogonality_and_vanishing_moments(name):
    b = wfmm_named_wavelet_basis(64, wavelet=name)
    low, high = b.scaling_filter, b.wavelet_filter
    np.testing.assert_allclose(low.sum(), np.sqrt(2), rtol=0, atol=2e-9)
    np.testing.assert_allclose(low @ low, 1, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(high, low[::-1] * ((-1.0) ** np.arange(low.size)))
    for shift in range(2, len(low), 2):
        assert abs(low[shift:] @ low[:-shift]) < 5e-9
    order = int(name[3:] if name.startswith("sym") else name[4:])
    moments = order if name.startswith("sym") else 2 * order
    positions = np.linspace(-1, 1, len(high))
    for power in range(moments):
        assert abs(high @ positions**power) < 2e-9
    assert not low.flags.writeable and not high.flags.writeable


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("haar", "db1"),
        *[(f"d{2 * i}", f"db{i}") for i in range(2, 11)],
        *[(f"s{2 * i}", f"sym{i}") for i in range(2, 9)],
        *[(f"c{6 * i}", f"coif{i}") for i in range(1, 6)],
    ],
)
def test_manual_aliases(alias, canonical):
    a = wfmm_named_wavelet_basis(32, wavelet=alias)
    b = wfmm_named_wavelet_basis(32, wavelet=canonical)
    np.testing.assert_array_equal(a.custom_matrix, b.custom_matrix)
    assert a.wavelet_name == canonical
    if canonical.startswith("db"):
        old = wfmm_extended_wavelet_basis(32, filter_length=2 * int(canonical[2:]))
        np.testing.assert_array_equal(a.custom_matrix, old.custom_matrix)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(wavelet="sym1"),
        dict(wavelet="coif6"),
        dict(wavelet=2),
        dict(time_count=1024),
        dict(levels=13),
        dict(max_work=1),
    ],
)
def test_named_validation_and_work_bounds(kwargs):
    with pytest.raises(ValueError):
        wfmm_named_wavelet_basis(**(dict(time_count=32) | kwargs))


@pytest.mark.parametrize("name", ["sym4", "coif1"])
def test_named_saved_study_and_legacy_default(name):
    y = np.random.default_rng(46).normal(size=(8, 9))
    study = WFMMStudy(
        y, np.ones((8, 1)), transform="extended_wavelet", wavelet_name=name, draws=8, warmup=4
    )
    result = run_wfmm_study(read_wfmm_study(study.to_json()))
    assert result.basis.wavelet_name == name
    assert result.fit is not None
    assert result.summary.mean.shape == (1, 9)
    older = WFMMStudy(y, np.ones((8, 1)), transform="identity")
    data = json.loads(older.to_json())
    del data["wavelet_name"]
    assert read_wfmm_study(json.dumps(data)).to_json() == older.to_json()


def test_unused_or_conflicting_saved_names_rejected():
    for kwargs in [
        dict(transform="wavelet", wavelet_name="sym4"),
        dict(transform="extended_wavelet", wavelet_name="sym4", filter_length=4),
    ]:
        with pytest.raises(ValueError):
            WFMMStudy(np.ones((8, 8)), np.ones((8, 1)), **kwargs)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("size,levels", [(2, 1), (3, 5), (2, 12)])
@pytest.mark.parametrize("boundary", ["zero", "periodic", "reflection"])
def test_short_signal_and_deep_level_reconstruction(name, size, levels, boundary):
    basis = wfmm_named_wavelet_basis(size, wavelet=name, levels=levels, boundary=boundary)
    np.testing.assert_allclose(
        basis.custom_matrix @ basis.synthesis_matrix, np.eye(size), rtol=0, atol=5e-8
    )
