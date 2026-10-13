"""Independent byte layout, native supplied-draw references and scientific MAT reports."""

import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from mdanderson_stats import (
    wfmm_basis,
    wfmm_decode_draws,
    wfmm_encode_draws,
    wfmm_summarize,
    wfmm_summary_mat,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/wfmm-native-files.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_multieffect_chain_order_and_report_moments(case):
    coefficients = np.array(case["coefficients"])
    curves = np.array(case["curves"])
    for chain in range(case["chains"]):
        block = coefficients[chain * 40 : (chain + 1) * 40]
        assert hashlib.sha256(wfmm_encode_draws(block)).hexdigest() == case["input_sha256"][chain]
    payload = wfmm_encode_draws(curves)
    assert hashlib.sha256(payload).hexdigest() == case["output_sha256"]
    decoded = wfmm_decode_draws(
        payload, draws=len(curves), components=case["components"], coordinates=8
    )
    np.testing.assert_array_equal(decoded, curves)
    assert decoded.flags.owndata and not decoded.flags.writeable
    # Re-split deterministic supplied draws into two chains only for Python diagnostics.
    summary = wfmm_summarize(
        coefficients.reshape(2, -1, case["components"], 8), wfmm_basis(8, levels=1, filter_length=2)
    )
    mat = loadmat(io.BytesIO(wfmm_summary_mat(summary)))
    np.testing.assert_allclose(mat["Lbeta_mean"], case["mean"], atol=3e-15)
    np.testing.assert_allclose(mat["Lbeta_sd"], case["sd"], atol=3e-15)
    np.testing.assert_array_equal(mat["Lbeta_quantiles"], summary.quantiles.reshape(3, -1))
    assert "scientific_not_native" in mat["summary_profile"].item()


@pytest.mark.parametrize("order,char", [("little", "<"), ("big", ">")])
def test_independent_struct_order_and_noncontiguous_input(order, char):
    values = np.arange(24, dtype=float).reshape(2, 3, 4)[:, :, ::-1]
    payload = wfmm_encode_draws(values, byte_order=order)
    assert payload == struct.pack(char + "d" * 24, *values.ravel())
    np.testing.assert_array_equal(
        wfmm_decode_draws(payload, draws=2, components=3, coordinates=4, byte_order=order), values
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00" * 7,
        b"\x00" * 9,
        struct.pack("<d", float("nan")),
        struct.pack("<d", float("inf")),
        bytearray(8),
    ],
)
def test_invalid_or_truncated_blocks(payload):
    with pytest.raises(ValueError):
        wfmm_decode_draws(payload, draws=1, components=1, coordinates=1)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, 2_000_001])
def test_shape_bounds_before_decoding(bad):
    with pytest.raises(ValueError):
        wfmm_decode_draws(b"", draws=bad, components=1, coordinates=1)


def test_product_budget_and_invalid_encoder_inputs():
    with pytest.raises(ValueError, match="cells"):
        wfmm_decode_draws(b"", draws=2000, components=2000, coordinates=1)
    for value in [np.ones((2, 3)), np.ones((1, 1, 1), dtype=bool), [[[np.inf]]], [[[1j]]]]:
        with pytest.raises(ValueError):
            wfmm_encode_draws(value)
    with pytest.raises(ValueError):
        wfmm_encode_draws([[[1]]], byte_order="native")


@pytest.mark.parametrize("prefix", ["Lbeta", "LbetaLt", "U"])
def test_scientific_fields_have_explicit_shapes_and_probabilities(prefix):
    draws = np.random.default_rng(2).normal(size=(2, 20, 2, 8))
    summary = wfmm_summarize(
        draws, wfmm_basis(8, levels=1, filter_length=2), effect_sizes=[0, 0.5], confidence=0.9
    )
    data = loadmat(io.BytesIO(wfmm_summary_mat(summary, prefix=prefix)))
    assert data[prefix + "_mean"].shape == (2, 8)
    assert data[prefix + "_quantiles"].shape == (3, 16)
    assert data[prefix + "_peffects"].shape == (2, 16)
    assert data[prefix + "_lowerCI"].shape == (1, 16)
    np.testing.assert_allclose(data["alpha"], [[0.1]])
    np.testing.assert_array_equal(data[prefix + "_simbas"], summary.simbas_probability)
    with pytest.raises(ValueError):
        wfmm_summary_mat(summary, prefix="unsafe")
