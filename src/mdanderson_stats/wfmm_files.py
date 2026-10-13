"""Explicit bounded WFMM double-block interchange and scientific MAT summaries."""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import savemat

from ._validation import FloatArray
from .wfmm_posterior import WFMMPosteriorSummary

_MAX_CELLS = 2_000_000


def _order(byte_order):
    if byte_order not in ("little", "big"):
        raise ValueError("byte_order must be little or big")
    return "<f8" if byte_order == "little" else ">f8"


def _shape(draws, components, coordinates):
    shape = (draws, components, coordinates)
    if any(
        isinstance(x, (bool, np.bool_))
        or not isinstance(x, (int, np.integer))
        or x < 1
        or x > _MAX_CELLS
        for x in shape
    ):
        raise ValueError("draw dimensions must be positive bounded integers")
    if int(draws) * int(components) * int(coordinates) > _MAX_CELLS:
        raise ValueError("draw blocks exceed 2000000 cells")
    return tuple(int(x) for x in shape)


def wfmm_encode_draws(draws: ArrayLike, *, byte_order: str = "little") -> bytes:
    """Encode samples × components × coordinates as headerless IEEE doubles.

    Coordinates vary fastest, then components, then samples. The little-endian
    profile matches the recovered Linux x86_64 files. The generic big-endian
    option is explicit byte conversion, not a validated native-platform claim.
    No chain labels, dimensions, basis, burn-in or thinning metadata is stored.
    Pass one chain or explicitly concatenate already retained chains yourself.
    """
    dtype = _order(byte_order)
    raw = np.asarray(draws)
    if raw.ndim != 3 or raw.dtype.kind not in "iuf":
        raise ValueError("draws must be a real samples-by-components-by-coordinates array")
    _shape(*raw.shape)
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("draws must be finite")
    return values.astype(dtype, copy=False).tobytes(order="C")


def wfmm_decode_draws(
    payload: bytes,
    *,
    draws: int,
    components: int,
    coordinates: int,
    byte_order: str = "little",
) -> FloatArray:
    """Decode an exactly sized native double block using caller-supplied shape.

    A wrong but equally sized shape or byte order cannot be detected from a
    headerless file; retain authoritative study metadata separately. Truncated,
    trailing and nonfinite data are rejected. The result is owned and read-only.
    """
    dtype = _order(byte_order)
    shape = _shape(draws, components, coordinates)
    if not isinstance(payload, bytes):
        raise ValueError("payload must be bytes")
    expected = 8 * shape[0] * shape[1] * shape[2]
    if len(payload) != expected:
        raise ValueError("payload length does not exactly match the declared draw dimensions")
    result = np.array(
        np.frombuffer(payload, dtype=dtype).reshape(shape), dtype=np.float64, copy=True
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("draw block contains nonfinite values")
    result.flags.writeable = False
    return result


def wfmm_summary_mat(summary: WFMMPosteriorSummary, *, prefix: str = "Lbeta") -> bytes:
    """Export scientific Python summaries under selected native-style MAT names.

    This is a report interchange subset, not an Init file or native numerical
    compatibility profile. Means/SDs use effects × coordinates; quantile and
    effect-size rows pack effects with coordinates fastest. Native quantile,
    integer-absolute-value and zero-SD defects are not reproduced. Correctness,
    native file naming and clinical validity are separate claims.
    """
    if not isinstance(summary, WFMMPosteriorSummary) or prefix not in ("Lbeta", "LbetaLt", "U"):
        raise ValueError("provide a WFMMPosteriorSummary and prefix Lbeta, LbetaLt or U")
    fields = {
        "mean": summary.mean,
        "sd": summary.standard_deviation,
        "quantiles": summary.quantiles.reshape(len(summary.quantile_probabilities), -1),
        "peffects": summary.effect_size_probability.reshape(len(summary.effect_sizes), -1),
        "p0": summary.sign_tail_score,
        "simbas": summary.simbas_probability,
        "lowerCI": summary.simultaneous_lower.reshape(1, -1),
        "upperCI": summary.simultaneous_upper.reshape(1, -1),
    }
    if sum(np.asarray(x).size for x in fields.values()) > _MAX_CELLS:
        raise ValueError("summary export exceeds 2000000 cells")
    data: dict = {prefix + "_" + key: value for key, value in fields.items()}
    data.update(
        quantiles=np.asarray(summary.quantile_probabilities),
        effect_size=np.asarray(summary.effect_sizes),
        alpha=np.array([1 - summary.confidence]),
        summary_profile="mdanderson_stats_scientific_not_native_numerical_identity",
        clinical_validity="not_established",
    )
    target = io.BytesIO()
    savemat(target, data, format="5", do_compression=False, oned_as="row")
    return target.getvalue()
