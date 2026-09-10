"""PerfectMatch PDNN energies, predicted signals and conditional expression estimates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import _owned


def pdnn_binding_energy(
    sequences: ArrayLike, stacking_energy: ArrayLike, position_weights: ArrayLike
) -> FloatArray:
    """Weighted nearest-neighbor energies for 25-base probes, read 5' to 3'.

    stacking_energy is 4x4 with both axes ordered A,C,G,T. Position weights
    have length 24. These are fitted PDNN energy parameters, not solution energies.
    """
    seq = np.asarray(sequences, dtype=str)
    energy, weights = (
        finite(stacking_energy, "stacking_energy"),
        finite(position_weights, "position_weights"),
    )
    if seq.ndim != 1 or seq.size == 0 or np.any(np.char.str_len(seq) != 25):
        raise ValueError("sequences must be a nonempty vector of 25-base probes")
    if energy.shape != (4, 4) or weights.shape != (24,):
        raise ValueError("stacking_energy must be 4x4 and position_weights length 24")
    try:
        encoded = np.frombuffer("".join(seq).upper().encode("ascii"), dtype=np.uint8).reshape(
            -1, 25
        )
    except UnicodeEncodeError as exc:
        raise ValueError("probe sequences may contain only A,C,G,T") from exc
    lookup = np.full(256, -1, dtype=int)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    bases = lookup[encoded]
    if np.any(bases < 0):
        raise ValueError("probe sequences may contain only A,C,G,T")
    result = np.sum(energy[bases[:, :-1], bases[:, 1:]] * weights, axis=1)
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("binding energies cannot be represented")
    return _owned(result)


def pdnn_signal(
    specific_energy: ArrayLike,
    nonspecific_energy: ArrayLike,
    *,
    log_expression: ArrayLike,
    nonspecific_amount: float,
    background: float,
) -> FloatArray:
    """Equation (1): N*expit(-E) + Nstar*expit(-Estar) + B, in stable log space."""
    e, ns, ln = np.broadcast_arrays(
        finite(specific_energy, "specific_energy"),
        finite(nonspecific_energy, "nonspecific_energy"),
        finite(log_expression, "log_expression"),
    )
    amount, base = (
        scalar(nonspecific_amount, "nonspecific_amount"),
        scalar(background, "background"),
    )
    if amount < 0 or base < 0:
        raise ValueError("nonspecific amount and background must be nonnegative")
    log_ns = -np.inf if amount == 0 else np.log(amount)
    log_base = -np.inf if base == 0 else np.log(base)
    logs = np.logaddexp(
        np.logaddexp(ln - np.logaddexp(0, e), log_ns - np.logaddexp(0, ns)), log_base
    )
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(logs)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ArithmeticError("predicted PDNN signal cannot be represented")
    return _owned(result)


@dataclass(frozen=True)
class PDNNExpression:
    probeset_ids: NDArray[np.int64]
    log_expression: FloatArray
    fitted_signal: FloatArray
    included: NDArray[np.bool_]
    probes_used: NDArray[np.int64]


def pdnn_expression(
    intensities: ArrayLike,
    probeset_ids: ArrayLike,
    specific_energy: ArrayLike,
    nonspecific_energy: ArrayLike,
    *,
    nonspecific_amount: float,
    background: float,
    reference_fitted: ArrayLike | None = None,
    fitness: float | None = None,
) -> PDNNExpression:
    """Equation (5), conditional on supplied energies, Nstar and background.

    Input is one array's positive probe intensities. Integer probeset IDs need not
    be consecutive; output IDs are sorted. Negative gene-specific residuals are
    excluded. Supply both reference_fitted and global log-MSE fitness to additionally
    apply the paper's three-sigma outlier rule. No optimizer or rescaling is inferred.
    """
    x = finite(intensities, "intensities")
    ids = count(probeset_ids, "probeset_ids")
    e, ns = (
        finite(specific_energy, "specific_energy"),
        finite(nonspecific_energy, "nonspecific_energy"),
    )
    if (
        x.ndim != 1
        or x.size == 0
        or any(v.shape != x.shape for v in [ids, e, ns])
        or np.any(x <= 0)
    ):
        raise ValueError("require equal nonempty probe vectors and positive intensities")
    amount, base = (
        scalar(nonspecific_amount, "nonspecific_amount"),
        scalar(background, "background"),
    )
    if amount < 0 or base < 0:
        raise ValueError("nonspecific amount and background must be nonnegative")
    log_affinity = -np.logaddexp(0, e)
    with np.errstate(under="ignore"):
        other = amount * np.exp(-np.logaddexp(0, ns)) + base
    if np.any(~np.isfinite(other)):
        raise ArithmeticError("nonspecific PDNN signal cannot be represented")
    residual = x - other
    included = residual >= 0
    if (reference_fitted is None) != (fitness is None):
        raise ValueError("supply reference_fitted and fitness together")
    if reference_fitted is not None:
        ref = finite(reference_fitted, "reference_fitted")
        assert fitness is not None
        f = scalar(fitness, "fitness")
        if ref.shape != x.shape or np.any(ref <= 0) or f < 0:
            raise ValueError("reference_fitted must be positive per probe and fitness nonnegative")
        included &= np.abs(np.log(x) - np.log(ref)) <= 3 * np.sqrt(f)
    labels, inverse = np.unique(ids.astype(np.int64), return_inverse=True)
    used = np.bincount(inverse[included], minlength=len(labels))
    positive = included & (residual > 0)
    good = np.bincount(inverse[positive], minlength=len(labels)) > 0
    if not np.all(good):
        raise ValueError(
            f"probeset {labels[np.flatnonzero(~good)[0]]} has no usable positive specific signal"
        )
    log_weight = 0.5 * (log_affinity - np.log(x))

    def grouped_sum(values: FloatArray, groups: NDArray[np.int64]) -> FloatArray:
        maximum = np.full(len(labels), -np.inf)
        np.maximum.at(maximum, groups, values)
        totals = np.bincount(
            groups, weights=np.exp(values - maximum[groups]), minlength=len(labels)
        )
        return maximum + np.log(totals)

    numerator = grouped_sum(np.log(residual[positive]) + log_weight[positive], inverse[positive])
    denominator = grouped_sum(log_affinity[included] + log_weight[included], inverse[included])
    log_n = numerator - denominator
    if np.any(~np.isfinite(log_n)):
        raise ArithmeticError("PDNN expression estimate cannot be represented")
    fitted = pdnn_signal(
        e, ns, log_expression=log_n[inverse], nonspecific_amount=amount, background=base
    )
    for v in [labels, included, used]:
        v.flags.writeable = False
    return PDNNExpression(labels, _owned(log_n), fitted, included, used)
