"""CDFLIB archive distribution descriptors, validation and root adapters.

This namespace exposes the historical support contracts. Descriptor limits are
not a statement of the domains supported by the package's distribution APIs.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .cdflib_root import ZeroFinder, final_zf_state, set_zero_finder


@dataclass(frozen=True)
class CDFParameter:
    """Archived parameter label, inverse selector and inclusive bounds."""

    name: str
    no_check: int
    low_bound: float
    high_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("Parameter name must be a string")
        if (
            isinstance(self.no_check, bool)
            or not isinstance(self.no_check, int)
            or self.no_check < 0
        ):
            raise ValueError("no_check must be a nonnegative integer")
        low = scalar(self.low_bound, "low_bound")
        high = scalar(self.high_bound, "high_bound")
        if low > high:
            raise ValueError("Parameter bounds must be increasing or equal")
        object.__setattr__(self, "low_bound", low)
        object.__setattr__(self, "high_bound", high)


@dataclass(frozen=True)
class CDFDistribution:
    """Archived descriptor, including six slots and inactive padding."""

    name: str
    max_which: int
    nparam: int
    parameters: tuple[CDFParameter, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("Distribution name must be a string")
        if not isinstance(self.parameters, tuple) or len(self.parameters) != 6:
            raise ValueError("parameters must be a tuple of six CDFParameter objects")
        if (
            isinstance(self.nparam, bool)
            or not isinstance(self.nparam, int)
            or not 0 <= self.nparam <= 6
        ):
            raise ValueError("nparam must be an integer in [0,6]")
        if (
            isinstance(self.max_which, bool)
            or not isinstance(self.max_which, int)
            or self.max_which < 0
        ):
            raise ValueError("max_which must be a nonnegative integer")
        for parameter in self.parameters:
            if not isinstance(parameter, CDFParameter):
                raise ValueError("parameters must contain CDFParameter objects")


def add_to_one(x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
    """Test x+y=1 within the source's three-epsilon allowance; no range check."""
    a, b = np.broadcast_arrays(finite(x, "x"), finite(y, "y"))
    with np.errstate(over="ignore"):
        result = np.abs(((a + b) - 0.5) - 0.5) <= 3 * np.finfo(float).eps
    return np.frombuffer(result.tobytes(), dtype=bool).reshape(result.shape)


def dbl_in_range(value: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> NDArray[np.bool_]:
    """Broadcast inclusive finite floating-point bounds; reject inverted limits."""
    v, low, high = np.broadcast_arrays(finite(value, "value"), finite(lo, "lo"), finite(hi, "hi"))
    if np.any(low > high):
        raise ValueError("Require lo <= hi")
    result = (v >= low) & (v <= high)
    return np.frombuffer(result.tobytes(), dtype=bool).reshape(result.shape)


def int_in_range(value: int, lo: int, hi: int) -> bool:
    """Inclusive Python integer bounds without conversion through binary64."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (value, lo, hi)):
        raise ValueError("value, lo and hi must be integers")
    if lo > hi:
        raise ValueError("Require lo <= hi")
    return lo <= value <= hi


def in_range(value: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> NDArray[np.bool_] | bool:
    """Use exact scalar Python integer comparisons, otherwise floating arrays."""
    if isinstance(value, int) and isinstance(lo, int) and isinstance(hi, int):
        return int_in_range(value, lo, hi)
    return dbl_in_range(value, lo, hi)


def check_complements(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    set_values: bool = True,
) -> tuple[FloatArray, FloatArray] | None:
    """Require one coordinate and optionally copy/complete the pair.

    Like the source, this checks presence only, not range or sum. Both supplied
    coordinates are preserved. If set_values=False, only presence is checked.
    """
    if not isinstance(set_values, bool):
        raise ValueError("set_values must be boolean")
    if x is None and y is None:
        raise ValueError("Provide at least one of x and y")
    if not set_values:
        return None
    if x is None:
        assert y is not None
        a = 1 - finite(y, "y")
    else:
        a = finite(x, "x")
    b = finite(y, "y") if y is not None else 1 - a
    a, b = np.broadcast_arrays(a, b)
    return _freeze(a), _freeze(b)


def validate_parameters(
    distrib: CDFDistribution,
    which: int,
    params: ArrayLike,
) -> NDArray[np.int64]:
    """Return source failure codes or explicit zero success for (...,6) rows.

    First error wins: selector -1; noncomplementary probabilities +3; parameter
    slot i (one-based) -(i+1). Unknown and padded slots are ignored. Nonfinite
    checked values fail; ignored slots may contain NaN placeholders.
    """
    if isinstance(which, bool) or not isinstance(which, int):
        raise ValueError("which must be an integer")
    values = np.asarray(params, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 6:
        raise ValueError("params must have shape (...,6)")
    status = np.zeros(values.shape[:-1], dtype=np.int64)
    if not 1 <= which <= distrib.max_which:
        status[...] = -1
    else:
        if which != 1:
            a, b = values[..., 0], values[..., 1]
            with np.errstate(over="ignore", invalid="ignore"):
                bad = (
                    ~np.isfinite(a)
                    | ~np.isfinite(b)
                    | (np.abs(((a + b) - 0.5) - 0.5) > 3 * np.finfo(float).eps)
                )
            status = np.where(bad, 3, status)
        for i, p in enumerate(distrib.parameters[: distrib.nparam]):
            if which == p.no_check:
                continue
            value = values[..., i]
            bad = ~np.isfinite(value) | (value < p.low_bound) | (value > p.high_bound)
            status = np.where((status == 0) & bad, -(i + 2), status)
    return np.frombuffer(status.tobytes(), dtype=np.int64).reshape(status.shape)


def cdf_set_zero_finder(distrib: CDFDistribution, which: int) -> ZeroFinder:
    """Configure a search for the one-based *parameter slot*, not inverse selector."""
    if isinstance(which, bool) or not isinstance(which, int) or not 1 <= which <= distrib.nparam:
        raise ValueError("which must identify an active one-based parameter slot")
    p = distrib.parameters[which - 1]
    return set_zero_finder(
        low_limit=p.low_bound,
        hi_limit=p.high_bound,
        abs_step=0.5,
        rel_step=0.5,
        step_multiplier=5,
        abs_tol=1e-50,
        rel_tol=1e-8,
    )


def cdf_finalize_status(local: ZeroFinder) -> int:
    """Translate completed root status to 0, -50 (left) or +50 (right)."""
    result = final_zf_state(local)
    if result is None or result.status == 1:
        raise ValueError("Root search has not completed")
    if result.status == -2:
        raise ArithmeticError("Root search exhausted its evaluation budget")
    if result.status == 0:
        return 0
    return -50 if result.crash_left else 50


def which_miss(which: int, routine_name: str, arg_name: str) -> NoReturn:
    """Raise an actionable missing-argument error instead of stopping the process."""
    raise ValueError(f"{routine_name}: which={which} requires missing argument {arg_name}")


# Values below come from unchanged native descriptors, including padding and
# restrictive max_which values. See tools/reference_cdflib_aux.py for provenance.

the_beta = CDFDistribution(
    "cdf_beta",
    4,
    6,
    (
        CDFParameter("cum", 1, 0.0, 1.0),
        CDFParameter("ccum", 1, 0.0, 1.0),
        CDFParameter("x", 2, 0.0, 1.0),
        CDFParameter("cx", 2, 0.0, 1.0),
        CDFParameter("a", 3, 1e-10, 10000000000.0),
        CDFParameter("b", 4, 1e-10, 10000000000.0),
    ),
)

the_binomial = CDFDistribution(
    "cdf_binomial",
    4,
    6,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("s", 2, 0.0, 10000000000.0),
        CDFParameter("n", 3, 0.0, 10000000000.0),
        CDFParameter("pr", 4, 0.0, 1.0),
        CDFParameter("cpr", 4, 0.0, 1.0),
    ),
)

the_chi_square = CDFDistribution(
    "cdf_chi_square",
    3,
    4,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("x", 2, 0.0, 1e100),
        CDFParameter("df", 3, 0.001, 10000000000.0),
        CDFParameter("", 0, 0.0, 0.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_dummy_binomial = CDFDistribution(
    "cdf_binomial",
    0,
    2,
    (
        CDFParameter("pr", 0, 0.0, 0.5),
        CDFParameter("cpr", 0, 0.0, 0.5),
        CDFParameter("", 0, 0.0, 0.0),
        CDFParameter("", 0, 0.0, 0.0),
        CDFParameter("", 0, 0.0, 0.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_f = CDFDistribution(
    "cdf_f",
    2,
    5,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("f", 2, 0.0, 1e100),
        CDFParameter("dfn", 3, 0.001, 10000000000.0),
        CDFParameter("dfd", 4, 0.001, 10000000000.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_gamma = CDFDistribution(
    "cdf_gamma",
    4,
    5,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("x", 2, 0.0, 1e100),
        CDFParameter("shape", 3, 1e-10, 1e100),
        CDFParameter("scale", 4, 1e-10, 1e100),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_negative_binomial = CDFDistribution(
    "cdf_negative_binomial",
    4,
    6,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("f", 2, 0.0, 10000000000.0),
        CDFParameter("s", 3, 0.0, 10000000000.0),
        CDFParameter("pr", 4, 0.0, 1.0),
        CDFParameter("cpr", 4, 0.0, 1.0),
    ),
)

the_non_central_chi_square = CDFDistribution(
    "cdf_nc_chisq",
    4,
    5,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("x", 2, 0.0, 1e100),
        CDFParameter("df", 3, 0.001, 10000000000.0),
        CDFParameter("pnonc", 4, 0.0, 10000.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_non_central_f = CDFDistribution(
    "cdf_nc_f",
    3,
    6,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("f", 2, 0.0, 1e100),
        CDFParameter("dfn", 3, 0.001, 10000000000.0),
        CDFParameter("dfd", 4, 0.001, 10000000000.0),
        CDFParameter("pnonc", 5, 0.0, 10000.0),
    ),
)

the_non_central_t = CDFDistribution(
    "cdf_non_central_t",
    4,
    5,
    (
        CDFParameter("cum", 1, 1e-10, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 0.9999999999),
        CDFParameter("t", 2, -1e100, 1e100),
        CDFParameter("df", 3, 0.001, 10000000000.0),
        CDFParameter("pnonc", 4, 0.0, 10000.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_normal = CDFDistribution(
    "cdf_normal",
    4,
    5,
    (
        CDFParameter("cum", 1, 1e-10, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 0.9999999999),
        CDFParameter("x", 2, -1e100, 1e100),
        CDFParameter("mean", 3, -1e100, 1e100),
        CDFParameter("sd", 4, 1e-10, 1e100),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_poisson = CDFDistribution(
    "cdf_poisson",
    3,
    4,
    (
        CDFParameter("cum", 1, 0.0, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 1.0),
        CDFParameter("s", 2, 0.0, 1e100),
        CDFParameter("lambda", 3, 1e-10, 1e100),
        CDFParameter("", 0, 0.0, 0.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

the_t = CDFDistribution(
    "cdf_t",
    3,
    4,
    (
        CDFParameter("cum", 1, 1e-10, 0.9999999999),
        CDFParameter("ccum", 1, 1e-10, 0.9999999999),
        CDFParameter("t", 2, -1e100, 1e100),
        CDFParameter("df", 3, 0.001, 10000000000.0),
        CDFParameter("", 0, 0.0, 0.0),
        CDFParameter("", 0, 0.0, 0.0),
    ),
)

DISTRIBUTIONS = MappingProxyType(
    {
        "the_beta": the_beta,
        "the_binomial": the_binomial,
        "the_chi_square": the_chi_square,
        "the_dummy_binomial": the_dummy_binomial,
        "the_f": the_f,
        "the_gamma": the_gamma,
        "the_negative_binomial": the_negative_binomial,
        "the_non_central_chi_square": the_non_central_chi_square,
        "the_non_central_f": the_non_central_f,
        "the_non_central_t": the_non_central_t,
        "the_normal": the_normal,
        "the_poisson": the_poisson,
        "the_t": the_t,
    }
)
