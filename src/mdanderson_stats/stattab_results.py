"""STATTAB distribution dispatch, source columns and neighboring count rows."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_beta import CDFBeta, cdf_beta
from .cdflib_binomial import CDFBinomial, cdf_binomial
from .cdflib_chisq import CDFChiSquare, cdf_chisq
from .cdflib_f import CDFF, cdf_f
from .cdflib_gamma import CDFGamma, cdf_gamma
from .cdflib_nc_chisq import CDFNoncentralChiSquare, cdf_nc_chisq
from .cdflib_nc_f import CDFNoncentralF, cdf_nc_f
from .cdflib_nc_t import CDFNoncentralT, cdf_nc_t
from .cdflib_neg_binomial import CDFNegativeBinomial, cdf_neg_binomial
from .cdflib_normal import CDFNormal, cdf_normal
from .cdflib_poisson import CDFPoisson, _parameter, cdf_poisson
from .cdflib_t import CDFStudentT, cdf_t
from .dcdflib_poisson import cdfpoi
from .stattab_probability import (
    stattab_binomial_term,
    stattab_negative_binomial_term,
    stattab_poisson_term,
)

CDFResult = (
    CDFBeta
    | CDFBinomial
    | CDFNegativeBinomial
    | CDFChiSquare
    | CDFNoncentralChiSquare
    | CDFF
    | CDFNoncentralF
    | CDFGamma
    | CDFNormal
    | CDFPoisson
    | CDFStudentT
    | CDFNoncentralT
)


@dataclass(frozen=True)
class STATTABDistribution:
    """Menu identity, source input order and kernel-ordered computed groups."""

    menu: int
    name: str
    parameters: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]


_TAILS = ("cum", "ccum")
_CHANCE = ("pr", "cpr")
STATTAB_DISTRIBUTIONS: Mapping[str, STATTABDistribution] = MappingProxyType(
    {
        name: STATTABDistribution(i, name, parameters + _TAILS, (_TAILS,) + groups)
        for i, (name, parameters, groups) in enumerate(
            [
                ("beta", ("x", "cx", "a", "b"), (("x", "cx"), ("a",), ("b",))),
                ("binomial", ("s", "n", "pr", "cpr"), (("s",), ("n",), _CHANCE)),
                ("neg_binomial", ("f", "s", "pr", "cpr"), (("f",), ("s",), _CHANCE)),
                ("chisq", ("x", "df"), (("x",), ("df",))),
                ("nc_chisq", ("x", "df", "pnonc"), (("x",), ("df",), ("pnonc",))),
                ("f", ("f", "dfn", "dfd"), (("f",),)),
                ("nc_f", ("f", "dfn", "dfd", "pnonc"), (("f",), ("pnonc",))),
                ("gamma", ("x", "rate", "shape"), (("x",), ("shape",), ("rate",))),
                ("normal", ("x", "mean", "sd"), (("x",), ("mean",), ("sd",))),
                ("poisson", ("s", "mean"), (("s",), ("mean",))),
                ("t", ("t", "df"), (("t",), ("df",))),
                ("nc_t", ("t", "df", "pnonc"), (("t",), ("df",), ("pnonc",))),
            ],
            1,
        )
    }
)
_KERNELS: Mapping[str, Callable[..., CDFResult]] = MappingProxyType(
    dict(
        zip(
            STATTAB_DISTRIBUTIONS,
            (
                cdf_beta,
                cdf_binomial,
                cdf_neg_binomial,
                cdf_chisq,
                cdf_nc_chisq,
                cdf_f,
                cdf_nc_f,
                cdf_gamma,
                cdf_normal,
                cdf_poisson,
                cdf_t,
                cdf_nc_t,
            ),
            strict=True,
        )
    )
)


@dataclass(frozen=True)
class STATTABNeighbor:
    """Candidate count and valid compact rows, indexed into the flattened batch.

    Invalid candidates have no numerical row. A false valid entry means the
    candidate is outside the count domain or floor+1 is not representable.
    """

    kind: str
    candidate: FloatArray
    valid: NDArray[np.bool_]
    source_indices: NDArray[np.intp]
    values: FloatArray


@dataclass(frozen=True)
class STATTABResult:
    """Completed parameters plus source-ordered output and optional neighbor rows."""

    distribution: str
    computed: tuple[str, ...]
    parameters: Mapping[str, FloatArray]
    columns: tuple[str, ...]
    values: FloatArray
    neighbors: tuple[STATTABNeighbor, ...]


def _poisson_forward(parameters: Mapping[str, ArrayLike]) -> CDFPoisson:
    # Extend only the zero-mean boundary; retain the CDFLIB90 positive-mean domain.
    count, mean = np.broadcast_arrays(
        _parameter(parameters["s"], "s"), finite(parameters["mean"], "mean")
    )
    zero = mean == 0
    positive = cdf_poisson(s=count[~zero], mean=mean[~zero])
    degenerate = cdfpoi(s=count[zero], mean=mean[zero])
    p, q = np.empty(count.shape), np.empty(count.shape)
    p[zero], q[zero] = degenerate.p, degenerate.q
    p[~zero], q[~zero] = positive.cum, positive.ccum
    return CDFPoisson(1, *map(_freeze, (p, q, count, mean)))


def _evaluate(
    name: str,
    which: int,
    parameters: Mapping[str, ArrayLike],
    df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
) -> Mapping[str, FloatArray]:
    if name == "poisson" and which == 1:
        result: CDFResult = _poisson_forward(parameters)
    elif df_bracket is not None:
        result = cdf_nc_t(which, **parameters, df_bracket=df_bracket)
    else:
        result = _KERNELS[name](which, **parameters)
    values = {key: getattr(result, key) for key in STATTAB_DISTRIBUTIONS[name].parameters}
    if any(np.any(~np.isfinite(value)) for value in values.values()):
        raise ArithmeticError("STATTAB kernel returned a nonfinite parameter")
    return MappingProxyType(values)


def _neighbors(
    name: str,
    computed: str,
    parameters: Mapping[str, FloatArray],
    columns: tuple[str, ...],
) -> tuple[STATTABNeighbor, ...]:
    count = parameters[computed]
    floor = np.floor(count)
    limit = 1e100 if name == "poisson" else 1e10
    rows = []
    for kind, candidate in (("floor", floor), ("floor_plus_one", floor + 1)):
        valid = (candidate >= 0) & (candidate <= limit)
        if kind == "floor_plus_one":
            valid &= candidate - floor == 1
        if name == "binomial":
            valid &= (
                candidate <= parameters["n"] if computed == "s" else candidate >= parameters["s"]
            )
        inputs = {key: value[valid] for key, value in parameters.items() if key not in _TAILS}
        inputs[computed] = candidate[valid]
        solved = _evaluate(name, 1, inputs)
        matrix = _freeze(np.stack([solved[key] for key in columns], axis=-1))
        indices = np.flatnonzero(valid)
        rows.append(
            STATTABNeighbor(
                kind,
                _freeze(candidate),
                np.frombuffer(valid.tobytes(), dtype=bool).reshape(valid.shape),
                np.frombuffer(indices.tobytes(), dtype=np.intp),
                matrix,
            )
        )
    return tuple(rows)


def stattab_solve(
    distribution: str,
    *,
    compute: str = "cum",
    df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
    **parameters: ArrayLike,
) -> STATTABResult:
    """Solve one of STATTAB's 42 parameter groups over a broadcast batch.

    Use canonical descriptor names; either member selects a complementary group.
    Omit the computed group and supply every other parameter, except one member
    of a complementary pair may be omitted. Gamma uses explicit rate/shape names.
    Forward calls include discrete terms or two-sided normal/t p-values. Count
    inversions retain their continuous result and separate neighboring rows.
    """
    if not isinstance(distribution, str) or distribution not in STATTAB_DISTRIBUTIONS:
        raise ValueError("unknown STATTAB distribution")
    descriptor = STATTAB_DISTRIBUTIONS[distribution]
    if not isinstance(compute, str):
        raise ValueError("compute must name a supported parameter")
    which = next((i for i, group in enumerate(descriptor.groups, 1) if compute in group), 0)
    if which == 0:
        raise ValueError(f"{distribution} does not support computing {compute}")
    computed = descriptor.groups[which - 1]
    if df_bracket is not None and (distribution != "nc_t" or compute != "df"):
        raise ValueError("df_bracket is only valid for noncentral-t df inversion")
    if any(value is None for value in parameters.values()):
        raise ValueError("omit missing complements instead of supplying None")
    if set(parameters) - set(descriptor.parameters):
        raise ValueError("unknown STATTAB parameter")
    if any(key in parameters for key in computed):
        raise ValueError("omit the parameter group being computed")
    # A pair is one required input group. Non-invertible F degrees of freedom
    # are still required inputs, although absent from the computed groups.
    for key in descriptor.parameters:
        if key in computed or key in parameters:
            continue
        partner = next((g for g in descriptor.groups if key in g and len(g) == 2), ())
        if not any(k in parameters for k in partner):
            raise ValueError(f"{key} is required")
    solved = _evaluate(distribution, which, parameters, df_bracket)
    order = descriptor.parameters
    if distribution == "gamma":
        order = ("x", "shape", "rate", "cum", "ccum")
    extra: dict[str, FloatArray] = {}
    if which == 1:
        if distribution in ("normal", "t"):
            extra["two_sided_p"] = 2 * np.minimum(solved["cum"], solved["ccum"])
        elif distribution == "binomial":
            extra["term"] = stattab_binomial_term(
                solved["s"], solved["n"], solved["pr"], solved["cpr"]
            )
        elif distribution == "neg_binomial":
            extra["term"] = stattab_negative_binomial_term(
                solved["f"], solved["s"], solved["pr"], solved["cpr"]
            )
        elif distribution == "poisson":
            extra["term"] = stattab_poisson_term(solved["s"], solved["mean"])
        columns = order + tuple(extra)
    else:
        columns = _TAILS + tuple(key for key in order if key not in _TAILS + computed) + computed
    outputs = dict(solved) | extra
    matrix = _freeze(np.stack([outputs[key] for key in columns], axis=-1))
    neighbors: tuple[STATTABNeighbor, ...] = ()
    if (distribution in ("binomial", "neg_binomial") and which in (2, 3)) or (
        distribution == "poisson" and which == 2
    ):
        neighbors = _neighbors(distribution, computed[0], solved, columns)
    return STATTABResult(distribution, computed, solved, columns, matrix, neighbors)
