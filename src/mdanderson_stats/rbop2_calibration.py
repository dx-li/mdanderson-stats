"""Finite-candidate calibration for binary rBOP2 designs.

The candidate grid is supplied by the caller. This implements the published
type-I-constrained power objective, not an undocumented native cutoff grid.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import finite, scalar
from .rbop2_binary import (
    Rbop2BinaryDesign,
    _exact_order,
    _prior,
    _probability,
    rbop2_binary_design,
)


def _freeze_bool(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    return np.frombuffer(array.tobytes(), dtype=np.bool_).reshape(array.shape)


@dataclass(frozen=True)
class Rbop2BinaryCalibration:
    """Candidate diagnostics and the selected design, if any candidate is feasible."""

    selected_index: int | None
    selected_design: Rbop2BinaryDesign | None
    calibration_prior: np.ndarray
    analysis_prior: np.ndarray
    null_rates: tuple[float, float]
    alternative_rates: tuple[float, float]
    cutoff_candidates: np.ndarray
    feasible: np.ndarray
    calibration_type_i_error: np.ndarray
    calibration_power: np.ndarray
    calibration_expected_null_sample_size: np.ndarray
    analysis_type_i_error: np.ndarray
    analysis_power: np.ndarray
    analysis_expected_null_sample_size: np.ndarray
    max_posterior_probability_error: np.ndarray
    alpha: float


def _decision_probability(
    design: Rbop2BinaryDesign,
    probabilities: tuple[np.ndarray, ...],
    errors: tuple[np.ndarray, ...],
    exact: dict[tuple[int, int, int], Fraction],
    refined: dict[tuple[int, int, int], tuple[float, float]],
    look: int,
    e: int,
    c: int,
) -> tuple[float, float, Fraction | None]:
    p, error = float(probabilities[look][e, c]), float(errors[look][e, c])
    lower, upper = float(design.lower_cutoffs[look]), float(design.upper_cutoffs[look])
    near = error > 0 and (abs(p - lower) <= error or abs(p - upper) <= error)
    exact_value: Fraction | None = None
    if near and design.margin == 0:
        shape = (
            design.prior[0, 0] + e,
            design.prior[0, 1] + int(design.looks[look, 0]) - e,
            design.prior[1, 0] + c,
            design.prior[1, 1] + int(design.looks[look, 1]) - c,
        )
        if all(float(value).is_integer() and value <= 1000 for value in shape):
            key = (look, e, c)
            exact_value = exact.get(key)
            if exact_value is None:
                exact_value = _exact_order(*map(int, shape[:2]), *map(int, shape[2:]))
                if design.endpoint == "toxicity":
                    exact_value = 1 - exact_value
                exact[key] = exact_value
            return float(exact_value), 0.0, exact_value
    if near:
        key = (look, e, c)
        if key in refined:
            p, error = refined[key]
            if error > 0 and (abs(p - lower) <= error or abs(p - upper) <= error):
                raise ArithmeticError("posterior probability is too close to cutoff to resolve")
            return p, error, None
        tighter = max(1e-12, design.absolute_tolerance / 10)
        if tighter < design.absolute_tolerance:
            p, error = _probability(
                design.endpoint,
                design.prior,
                e,
                c,
                int(design.looks[look, 0]),
                int(design.looks[look, 1]),
                design.margin,
                tighter,
            )
            refined[key] = (p, error)
            near = error > 0 and (abs(p - lower) <= error or abs(p - upper) <= error)
        if near:
            raise ArithmeticError("posterior probability is too close to cutoff to resolve")
    return p, error, None


def _surface(
    design: Rbop2BinaryDesign,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    dict[tuple[int, int, int], Fraction],
    dict[tuple[int, int, int], tuple[float, float]],
]:
    probabilities, errors = [], []
    exact: dict[tuple[int, int, int], Fraction] = {}
    refined: dict[tuple[int, int, int], tuple[float, float]] = {}
    for i, (ne0, nc0) in enumerate(design.looks):
        ne, nc = int(ne0), int(nc0)
        p = np.empty((ne + 1, nc + 1))
        err = np.empty_like(p)
        for e in range(ne + 1):
            for c in range(nc + 1):
                p[e, c], err[e, c] = _probability(
                    design.endpoint,
                    design.prior,
                    e,
                    c,
                    ne,
                    nc,
                    design.margin,
                    design.absolute_tolerance,
                )
        probabilities.append(p)
        errors.append(err)
    return tuple(probabilities), tuple(errors), exact, refined


def _tables(
    design: Rbop2BinaryDesign,
    probabilities: tuple[np.ndarray, ...],
    errors: tuple[np.ndarray, ...],
    exact: dict[tuple[int, int, int], Fraction],
    refined: dict[tuple[int, int, int], tuple[float, float]],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    result = []
    for i, (ne0, nc0) in enumerate(design.looks):
        ne, nc = int(ne0), int(nc0)
        futile = np.zeros((ne + 1, nc + 1), dtype=bool)
        superior = np.zeros_like(futile)
        lower, upper = float(design.lower_cutoffs[i]), float(design.upper_cutoffs[i])
        for e in range(ne + 1):
            for c in range(nc + 1):
                p, _, exact_value = _decision_probability(
                    design, probabilities, errors, exact, refined, i, e, c
                )
                if exact_value is None:
                    futile[e, c] = p < lower
                    superior[e, c] = p >= upper
                else:
                    futile[e, c] = exact_value < Fraction(str(lower))
                    superior[e, c] = exact_value >= Fraction(str(upper))
        result.append((futile, superior))
    return tuple(result)


def _oc(
    design: Rbop2BinaryDesign,
    rates: tuple[float, float],
    tables: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, float, float]:
    output = design._oc_one(rates[0], rates[1], tables)
    return cast(float, output[5]), cast(float, output[10]), cast(float, output[3])


def _state_work(looks: np.ndarray) -> tuple[int, int]:
    states = 0
    work = 0
    old_e = old_c = 0
    for ne0, nc0 in looks:
        ne, nc = int(ne0), int(nc0)
        states += (ne + 1) * (nc + 1)
        work += (ne + 1) * (old_e + 1) * (old_c + 1)
        work += (ne + 1) * (old_c + 1) * (nc + 1)
        old_e, old_c = ne, nc
    return states, work


def calibrate_rbop2_binary(
    looks: ArrayLike,
    *,
    calibration_prior: ArrayLike,
    endpoint: str,
    margin: float,
    null_rates: tuple[float, float],
    alternative_rates: tuple[float, float],
    cutoff_candidates: ArrayLike,
    alpha: float,
    analysis_prior: ArrayLike | None = None,
    absolute_tolerance: float = 1e-9,
    max_work: int = 20_000_000,
) -> Rbop2BinaryCalibration:
    """Select the supplied cutoff curve with greatest power subject to null error <= alpha.

    Candidate shape is ``(candidate, look, 2)`` with lower/upper cutoffs.
    Calibration and optional analysis priors are explicit; the returned analysis
    operating characteristics need not retain the calibration alpha when priors differ.
    Ties choose smaller expected total enrollment under the calibration null, then
    the earlier candidate in input order. No randomness is used.
    """
    raw_looks = np.asarray(looks)
    raw_candidates = np.asarray(cutoff_candidates)
    if raw_looks.ndim != 2 or raw_looks.shape[1] != 2 or not 1 <= raw_looks.shape[0] <= 200:
        raise ValueError("looks must have shape (look, 2), with 1 to 200 looks")
    if (
        raw_candidates.ndim != 3
        or not 1 <= raw_candidates.shape[0] <= 1000
        or raw_candidates.shape[1:] != (raw_looks.shape[0], 2)
    ):
        raise ValueError("cutoff_candidates must have shape (candidate, look, 2)")
    if np.iscomplexobj(raw_candidates):
        raise ValueError("cutoff_candidates must be real")
    candidates = finite(raw_candidates, "cutoff_candidates")
    if (
        np.any((candidates < 0) | (candidates > 1))
        or np.any(candidates[:, :, 0] > candidates[:, :, 1])
        or np.any(candidates[:, -1, 0] != candidates[:, -1, 1])
    ):
        raise ValueError("candidate lower/upper cutoffs are invalid")
    calibration = _prior(calibration_prior)
    analysis = calibration if analysis_prior is None else _prior(analysis_prior)
    raw_null, raw_alt = np.asarray(null_rates), np.asarray(alternative_rates)
    if np.iscomplexobj(raw_null) or np.iscomplexobj(raw_alt):
        raise ValueError("scenario rates must be real")
    if raw_null.shape != (2,) or raw_alt.shape != (2,):
        raise ValueError("null_rates and alternative_rates must each be (experimental, control)")
    null_array, alt_array = finite(raw_null, "null_rates"), finite(raw_alt, "alternative_rates")
    null = (float(null_array[0]), float(null_array[1]))
    alt = (float(alt_array[0]), float(alt_array[1]))
    if any(x < 0 or x > 1 for x in (*null, *alt)):
        raise ValueError("scenario rates must be in [0,1]")
    alpha_value = scalar(alpha, "alpha")
    if not 0 < alpha_value < 1:
        raise ValueError("alpha must be strictly between zero and one")
    budget = scalar(max_work, "max_work")
    if int(budget) != budget or budget < 1:
        raise ValueError("max_work must be a positive integer")

    template = rbop2_binary_design(
        raw_looks,
        prior=calibration,
        endpoint=endpoint,
        margin=margin,
        lower_cutoffs=candidates[0, :, 0],
        upper_cutoffs=candidates[0, :, 1],
        absolute_tolerance=absolute_tolerance,
    )
    states, transition_work = _state_work(template.looks)
    if transition_work > 5_000_000:
        raise ValueError("operating-characteristic state space exceeds 5000000")
    priors = 1 if np.array_equal(calibration, analysis) else 2
    total_work = 2 * priors * states + candidates.shape[0] * priors * (states + 2 * transition_work)
    if total_work > budget:
        raise ValueError(f"calibration exceeds max_work={int(budget)}")

    surfaces = [_surface(template)]
    if priors == 2:
        analysis_template = rbop2_binary_design(
            template.looks,
            prior=analysis,
            endpoint=endpoint,
            margin=margin,
            lower_cutoffs=candidates[0, :, 0],
            upper_cutoffs=candidates[0, :, 1],
            absolute_tolerance=absolute_tolerance,
        )
        surfaces.append(_surface(analysis_template))
    k = candidates.shape[0]
    cal_i = np.empty(k)
    cal_power = np.empty(k)
    cal_n = np.empty(k)
    ana_i = np.empty(k)
    ana_power = np.empty(k)
    ana_n = np.empty(k)
    errors = np.empty(k)
    feasible = np.empty(k, dtype=bool)
    for j in range(k):
        design = rbop2_binary_design(
            template.looks,
            prior=calibration,
            endpoint=endpoint,
            margin=margin,
            lower_cutoffs=candidates[j, :, 0],
            upper_cutoffs=candidates[j, :, 1],
            absolute_tolerance=absolute_tolerance,
        )
        p, err, exact, refined = surfaces[0]
        table = _tables(design, p, err, exact, refined)
        cal_i[j], cal_n[j], _ = _oc(design, null, table)
        cal_power[j], _, _ = _oc(design, alt, table)
        errors[j] = max(float(np.max(x)) for x in err)
        if priors == 1:
            ana_i[j], ana_n[j] = cal_i[j], cal_n[j]
            ana_power[j] = cal_power[j]
        else:
            analysis_design = rbop2_binary_design(
                template.looks,
                prior=analysis,
                endpoint=endpoint,
                margin=margin,
                lower_cutoffs=candidates[j, :, 0],
                upper_cutoffs=candidates[j, :, 1],
                absolute_tolerance=absolute_tolerance,
            )
            ap, ae, ax, arefined = surfaces[1]
            analysis_table = _tables(analysis_design, ap, ae, ax, arefined)
            errors[j] = max(errors[j], *(float(np.max(x)) for x in ae))
            ana_i[j], ana_n[j], _ = _oc(analysis_design, null, analysis_table)
            ana_power[j], _, _ = _oc(analysis_design, alt, analysis_table)
        feasible[j] = cal_i[j] <= alpha_value

    eligible = np.flatnonzero(feasible)
    selected: int | None = None
    selected_design: Rbop2BinaryDesign | None = None
    if eligible.size:
        selected = int(min(eligible, key=lambda i: (-cal_power[i], cal_n[i], int(i))))
        selected_design = rbop2_binary_design(
            template.looks,
            prior=analysis,
            endpoint=endpoint,
            margin=margin,
            lower_cutoffs=candidates[selected, :, 0],
            upper_cutoffs=candidates[selected, :, 1],
            absolute_tolerance=absolute_tolerance,
        )
    return Rbop2BinaryCalibration(
        selected,
        selected_design,
        _freeze(calibration),
        _freeze(analysis),
        null,
        alt,
        _freeze(candidates),
        _freeze_bool(feasible),
        _freeze(cal_i),
        _freeze(cal_power),
        _freeze(cal_n),
        _freeze(ana_i),
        _freeze(ana_power),
        _freeze(ana_n),
        _freeze(errors),
        alpha_value,
    )
