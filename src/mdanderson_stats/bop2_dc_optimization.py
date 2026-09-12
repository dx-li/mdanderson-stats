"""Finite-grid calibration for binary BOP2-DC designs."""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bop2_dc import BOP2DCDesign, BOP2DCOperatingCharacteristics, bop2_dc_design


class BOP2DCInfeasibleError(ValueError):
    """No candidate on the declared finite grid satisfies the constraints."""


@dataclass(frozen=True)
class BOP2DCOptimization:
    design: BOP2DCDesign
    futile_oc: BOP2DCOperatingCharacteristics
    effective_oc: BOP2DCOperatingCharacteristics
    false_go_rate: float
    false_no_go_rate: float
    correct_go_rate: float
    false_consider_rate: float
    futile_expected_sample_size: float
    objective: str
    candidate_count: int
    grid: dict[str, FloatArray]


def _grid(value: ArrayLike | None, default: np.ndarray, name: str) -> np.ndarray:
    result = default if value is None else finite(value, name)
    result = np.asarray(result, dtype=float)
    if result.ndim != 1 or not result.size or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite 1D grid")
    result = np.array(result, dtype=float, copy=True)
    result.flags.writeable = False
    return result


def optimize_bop2_dc(
    max_subjects: int,
    lrv: float,
    cmv: float,
    theta_futile: float,
    theta_effective: float,
    *,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    lambda_lrv_grid: ArrayLike | None = None,
    lambda_cmv_grid: ArrayLike | None = None,
    gamma_lrv_grid: ArrayLike | None = None,
    gamma_cmv_grid: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2DCOptimization:
    """Optimize CGR or futile expected sample size on a finite candidate grid.

    Constraints use exact recursion: FGR is go at ``theta_futile``, FNGR is
    total no-go at ``theta_effective``, and FCR is the maximum consider rate at
    the two scenarios. All constraints are enforced with ``<=``. Results are a
    finite-grid optimum and make no claim about continuous optimization.
    """
    futile, effective = (
        scalar(theta_futile, "theta_futile"),
        scalar(theta_effective, "theta_effective"),
    )
    if not 0 <= futile < effective <= 1:
        raise ValueError("require 0 <= theta_futile < theta_effective <= 1")
    fg, fn = (
        scalar(false_go_limit, "false_go_limit"),
        scalar(false_no_go_limit, "false_no_go_limit"),
    )
    fc = (
        None
        if false_consider_limit is None
        else scalar(false_consider_limit, "false_consider_limit")
    )
    if not 0 <= fg <= 1 or not 0 <= fn <= 1 or (fc is not None and not 0 <= fc <= 1):
        raise ValueError("error limits must lie in (0,1]")
    if objective not in ("cgr", "ess_futile"):
        raise ValueError("objective must be 'cgr' or 'ess_futile'")
    grids = {
        "lambda_lrv": _grid(lambda_lrv_grid, np.array([0.5, 0.8, 0.9]), "lambda_lrv_grid"),
        "lambda_cmv": _grid(lambda_cmv_grid, np.array([0.1, 0.3, 0.5]), "lambda_cmv_grid"),
        "gamma_lrv": _grid(gamma_lrv_grid, np.array([0.0, 0.5, 1.0]), "gamma_lrv_grid"),
        "gamma_cmv": _grid(gamma_cmv_grid, np.array([0.0, 0.5, 1.0]), "gamma_cmv_grid"),
    }
    if np.any((grids["lambda_lrv"] <= 0) | (grids["lambda_lrv"] >= 1)):
        raise ValueError("lambda_lrv_grid values must lie in (0,1)")
    if np.any((grids["lambda_cmv"] <= 0) | (grids["lambda_cmv"] >= 1)):
        raise ValueError("lambda_cmv_grid values must lie in (0,1)")
    if np.any((grids["gamma_lrv"] < 0) | (grids["gamma_lrv"] > 1)):
        raise ValueError("gamma_lrv_grid values must lie in [0,1]")
    if np.any((grids["gamma_cmv"] < 0) | (grids["gamma_cmv"] > 1)):
        raise ValueError("gamma_cmv_grid values must lie in [0,1]")
    candidate_count = prod(v.size for v in grids.values())
    if candidate_count > 100_000:
        raise ValueError("candidate grid contains more than 100000 combinations")
    n_value = scalar(max_subjects, "max_subjects")
    if n_value != int(n_value):
        raise ValueError("max_subjects must be an integer")
    if candidate_count * 2 * (int(n_value) + 1) ** 2 > 5_000_000:
        raise ValueError("candidate grid exceeds the exact-recursion work budget")
    bop2_dc_design(
        max_subjects,
        lrv,
        cmv,
        lambda_lrv=float(grids["lambda_lrv"][0]),
        lambda_cmv=float(grids["lambda_cmv"][0]),
        gamma_lrv=float(grids["gamma_lrv"][0]),
        gamma_cmv=float(grids["gamma_cmv"][0]),
        prior=prior,
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    best: tuple[tuple[float, ...], BOP2DCOptimization] | None = None
    for ll in grids["lambda_lrv"]:
        for lc in grids["lambda_cmv"]:
            for gl in grids["gamma_lrv"]:
                for gc in grids["gamma_cmv"]:
                    design = bop2_dc_design(
                        max_subjects,
                        lrv,
                        cmv,
                        lambda_lrv=float(ll),
                        lambda_cmv=float(lc),
                        gamma_lrv=float(gl),
                        gamma_cmv=float(gc),
                        prior=prior,
                        looks=looks,
                        min_subjects=min_subjects,
                        cohort_size=cohort_size,
                    )
                    futile_oc = design.operating_characteristics([futile])
                    effective_oc = design.operating_characteristics([effective])
                    fgr = float(futile_oc.final_go[0])
                    fnr = float(effective_oc.no_go_probability[0])
                    cgr = float(effective_oc.final_go[0])
                    fcr = float(max(futile_oc.final_consider[0], effective_oc.final_consider[0]))
                    ess = float(futile_oc.expected_sample_size[0])
                    if fgr > fg or fnr > fn or (fc is not None and fcr > fc):
                        continue
                    metrics = BOP2DCOptimization(
                        design,
                        futile_oc,
                        effective_oc,
                        fgr,
                        fnr,
                        cgr,
                        fcr,
                        ess,
                        objective,
                        candidate_count,
                        {k: _grid(v, v, k) for k, v in grids.items()},
                    )
                    key = (
                        (-cgr, ess, float(ll), float(lc), float(gl), float(gc))
                        if objective == "cgr"
                        else (ess, -cgr, float(ll), float(lc), float(gl), float(gc))
                    )
                    if best is None or key < best[0]:
                        best = (key, metrics)
    if best is None:
        raise BOP2DCInfeasibleError("no finite-grid BOP2-DC candidate satisfies the constraints")
    result = best[1]
    return BOP2DCOptimization(
        result.design,
        result.futile_oc,
        result.effective_oc,
        result.false_go_rate,
        result.false_no_go_rate,
        result.correct_go_rate,
        result.false_consider_rate,
        result.futile_expected_sample_size,
        result.objective,
        candidate_count,
        result.grid,
    )
