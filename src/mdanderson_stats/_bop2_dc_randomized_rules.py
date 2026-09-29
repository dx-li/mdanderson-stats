"""Shared private decision rules for randomized BOP2-DC endpoints."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, erfinv


def randomized_obf_cutoffs(
    *,
    look: int,
    max_subjects: int,
    lambda_lrv: float,
    lambda_cmv: float,
) -> tuple[float, float]:
    """Return source O'Brien–Fleming cutoffs for one interim look."""
    fraction = look / max_subjects
    return (
        float(erf(erfinv(lambda_lrv) / np.sqrt(fraction))),
        float(erf(erfinv(lambda_cmv) / np.sqrt(fraction))),
    )


def _randomized_dual_decisions_from_tails(
    total_n: np.ndarray,
    posterior_lrv: np.ndarray,
    posterior_cmv: np.ndarray,
    *,
    max_subjects: int,
    looks: NDArray[np.int64],
    lambda_lrv: float,
    lambda_cmv: float,
    gamma_lrv: float,
    gamma_cmv: float,
    graduate_at_interim: bool,
) -> NDArray[np.str_]:
    decision = np.full(total_n.shape, "continue", dtype="U16")
    for raw_look in looks[:-1]:
        look = int(raw_look)
        at_look = total_n == look
        no_go = (
            at_look
            & (posterior_lrv < lambda_lrv * (look / max_subjects) ** gamma_lrv)
            & (posterior_cmv < lambda_cmv * (look / max_subjects) ** gamma_cmv)
        )
        decision[no_go] = "stop_no_go"
        if graduate_at_interim:
            grad_lrv, grad_cmv = randomized_obf_cutoffs(
                look=look,
                max_subjects=max_subjects,
                lambda_lrv=lambda_lrv,
                lambda_cmv=lambda_cmv,
            )
            graduate = at_look & (posterior_lrv > grad_lrv) & (posterior_cmv > grad_cmv)
            if np.any(no_go & graduate):
                raise ArithmeticError("interim graduation and no-go rules overlap")
            decision[graduate] = "graduate"
    final = total_n == max_subjects
    go = final & (posterior_lrv > lambda_lrv) & (posterior_cmv > lambda_cmv)
    no_go = final & (posterior_lrv < lambda_lrv) & (posterior_cmv < lambda_cmv)
    decision[go] = "final_go"
    decision[no_go] = "final_no_go"
    decision[final & ~(go | no_go)] = "final_consider"
    decision.flags.writeable = False
    return decision


def randomized_dual_decisions(
    total_n: np.ndarray,
    posterior_lrv: np.ndarray,
    posterior_cmv: np.ndarray,
    error_lrv: np.ndarray,
    error_cmv: np.ndarray,
    *,
    max_subjects: int,
    looks: NDArray[np.int64],
    lambda_lrv: float,
    lambda_cmv: float,
    gamma_lrv: float,
    gamma_cmv: float,
    graduate_at_interim: bool,
) -> NDArray[np.str_]:
    """Classify when reported quadrature errors cannot change the action.

    Error values are estimates from beta-comparison integration, not rigorous
    bounds. The four corners suffice because all monitoring regions are
    intersections of coordinate-wise strict cutoff half-spaces.
    """
    low_lrv = np.maximum(0.0, posterior_lrv - error_lrv)
    high_lrv = np.minimum(1.0, posterior_lrv + error_lrv)
    low_cmv = np.maximum(0.0, posterior_cmv - error_cmv)
    high_cmv = np.minimum(1.0, posterior_cmv + error_cmv)
    choices = (
        _randomized_dual_decisions_from_tails(
            total_n,
            low_lrv,
            low_cmv,
            max_subjects=max_subjects,
            looks=looks,
            lambda_lrv=lambda_lrv,
            lambda_cmv=lambda_cmv,
            gamma_lrv=gamma_lrv,
            gamma_cmv=gamma_cmv,
            graduate_at_interim=graduate_at_interim,
        ),
        _randomized_dual_decisions_from_tails(
            total_n,
            low_lrv,
            high_cmv,
            max_subjects=max_subjects,
            looks=looks,
            lambda_lrv=lambda_lrv,
            lambda_cmv=lambda_cmv,
            gamma_lrv=gamma_lrv,
            gamma_cmv=gamma_cmv,
            graduate_at_interim=graduate_at_interim,
        ),
        _randomized_dual_decisions_from_tails(
            total_n,
            high_lrv,
            low_cmv,
            max_subjects=max_subjects,
            looks=looks,
            lambda_lrv=lambda_lrv,
            lambda_cmv=lambda_cmv,
            gamma_lrv=gamma_lrv,
            gamma_cmv=gamma_cmv,
            graduate_at_interim=graduate_at_interim,
        ),
        _randomized_dual_decisions_from_tails(
            total_n,
            high_lrv,
            high_cmv,
            max_subjects=max_subjects,
            looks=looks,
            lambda_lrv=lambda_lrv,
            lambda_cmv=lambda_cmv,
            gamma_lrv=gamma_lrv,
            gamma_cmv=gamma_cmv,
            graduate_at_interim=graduate_at_interim,
        ),
    )
    reference = choices[0]
    if any(np.any(choice != reference) for choice in choices[1:]):
        raise ArithmeticError(
            "reported beta-comparison quadrature error could change a strict trial decision"
        )
    return reference
