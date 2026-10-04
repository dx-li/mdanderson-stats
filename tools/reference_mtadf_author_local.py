"""Independent transformed-Cauchy quadrature for MTADF local windows.

This computes posterior summaries for the cached author's bivariate local
logistic likelihood. It does not run the author's Metropolis sampler.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec
from scipy.special import expit, log_expit

CASES = (
    # A zero-count neighbor remains at its full-grid x coordinate but adds no
    # Bernoulli rows to the likelihood.
    {
        "name": "lower_window_zero_upper_neighbor",
        "doses": 5,
        "indices": [1, 2],
        "subjects": [12, 0],
        "responses": [8, 0],
    },
    {
        "name": "upper_window_zero_lower_neighbor",
        "doses": 5,
        "indices": [4, 5],
        "subjects": [0, 11],
        "responses": [0, 3],
    },
    {
        "name": "interior_window_both_observed",
        "doses": 5,
        "indices": [2, 3],
        "subjects": [12, 9],
        "responses": [10, 4],
    },
)

GATE_CASES = (
    {
        "name": "lower_at_ce1_stays",
        "position": "lower",
        "dose": 1,
        "p_backward_nonpositive": "",
        "p_forward_positive": 0.3,
        "admissible": 5,
    },
    {
        "name": "lower_above_ce1_moves_up",
        "position": "lower",
        "dose": 1,
        "p_backward_nonpositive": "",
        "p_forward_positive": 0.300001,
        "admissible": 5,
    },
    {
        "name": "top_at_1_minus_ce1_stays",
        "position": "top",
        "dose": 5,
        "p_backward_nonpositive": 0.7,
        "p_forward_positive": "",
        "admissible": 5,
    },
    {
        "name": "top_above_1_minus_ce1_moves_down",
        "position": "top",
        "dose": 5,
        "p_backward_nonpositive": 0.700001,
        "p_forward_positive": "",
        "admissible": 5,
    },
    {
        "name": "untried_at_1_minus_ce2_moves_up",
        "position": "interior_untried",
        "dose": 3,
        "p_backward_nonpositive": 0.6,
        "p_forward_positive": "",
        "admissible": 5,
    },
    {
        "name": "untried_above_1_minus_ce2_stays",
        "position": "interior_untried",
        "dose": 3,
        "p_backward_nonpositive": 0.600001,
        "p_forward_positive": "",
        "admissible": 5,
    },
    {
        "name": "tried_both_strict_gates_move_up",
        "position": "interior_tried",
        "dose": 3,
        "p_backward_nonpositive": 0.6,
        "p_forward_positive": 0.300001,
        "admissible": 5,
    },
    {
        "name": "tried_backward_above_1_minus_ce1_moves_down",
        "position": "interior_tried",
        "dose": 3,
        "p_backward_nonpositive": 0.700001,
        "p_forward_positive": 0.9,
        "admissible": 5,
    },
    {
        "name": "tried_forward_at_ce1_does_not_move_up",
        "position": "interior_tried",
        "dose": 3,
        "p_backward_nonpositive": 0.6,
        "p_forward_positive": 0.3,
        "admissible": 5,
    },
    {
        "name": "admissibility_one_shortcuts_fit_and_caps_at_one",
        "position": "safety_cap",
        "dose": 4,
        "p_backward_nonpositive": "",
        "p_forward_positive": "",
        "admissible": 1,
    },
)


def author_action(
    case: dict[str, object], ce1: float = 0.3, ce2: float = 0.4
) -> tuple[str, int, int]:
    """Evaluate the recovered local strict/inclusive gates and cap."""
    position = str(case["position"])
    dose = int(case["dose"])
    admissible = int(case["admissible"])
    if admissible == 1:
        return "safety_only_no_fit", 1, 1
    desired = dose
    backward = case["p_backward_nonpositive"]
    forward = case["p_forward_positive"]
    if position == "lower":
        if float(forward) > ce1:
            desired += 1
    elif position == "top":
        if float(backward) > 1 - ce1:
            desired -= 1
    elif position == "interior_untried":
        if float(backward) > 1 - ce1:
            desired -= 1
        elif float(backward) <= 1 - ce2:
            desired += 1
    elif position == "interior_tried":
        if float(backward) <= 1 - ce2 and float(forward) > ce1:
            desired += 1
        elif float(backward) > 1 - ce1:
            desired -= 1
    else:
        raise ValueError(f"unknown gate position {position!r}")
    return "posterior_gate", desired, min(desired, admissible)


def local_reference(case: dict[str, object], order: int) -> tuple[float, np.ndarray]:
    """Return P(beta>0) and posterior mean probabilities at all grid doses."""
    dose_count = int(case["doses"])
    indices = np.asarray(case["indices"], dtype=int) - 1
    n = np.asarray(case["subjects"], dtype=float)
    y = np.asarray(case["responses"], dtype=float)
    if (
        indices.shape != n.shape
        or n.shape != y.shape
        or np.any(n < 0)
        or np.any(y < 0)
        or np.any(y > n)
    ):
        raise ValueError("invalid local-window counts")
    dose_grid = np.arange(1, dose_count + 1, dtype=float)
    x_grid = (dose_grid - dose_grid.mean()) / (2 * dose_grid.std(ddof=1))
    # Keep the author's global-grid coding. Zero-count endpoints contribute
    # no rows, exactly as rep(xs, n) does in R.
    x = np.repeat(x_grid[indices], n.astype(int))
    response = np.concatenate(
        [np.r_[np.ones(int(yi)), np.zeros(int(ni - yi))] for ni, yi in zip(n, y, strict=True)]
    )
    nodes, weights = leggauss(order)
    # Cauchy CDF coordinates make each independent prior exactly uniform.
    # Integrate the two beta signs separately so the positive-slope indicator
    # is not a discontinuity inside a quadrature interval.
    alpha = 10.0 * np.tan(np.pi * nodes / 2)
    alpha_weights = weights / 2
    unit_nodes = (nodes + 1) / 2
    half_weights = weights / 4
    beta_negative = 2.5 * np.tan(np.pi * (unit_nodes / 2 - 0.5))
    beta_positive = 2.5 * np.tan(np.pi * ((0.5 + unit_nodes / 2) - 0.5))
    denominator = 0.0
    positive_slope_mass = 0.0
    probability_mass = np.zeros(dose_count)
    # Shift by the independently saturated likelihood at each observed dose.
    shift = 0.0
    for ni, yi in zip(n, y, strict=True):
        if ni:
            if yi:
                shift += yi * np.log(yi / ni)
            if ni - yi:
                shift += (ni - yi) * np.log1p(-yi / ni)
    for beta, is_positive in ((beta_negative, False), (beta_positive, True)):
        for ai, aw in zip(alpha, alpha_weights, strict=True):
            eta = ai + beta[:, None] * x[None, :]
            if response.size:
                log_likelihood = np.sum(
                    response[None, :] * -np.logaddexp(0.0, -eta)
                    + (1.0 - response[None, :]) * -np.logaddexp(0.0, eta),
                    axis=1,
                )
            else:
                log_likelihood = np.zeros(order)
            likelihood = np.exp(log_likelihood - shift)
            joint_weight = aw * half_weights * likelihood
            mass = joint_weight.sum()
            denominator += mass
            if is_positive:
                positive_slope_mass += mass
            probability_mass += (joint_weight[:, None] * expit(ai + beta[:, None] * x_grid)).sum(
                axis=0
            )
    if not denominator or not np.isfinite(denominator):
        raise ArithmeticError("quadrature failed to resolve local posterior mass")
    return positive_slope_mass / denominator, probability_mass / denominator


def one_dose_reference(
    case: dict[str, object], tolerance: float
) -> tuple[float, np.ndarray, float]:
    """Adaptive nested integral for a window with only one observed dose.

    Reparameterize from intercept alpha to the identified linear predictor at
    the observed dose; this removes the long alpha/slope ridge from tensor
    Cauchy-coordinate quadrature.
    """
    dose_count = int(case["doses"])
    indices = np.asarray(case["indices"], dtype=int) - 1
    n = np.asarray(case["subjects"], dtype=float)
    y = np.asarray(case["responses"], dtype=float)
    observed = np.flatnonzero(n > 0)
    if observed.size != 1:
        raise ValueError("adaptive one-dose reference requires exactly one observed window dose")
    observed_dose = indices[observed[0]]
    subjects = int(n[observed[0]])
    responses = int(y[observed[0]])
    dose_grid = np.arange(1, dose_count + 1, dtype=float)
    x_grid = (dose_grid - dose_grid.mean()) / (2 * dose_grid.std(ddof=1))
    x_observed = x_grid[observed_dose]
    saturated = (responses * np.log(responses / subjects) if responses else 0.0) + (
        (subjects - responses) * np.log1p(-responses / subjects) if responses < subjects else 0.0
    )

    def integrate_beta(lower: float, upper: float) -> tuple[np.ndarray, float]:
        max_inner_error = 0.0

        def at_beta(beta: float) -> np.ndarray:
            nonlocal max_inner_error

            def at_eta(eta: float) -> np.ndarray:
                log_likelihood = (
                    responses * log_expit(eta)
                    + (subjects - responses) * log_expit(-eta)
                    - saturated
                )
                alpha = eta - beta * x_observed
                alpha_density = 10.0 / (np.pi * (100.0 + alpha * alpha))
                likelihood = np.exp(log_likelihood)
                probs = expit(eta + beta * (x_grid - x_observed))
                return alpha_density * likelihood * np.r_[1.0, probs]

            inner, error = quad_vec(
                at_eta, -np.inf, np.inf, epsabs=tolerance / 10, epsrel=tolerance / 10, limit=300
            )
            max_inner_error = max(max_inner_error, float(error))
            beta_density = 2.5 / (np.pi * (6.25 + beta * beta))
            return beta_density * inner

        integral, error = quad_vec(
            at_beta, lower, upper, epsabs=tolerance, epsrel=tolerance, limit=300
        )
        # The beta Cauchy density integrates to at most one on either half;
        # max-inner-error is therefore a conservative propagated allowance.
        return integral, float(error) + max_inner_error

    negative, negative_error = integrate_beta(-np.inf, 0.0)
    positive, positive_error = integrate_beta(0.0, np.inf)
    total = negative[0] + positive[0]
    if total <= 0 or not np.isfinite(total):
        raise ArithmeticError("adaptive quadrature failed to resolve local posterior mass")
    return (
        positive[0] / total,
        (negative[1:] + positive[1:]) / total,
        (negative_error + positive_error) / total,
    )


def write_fixture() -> None:
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/mtadf-author-local-reference.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "case",
                "dose_count",
                "window_dose_indices",
                "window_subjects",
                "window_responses",
                "quadrature_order",
                "refinement_max_abs_error",
                "quadrature_error_estimate",
                "probability_positive_slope",
                "posterior_mean_probabilities",
            ],
        )
        writer.writeheader()
        for case in CASES:
            n = np.asarray(case["subjects"])
            if np.count_nonzero(n) == 1:
                coarse_slope, coarse_means, _ = one_dose_reference(case, 2e-8)
                slope, means, error_estimate = one_dose_reference(case, 5e-10)
                order = "adaptive"
            else:
                coarse_slope, coarse_means = local_reference(case, 192)
                slope, means = local_reference(case, 256)
                order = 256
                error_estimate = float("nan")
            refinement = max(abs(slope - coarse_slope), float(np.max(np.abs(means - coarse_means))))
            if refinement > 2e-6 or (order == "adaptive" and error_estimate > 2e-8):
                raise ArithmeticError(
                    f"quadrature did not converge for {case['name']}: {refinement:g}"
                )
            writer.writerow(
                {
                    "case": case["name"],
                    "dose_count": case["doses"],
                    "window_dose_indices": json.dumps(case["indices"], separators=(",", ":")),
                    "window_subjects": json.dumps(case["subjects"], separators=(",", ":")),
                    "window_responses": json.dumps(case["responses"], separators=(",", ":")),
                    "quadrature_order": order,
                    "refinement_max_abs_error": f"{refinement:.17g}",
                    "quadrature_error_estimate": ""
                    if not np.isfinite(error_estimate)
                    else f"{error_estimate:.17g}",
                    "probability_positive_slope": f"{slope:.17g}",
                    "posterior_mean_probabilities": json.dumps(
                        means.tolist(), separators=(",", ":")
                    ),
                }
            )
    print(f"Wrote {output}")

    gate_output = output.with_name("mtadf-author-local-gates.csv")
    with gate_output.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case",
            "position",
            "dose",
            "p_backward_nonpositive",
            "p_forward_positive",
            "ce1",
            "ce2",
            "admissible_count",
            "action",
            "desired_dose",
            "capped_dose",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in GATE_CASES:
            action, desired, capped = author_action(case)
            writer.writerow(
                {
                    "case": case["name"],
                    "position": case["position"],
                    "dose": case["dose"],
                    "p_backward_nonpositive": case["p_backward_nonpositive"],
                    "p_forward_positive": case["p_forward_positive"],
                    "ce1": "0.3",
                    "ce2": "0.4",
                    "admissible_count": case["admissible"],
                    "action": action,
                    "desired_dose": desired,
                    "capped_dose": capped,
                }
            )
    print(f"Wrote {gate_output}")


if __name__ == "__main__":
    write_fixture()
