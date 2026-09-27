"""Bounded posterior sampling for the CiBolus response/toxicity model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .cibolus import (
    CiBolusObservation,
    CiBolusPrediction,
    CiBolusPrior,
    _prediction_inputs,
    cibolus_loglikelihood,
    cibolus_parameter_names,
    cibolus_predict,
)
from .hierarchical_binomial import ChainSummary, summarize_chains
from .uaroet import _integer

_MAX_DRAWS = 100_000
_MAX_WARMUP = 100_000
_MAX_CHAINS = 8
_MAX_OBSERVATIONS = 200
_MAX_RETAINED_CELLS = 2_000_000
_MAX_EVALUATIONS = 2_000_000
_MAX_WORK = 50_000_000


@dataclass(frozen=True)
class CiBolusFit:
    """Retained model coordinates, grid probabilities, utility and diagnostics."""

    parameter_names: tuple[str, ...]
    prior: CiBolusPrior
    concentrations: FloatArray
    bolus_fractions: FloatArray
    endpoints: FloatArray
    utility: FloatArray
    log_parameters: FloatArray
    log_likelihood: FloatArray
    joint: FloatArray
    expected_utility: FloatArray
    response_at_one: FloatArray
    toxicity_at_one_response: FloatArray
    toxicity_at_one_failure: FloatArray
    marginal_toxicity: FloatArray
    parameter_summary: ChainSummary
    utility_summary: ChainSummary
    likelihood_evaluations: int
    work_units: int
    warmup: int
    direct_prior: bool


def _copy_prediction(
    output: CiBolusPrediction,
    joint: FloatArray,
    utility: FloatArray,
    response: FloatArray,
    tox_response: FloatArray,
    tox_failure: FloatArray,
    marginal_tox: FloatArray,
    chain: int,
    draw: int,
) -> None:
    joint[chain, draw] = output.joint
    utility[chain, draw] = output.expected_utility
    response[chain, draw] = output.response_at_one
    tox_response[chain, draw] = output.toxicity_at_one_response
    tox_failure[chain, draw] = output.toxicity_at_one_failure
    marginal_tox[chain, draw] = output.marginal_toxicity


def fit_cibolus(
    observations: tuple[CiBolusObservation, ...] | list[CiBolusObservation],
    prior: CiBolusPrior,
    concentrations: ArrayLike,
    bolus_fractions: ArrayLike,
    endpoints: ArrayLike,
    *,
    utility: ArrayLike,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    rng: np.random.Generator,
    max_evaluations: int = 200_000,
    max_work: int = 20_000_000,
) -> CiBolusFit:
    """Sample a posterior with serial elliptical slice updates.

    No-observation fits are exact independent prior draws. ``max_evaluations``
    counts likelihood calls; ``max_work`` separately bounds patient and
    concentration/bolus/category prediction operations.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    if not isinstance(prior, CiBolusPrior):
        raise ValueError("prior must be a CiBolusPrior")
    if not isinstance(observations, (tuple, list)) or len(observations) > _MAX_OBSERVATIONS:
        raise ValueError(f"observations must contain at most {_MAX_OBSERVATIONS} patients")
    if any(not isinstance(item, CiBolusObservation) for item in observations):
        raise ValueError("every observation must be a CiBolusObservation")
    draw_count = _integer(draws, "draws", 8, _MAX_DRAWS)
    warmup_count = _integer(warmup, "warmup", 0, _MAX_WARMUP)
    chain_count = _integer(chains, "chains", 2, _MAX_CHAINS)
    evaluation_limit = _integer(max_evaluations, "max_evaluations", 1, _MAX_EVALUATIONS)
    work_limit = _integer(max_work, "max_work", 1, _MAX_WORK)
    concentration_grid, bolus_grid, endpoint_grid, utility_grid = _prediction_inputs(
        concentrations, bolus_fractions, endpoints, utility
    )
    c_count, q_count, category_count = (
        concentration_grid.size,
        bolus_grid.size,
        endpoint_grid.size + 2,
    )
    joint_cells = c_count * q_count * category_count * 2
    retained_per_draw = 11 + 1 + joint_cells + 5 * c_count * q_count
    retained = chain_count * draw_count * retained_per_draw
    if retained > _MAX_RETAINED_CELLS:
        raise ValueError("retained CiBolus posterior arrays exceed two million cells")
    direct = len(observations) == 0
    minimum_evaluations = 0 if direct else chain_count * (1 + warmup_count + draw_count)
    prediction_work = chain_count * draw_count * c_count * q_count * category_count
    minimum_work = minimum_evaluations * len(observations) + prediction_work
    if minimum_evaluations > evaluation_limit:
        raise ValueError("max_evaluations is below the minimum likelihood-call count")
    if minimum_work > work_limit:
        raise ValueError("max_work is below the minimum patient/prediction work")

    log_draws = np.empty((chain_count, draw_count, 11))
    likelihood_draws = np.empty((chain_count, draw_count))
    joint_draws = np.empty((chain_count, draw_count, c_count, q_count, category_count, 2))
    utility_draws = np.empty((chain_count, draw_count, c_count, q_count))
    response_draws = np.empty_like(utility_draws)
    tox_response_draws = np.empty_like(utility_draws)
    tox_failure_draws = np.empty_like(utility_draws)
    marginal_tox_draws = np.empty_like(utility_draws)
    evaluations = 0
    work = 0

    def evaluate(theta: FloatArray) -> float:
        nonlocal evaluations, work
        if evaluations >= evaluation_limit:
            raise RuntimeError("CiBolus likelihood calls exceeded max_evaluations")
        if work + len(observations) > work_limit:
            raise RuntimeError("CiBolus patient work exceeded max_work")
        evaluations += 1
        work += len(observations)
        return cibolus_loglikelihood(observations, theta)

    def save(chain: int, draw: int, theta: FloatArray, likelihood: float) -> None:
        nonlocal work
        units = c_count * q_count * category_count
        if work + units > work_limit:
            raise RuntimeError("CiBolus prediction work exceeded max_work")
        work += units
        prediction = cibolus_predict(
            theta,
            concentration_grid,
            bolus_grid,
            endpoint_grid,
            utility=utility_grid,
        )
        log_draws[chain, draw] = theta
        likelihood_draws[chain, draw] = likelihood
        _copy_prediction(
            prediction,
            joint_draws,
            utility_draws,
            response_draws,
            tox_response_draws,
            tox_failure_draws,
            marginal_tox_draws,
            chain,
            draw,
        )

    for chain in range(chain_count):
        if direct:
            states = rng.normal(prior.mean, prior.sd, size=(draw_count, 11))
            if not np.all(np.isfinite(states)):
                raise ArithmeticError("normal prior log-parameter draw is not representable")
            for draw, state in enumerate(states):
                save(chain, draw, state, 0.0)
            continue

        for _ in range(1000):
            state = rng.normal(prior.mean, prior.sd)
            likelihood = evaluate(state)
            if np.isfinite(likelihood):
                break
        else:
            raise ArithmeticError("could not find finite-likelihood CiBolus prior initialization")

        for iteration in range(warmup_count + draw_count):
            centered = state - prior.mean
            direction = rng.normal(size=11) * prior.sd
            height = likelihood + np.log1p(-rng.random())
            angle = float(rng.uniform(0, 2 * np.pi))
            lower, upper = angle - 2 * np.pi, angle
            for _ in range(1000):
                proposal = prior.mean + centered * np.cos(angle) + direction * np.sin(angle)
                proposal_likelihood = evaluate(proposal)
                if proposal_likelihood >= height:
                    state, likelihood = proposal, proposal_likelihood
                    break
                if angle < 0:
                    lower = angle
                else:
                    upper = angle
                angle = float(rng.uniform(lower, upper))
            else:
                raise ArithmeticError(f"CiBolus elliptical slice failed in chain {chain}")
            if iteration >= warmup_count:
                save(chain, iteration - warmup_count, state, likelihood)

    return CiBolusFit(
        cibolus_parameter_names(),
        prior,
        _freeze(concentration_grid),
        _freeze(bolus_grid),
        _freeze(endpoint_grid),
        _freeze(utility_grid),
        _freeze(log_draws),
        _freeze(likelihood_draws),
        _freeze(joint_draws),
        _freeze(utility_draws),
        _freeze(response_draws),
        _freeze(tox_response_draws),
        _freeze(tox_failure_draws),
        _freeze(marginal_tox_draws),
        summarize_chains(log_draws),
        summarize_chains(utility_draws),
        evaluations,
        work,
        0 if direct else warmup_count,
        direct,
    )
