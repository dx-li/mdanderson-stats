"""Calendar-based U2OET trial simulation with auditable pending outcomes."""

import json
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import summarize_chains
from .u2oet import _real, u2oet_standardize
from .u2oet_decision import U2OETCriteria, U2OETPosterior, _integer, u2oet_posterior
from .u2oet_fit import fit_u2oet, u2oet_parameter_names
from .u2oet_patients import U2OETPatients, u2oet_next_patient, u2oet_patients
from .u2oet_scenario import U2OETScenario


@dataclass(frozen=True)
class U2OETTrialDecision:
    time: float
    complete_patients: int
    toxicity_only_patients: int
    ignored_patients: int
    probabilities: FloatArray
    posterior: U2OETPosterior
    max_split_rhat: float
    continuing_cohort: bool
    reason: str


@dataclass(frozen=True)
class U2OETTrial:
    """True records and outcome times allow reconstruction of every interim snapshot."""

    patients: U2OETPatients
    arrival_times: FloatArray
    outcome_times: FloatArray
    decisions: tuple[U2OETTrialDecision, ...]
    final_posterior: U2OETPosterior
    final_max_split_rhat: float
    selected: tuple[int, int] | None
    stopped_early: bool
    stop_time: float
    analysis_time: float
    posterior_fits: int
    data_seed: int
    posterior_seed: int
    final_scope: str
    design_json: str = ""


def _window(value: ArrayLike, name: str) -> FloatArray:
    x = _real(value, name)
    if x.shape != (2,) or x[0] < 0 or x[1] < x[0]:
        raise ValueError(f"{name} must be a nonnegative [minimum,maximum] pair")
    return x


def simulate_u2oet_trial(
    doses1: ArrayLike,
    doses2: ArrayLike,
    scenario: U2OETScenario,
    utility: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    initial: tuple[int, int],
    criteria: U2OETCriteria = U2OETCriteria(),
    max_patients: int = 60,
    cohort_size: int = 3,
    surplus: int | None = None,
    top: int | None = 2,
    greedy: bool = False,
    efficacy_window: ArrayLike = (42.0, 42.0),
    toxicity_window: ArrayLike = (42.0, 42.0),
    mean_interarrival: float = 20.0,
    final_scope: str = "acceptable",
    model: str = "pds",
    centering: str = "log",
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    coordinate_updates: bool = True,
    rng: np.random.Generator,
) -> U2OETTrial:
    """Simulate arrivals, outcomes, cohort decisions and complete final follow-up.

    Initial enrollment is at time zero. Outcomes at time <= arrival are available.
    No further enrollment follows an unacceptable-dose stop. Final selection
    maximizes utility among acceptable pairs, optionally restricted to tried
    pairs; neither final-selection convention claims native executable parity.
    """
    if not isinstance(rng, np.random.Generator) or not isinstance(scenario, U2OETScenario):
        raise ValueError("require an explicit Generator and U2OETScenario")
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    u2oet_standardize(d1)
    u2oet_standardize(d2)
    shape = scenario.joint.shape
    if shape[:2] != (d1.size, d2.size):
        raise ValueError("dose arrays and scenario grid must match")
    u = _real(utility, "utility")
    if u.shape != shape[2:] or np.any(u < 0):
        raise ValueError("utility must be a nonnegative efficacy-by-toxicity matrix")
    if (
        not isinstance(criteria, U2OETCriteria)
        or criteria.efficacy_level >= shape[2]
        or criteria.toxicity_level >= shape[3]
    ):
        raise ValueError("criteria must match outcome categories")
    names = u2oet_parameter_names(shape[2], shape[3], model=model)
    mu, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if mu.shape != (len(names) - 1,) or sd.shape != mu.shape or np.any(sd <= 0):
        raise ValueError("prior arrays must match named coordinates and SDs must be positive")
    maximum = _integer(max_patients, "max_patients", 1, 2500)
    cohort = _integer(cohort_size, "cohort_size", 1, 2500)
    surplus = cohort if surplus is None else _integer(surplus, "surplus", 0, 2500)
    if top is not None:
        top = _integer(top, "top", 1, d1.size * d2.size)
    if len(initial) != 2:
        raise ValueError("initial must contain two dose indices")
    first = (
        _integer(initial[0], "initial", 0, shape[0] - 1),
        _integer(initial[1], "initial", 0, shape[1] - 1),
    )
    if final_scope not in ("acceptable", "tried"):
        raise ValueError("final_scope must be acceptable or tried")
    if centering not in ("log", "linear") or (model == "cmi" and centering != "log"):
        raise ValueError("invalid model centering")
    if not isinstance(greedy, (bool, np.bool_)) or not isinstance(
        coordinate_updates, (bool, np.bool_)
    ):
        raise ValueError("greedy and coordinate_updates must be booleans")
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    if draws * chains * scenario.joint.size > 20_000_000:
        raise ValueError("retained joint draws per posterior exceed 20 million cells")
    ew, tw = (
        _window(efficacy_window, "efficacy_window"),
        _window(toxicity_window, "toxicity_window"),
    )
    gap = _real(mean_interarrival, "mean_interarrival")
    if gap.ndim or gap <= 0:
        raise ValueError("mean_interarrival must be a positive scalar")
    design_json = json.dumps(
        {
            "format_version": 1,
            "doses1": d1.tolist(),
            "doses2": d2.tolist(),
            "scenario_joint": scenario.joint.tolist(),
            "utility": u.tolist(),
            "criteria": asdict(criteria),
            "prior_mean": mu.tolist(),
            "prior_sd": sd.tolist(),
            "initial": first,
            "max_patients": maximum,
            "cohort_size": cohort,
            "surplus": surplus,
            "top": top,
            "greedy": bool(greedy),
            "efficacy_window": ew.tolist(),
            "toxicity_window": tw.tolist(),
            "mean_interarrival": float(gap),
            "final_scope": final_scope,
            "model": model,
            "centering": centering,
            "draws": draws,
            "warmup": warmup,
            "chains": chains,
            "coordinate_updates": bool(coordinate_updates),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    # Posterior workload must not change the data-generating RNG stream.
    seeds = rng.integers(0, 2**63, size=2, dtype=np.int64)
    data_rng, fit_rng = (np.random.default_rng(int(s)) for s in seeds)
    records: list[list[int]] = []
    arrivals: list[float] = []
    times: list[tuple[float, float]] = []
    history: list[U2OETTrialDecision] = []
    cached_counts: tuple[FloatArray, FloatArray] | None = None
    posterior: U2OETPosterior | None = None
    rhat = np.nan
    fits = 0

    def snapshot(now: float) -> U2OETPatients:
        rows = np.array(records, dtype=float).reshape(-1, 5)
        if len(rows):
            available = np.array(times) <= now
            rows[:, 3:] = np.where(available, rows[:, 3:], -1)
        return u2oet_patients(
            rows, dose_counts=(d1.size, d2.size), efficacy_levels=shape[2], toxicity_levels=shape[3]
        )

    def update(data: U2OETPatients) -> tuple[U2OETPosterior, float]:
        nonlocal posterior, cached_counts, fits, rhat
        if cached_counts is None or not (
            np.array_equal(data.complete, cached_counts[0])
            and np.array_equal(data.toxicity_only, cached_counts[1])
        ):
            fit = fit_u2oet(
                d1,
                d2,
                data.complete,
                toxicity_only=data.toxicity_only,
                prior_mean=mu,
                prior_sd=sd,
                model=model,
                centering=centering,
                draws=draws,
                warmup=warmup,
                chains=chains,
                coordinate_updates=coordinate_updates,
                rng=fit_rng,
            )
            posterior = u2oet_posterior(fit.joint.reshape(-1, *shape), u, criteria=criteria)
            diagnostics = summarize_chains(fit.parameters)
            rhat = float(np.max(diagnostics.split_rhat))
            cached_counts = (data.complete, data.toxicity_only)
            fits += 1
        assert posterior is not None
        return posterior, rhat

    now = 0.0
    early = False
    for patient in range(maximum):
        if patient == 0:
            pair = first
        else:
            next_time = now + float(data_rng.exponential(float(gap)))
            if not np.isfinite(next_time) or next_time <= now:
                raise ArithmeticError("arrival times are not representable as strictly increasing")
            now = next_time
            data = snapshot(now)
            current, diagnostic = update(data)
            decision = u2oet_next_patient(
                current,
                data,
                cohort_size=cohort,
                max_patients=maximum,
                surplus=surplus,
                top=top,
                initial=first,
                greedy=greedy,
            )
            history.append(
                U2OETTrialDecision(
                    now,
                    int(data.complete.sum()),
                    int(data.toxicity_only.sum()),
                    data.ignored_outcomes,
                    decision.probabilities,
                    current,
                    diagnostic,
                    decision.continuing_cohort,
                    decision.reason,
                )
            )
            if not np.any(decision.probabilities):
                early = True
                break
            chosen = int(data_rng.choice(d1.size * d2.size, p=decision.probabilities.ravel()))
            pair = (chosen // d2.size, chosen % d2.size)
        outcome = int(data_rng.choice(shape[2] * shape[3], p=scenario.joint[pair].ravel()))
        records.append(
            [patient + 1, pair[0] + 1, pair[1] + 1, outcome // shape[3], outcome % shape[3]]
        )
        arrivals.append(now)
        observed = (now + float(data_rng.uniform(*ew)), now + float(data_rng.uniform(*tw)))
        if not np.all(np.isfinite(observed)):
            raise ArithmeticError("outcome observation times exceed floating-point range")
        times.append(observed)
    analysis_time = max(now, float(np.max(times)))
    final_data = snapshot(analysis_time)
    final, final_rhat = update(final_data)
    acceptable = final.acceptable.copy()
    if final_scope == "tried":
        acceptable &= final_data.treated > 0
    selected = None
    if not early and np.any(acceptable):
        best = int(np.argmax(np.where(acceptable, final.mean_utility, -np.inf)))
        selected = (best // d2.size, best % d2.size)
    return U2OETTrial(
        final_data,
        _freeze(arrivals),
        _freeze(times),
        tuple(history),
        final,
        final_rhat,
        selected,
        early,
        now,
        analysis_time,
        fits,
        int(seeds[0]),
        int(seeds[1]),
        final_scope,
        design_json,
    )
