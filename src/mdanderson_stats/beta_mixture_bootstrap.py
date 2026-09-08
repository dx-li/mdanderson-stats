"""MULTI SIMCVM parametric refitting and strict-tail simulation estimate."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, scalar
from .beta_mixture import BetaMixture
from .beta_mixture_fit import BetaMixtureFit, BetaMixtureFitError, fit_beta_mixture_em
from .beta_mixture_ml import fit_beta_mixture_ml
from .multiplicity import _pvalues


@dataclass(frozen=True)
class BetaMixtureBootstrap:
    statistic: float
    pvalue: float
    simulated_statistics: FloatArray
    attempts: int
    failures: tuple[str, ...]


def _controls(algorithm: str, tolerance: float, max_iterations: int) -> float:
    if algorithm not in ("ml", "em"):
        raise ValueError("algorithm must be 'ml' or 'em'")
    tolerance = scalar(tolerance, "tolerance")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must lie strictly between zero and one")
    _positive_integer(max_iterations, "max_iterations")
    return tolerance


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _fit(
    x: FloatArray,
    model: BetaMixture,
    algorithm: str,
    tolerance: float,
    max_iterations: int,
    legacy_endpoints: bool,
) -> BetaMixtureFit:
    fitter = fit_beta_mixture_ml if algorithm == "ml" else fit_beta_mixture_em
    return fitter(
        x,
        model,
        tolerance=tolerance,
        max_iterations=max_iterations,
        legacy_endpoints=legacy_endpoints,
    )


def _sample(model: BetaMixture, n: int, rng: np.random.Generator) -> FloatArray:
    if not model.weights.size:
        return rng.random(n)
    component = rng.choice(
        model.weights.size + 1, size=n, p=np.r_[model.null_weight, model.weights]
    )
    a, b = np.r_[1.0, model.a], np.r_[1.0, model.b]
    return rng.beta(a[component], b[component])


def beta_mixture_bootstrap(
    pvalues: ArrayLike,
    model: BetaMixture,
    *,
    algorithm: str = "ml",
    replicates: int = 100,
    max_attempts: int | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 10000,
    rng: np.random.Generator | int | None = None,
) -> BetaMixtureBootstrap:
    """Refit simulated samples from model, preserving SIMCVM's stopping/tail rules.

    The supplied model should be fitted to pvalues. Each replicate starts from
    it, holds component count fixed, and refits parameters. NumPy mixture draws
    replace the original inverse-CDF/RANF stream. Failed fits are retried up to
    twice replicates by default; failures are reported, never replaced by zeros.
    The p-value is mean(simulated statistic > observed), without a +1 correction.
    """
    x = _pvalues(pvalues)
    if x.ndim != 1:
        raise ValueError("Bootstrap accepts one p-value vector")
    tolerance = _controls(algorithm, tolerance, max_iterations)
    _positive_integer(replicates, "replicates")
    max_attempts = 2 * replicates if max_attempts is None else max_attempts
    _positive_integer(max_attempts, "max_attempts")
    if max_attempts < replicates:
        raise ValueError("max_attempts must be at least replicates")
    if model.weights.size > 10 or x.size <= 3 * model.weights.size + 1:
        raise ValueError("SIMCVM requires k<=10 and n>3*k+1")
    # Validate the optimizer's model domain before drawing or retrying anything.
    if algorithm == "ml" and np.any(
        (model.a < 1e-10) | (model.a > 1e9) | (model.b < 1e-10) | (model.b > 1e9)
    ):
        raise ValueError("ML beta shapes must lie in [1e-10,1e9]")
    generator = np.random.default_rng(rng)
    statistic = float(model.cramer_von_mises(x))
    statistics: list[float] = []
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        sample = _sample(model, x.size, generator)
        if np.any((sample <= 0) | (sample >= 1)):
            failures.append("Generated sample rounded to an endpoint")
            continue
        try:
            fit = _fit(sample, model, algorithm, tolerance, max_iterations, False)
        except BetaMixtureFitError as error:
            failures.append(str(error))
            continue
        statistics.append(fit.cramer_von_mises)
        if len(statistics) == replicates:
            values = np.array(statistics)
            values.setflags(write=False)
            return BetaMixtureBootstrap(
                statistic, float(np.mean(values > statistic)), values, attempt, tuple(failures)
            )
    raise BetaMixtureFitError(
        f"SIMCVM obtained {len(statistics)}/{replicates} valid fits in {max_attempts} attempts; "
        f"last failure: {failures[-1]}"
    )
