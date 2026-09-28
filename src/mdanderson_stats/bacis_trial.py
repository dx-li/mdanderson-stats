"""One-trial BaCIS summaries assembled from the package's two model stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .bacis import BaCISFit, _readonly, bacis_fit
from .bacis_ess import BaCISEquivalentSampleSize, bacis_equivalent_sample_size

_BACIS_TRIAL_ROWS = (
    "Prob(p_i>phi_1)",
    "Prob(p_i>phi_2)",
    "Prob(theta>0)",
    "Classified to high response cluster",
    "The treatment is effective",
    "Posterior Resp.",
    "Observed Resp.",
    "Number of response",
    "Total sample size",
    "Effective sample size",
)


@dataclass(frozen=True)
class BaCISOneTrialResult:
    """Full-precision one-trial rows plus the native-style rounded report.

    ``values`` and ``report_values`` have shape ``(10, groups)``. The report
    is rounded to three decimal places, while all model decisions remain based
    on the full precision values. Singleton clusters use the package's exact
    beta posterior summaries rather than a finite-chain approximation.
    """

    successes: FloatArray
    trials: FloatArray
    row_labels: tuple[str, ...]
    values: FloatArray
    report_values: FloatArray
    fit: BaCISFit
    equivalent_sample_size: BaCISEquivalentSampleSize


def bacis_one_trial(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    phi_low: float = 0.1,
    phi_high: float = 0.3,
    classification_precision: float | None = None,
    classification_cutoff: float | None = None,
    adaptive_weighting: str = "subgroup",
    mean_precision: float = 0.1,
    precision_shape: float = 50,
    precision_rate: float = 2,
    efficacy_cutoff: float = 0.92,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    seed: int | None = None,
) -> BaCISOneTrialResult:
    """Fit BaCIS once and assemble its ten-row one-trial numerical summary.

    The wrapper follows native ``bacisOneTrial`` row semantics, including
    strict ``>`` cluster and efficacy indicators, and computes ESS from the
    same fitted probability draws. Its hierarchical precision-rate default
    is the native value 2 (``bacis_fit`` itself defaults to 10 in this
    package). Sampling defaults stay bounded at the package's 2,000 draws,
    rather than the native 50,000 MCMC iterations. No DIC or latent-theta
    sampling is performed here.
    """
    fit = bacis_fit(
        successes,
        trials,
        phi_low=phi_low,
        phi_high=phi_high,
        classification_precision=classification_precision,
        classification_cutoff=classification_cutoff,
        adaptive_weighting=adaptive_weighting,
        mean_precision=mean_precision,
        precision_shape=precision_shape,
        precision_rate=precision_rate,
        efficacy_cutoff=efficacy_cutoff,
        draws=draws,
        warmup=warmup,
        chains=chains,
        seed=seed,
    )
    ess = bacis_equivalent_sample_size(fit.probability_samples, successes, trials)
    y = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    classification = fit.classification
    values = np.vstack(
        (
            fit.efficacy_probability,
            fit.high_response_probability,
            classification.high_probability,
            (classification.cluster == 2).astype(float),
            fit.efficacious.astype(float),
            fit.posterior_mean,
            y / n,
            y,
            n,
            ess.equivalent_sample_size,
        )
    )
    if values.shape != (len(_BACIS_TRIAL_ROWS), y.size) or not np.isfinite(values).all():
        raise ArithmeticError("BaCIS one-trial summary is not finite")
    return BaCISOneTrialResult(
        _readonly(y),
        _readonly(n),
        _BACIS_TRIAL_ROWS,
        _readonly(values),
        _readonly(np.round(values, 3)),
        fit,
        ess,
    )
