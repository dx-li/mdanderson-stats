"""EXPSURV two-sample exponential examples with explicit random state."""

import numpy as np

from ._validation import count, scalar
from .expsurv_data import ExploratoryTable


def generate_exponential_samples(
    n_first: int = 100,
    n_second: int = 50,
    censor_probability: float = 0.1,
    rate_first: float = 1,
    rate_second: float = 10,
    *,
    rng: int | np.random.Generator | None = None,
) -> tuple[ExploratoryTable, ExploratoryTable]:
    """Generate GEN-EXPO-DATA samples; defaults implement GEN-EXPO-EXAMPLE.

    Censoring status is drawn independently of latent exponential survival.
    Censored follow-up is a uniform fraction of that latent time. This is the
    source's illustrative mechanism, not independent random censoring times.
    NumPy random streams replace XLISP-STAT; identical source seeds need not match.
    """
    sizes = count([n_first, n_second], "sample sizes")
    if sizes.shape != (2,) or np.any(sizes < 1):
        raise ValueError("sample sizes must be positive integers")
    probability = scalar(censor_probability, "censor_probability")
    rates = np.array([scalar(rate_first, "rate_first"), scalar(rate_second, "rate_second")])
    if probability < 0 or probability > 1:
        raise ValueError("censor_probability must be between zero and one")
    if np.any(rates <= 0):
        raise ValueError("rates must be positive")
    with np.errstate(over="raise", divide="raise"):
        try:
            scales = 1 / rates
        except FloatingPointError as error:
            raise ValueError("rates imply unrepresentable exponential scales") from error
    generator = np.random.default_rng(rng)
    times = [generator.exponential(scale, int(n)) for scale, n in zip(scales, sizes, strict=True)]
    statuses = [generator.binomial(1, 1 - probability, int(n)) for n in sizes]
    result = []
    for time, status in zip(times, statuses, strict=True):
        censored = status == 0
        time[censored] *= generator.random(int(censored.sum()))
        if not np.all(np.isfinite(time)):
            raise RuntimeError("generated follow-up overflowed; use less extreme rates")
        result.append(
            ExploratoryTable(("time", "status"), np.column_stack((time, status))).cosort("time")
        )
    return result[0], result[1]
