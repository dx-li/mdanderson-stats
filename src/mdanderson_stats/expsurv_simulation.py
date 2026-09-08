"""EXPSURV exponential and covariate examples with explicit random state."""

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


def _study_followup(lifetime: np.ndarray, arrival: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale by sample SD without squaring potentially extreme latent lifetimes."""
    if not np.all(np.isfinite(lifetime)) or lifetime.max() <= 0:
        raise RuntimeError("latent lifetimes must be finite with positive variation")
    relative = lifetime / lifetime.max()
    deviation = np.std(relative, ddof=1)
    if not np.isfinite(deviation) or deviation <= 0:
        raise RuntimeError("latent lifetimes have no representable sample standard deviation")
    death = arrival + relative / deviation
    return np.minimum(2, death) - arrival, (death < 2).astype(float)


def generate_exploratory_data(
    n: int,
    *,
    rng: int | np.random.Generator | None = None,
) -> ExploratoryTable:
    """Generate EXPSURV GEN-DATA covariates, arrivals and study-end follow-up.

    X,Y are independent uniform draws, Z=X*Y. Latent survival has rate 10X+Y,
    then is divided by its sample SD. Arrival is uniform on [0,2); deaths at
    or after 2 are censored at 2. Returns duration-sorted aligned columns.
    NumPy streams replace XLISP-STAT; no source seed equivalence is promised.
    """
    size = count(n, "n")
    if size.ndim != 0 or size < 2:
        raise ValueError("n must be an integer of at least two for sample standard deviation")
    generator = np.random.default_rng(rng)
    x, y = generator.random(int(size)), generator.random(int(size))
    rates = 10 * x + y
    if np.any(rates <= 0):
        raise RuntimeError("generated covariates imply a zero exponential rate")
    lifetime = generator.exponential(1 / rates)
    arrival = 2 * generator.random(int(size))
    duration, status = _study_followup(lifetime, arrival)
    return ExploratoryTable(
        ("length", "arrive", "status", "x", "y", "z"),
        np.column_stack((duration, arrival, status, x, y, x * y)),
    ).cosort("length")
