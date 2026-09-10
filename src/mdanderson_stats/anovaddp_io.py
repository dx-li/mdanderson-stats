"""Readers and predictive text outputs for the original ANOVA DDP workflow."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .anovaddp_clusters import _spd
from .anovaddp_prediction import AnovaDDPPrediction


@dataclass(frozen=True)
class AnovaDDPData:
    time: FloatArray
    observations: FloatArray
    subject: NDArray[np.int64]
    design: FloatArray
    prediction_design: FloatArray
    initial_parameters: FloatArray
    initial_covariance: FloatArray
    base_covariance: FloatArray
    base_prior: FloatArray
    observations_per_subject: NDArray[np.int64]


def _read(path: str | Path, *, matrix: bool) -> FloatArray:
    """Whitespace-separated numeric input; vectors may span arbitrary lines."""
    p = Path(path)
    if matrix:
        values = np.loadtxt(p, dtype=float, ndmin=2)
    else:
        tokens = [
            token for line in p.read_text().splitlines() for token in line.split("#")[0].split()
        ]
        values = np.array([float(token) for token in tokens])
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"{p}: expected nonempty finite numeric data")
    return values


def read_anovaddp_data(
    *,
    time: str | Path,
    observations: str | Path,
    initial_covariance: str | Path,
    initial_parameters: str | Path,
    base_covariance: str | Path,
    observations_per_subject: str | Path,
    design: str | Path,
    prediction_design: str | Path,
    base_prior: ArrayLike | str | Path,
) -> AnovaDDPData:
    """Read all nine R input roles, deriving zero-based IDs from Npat counts.

    Observation vectors must be grouped in design-row order, as in the source.
    Matrix files have one row per subject/scenario, without R's internal transpose.
    base_prior may be a numeric vector or a whitespace-separated file.
    """
    t = _read(time, matrix=False)
    y = _read(observations, matrix=False)
    n = _read(observations_per_subject, matrix=False)
    x = _read(design, matrix=True)
    xp = _read(prediction_design, matrix=True)
    theta = _read(initial_parameters, matrix=True)
    if (
        t.shape != y.shape
        or n.size != x.shape[0]
        or not 1 <= n.size <= 10000
        or np.any(n < 1)
        or np.any(n != np.floor(n))
        or np.any(n > t.size)
        or n.sum() != t.size
        or not 1 <= x.shape[1] <= 20
        or xp.shape[1] != x.shape[1]
        or theta.shape != (n.size, 6)
        or np.any(x[:, 0] != 1)
        or np.any(xp[:, 0] != 1)
    ):
        raise ValueError("inconsistent observations, positive subject counts or design dimensions")
    s, _ = _spd(_read(initial_covariance, matrix=True), 6, "initial_covariance")
    c, _ = _spd(_read(base_covariance, matrix=True), 5 * x.shape[1], "base_covariance")
    if np.any(c[:5, 5:] != 0):
        raise ValueError("base intercept covariance must be independent of other blocks")
    if isinstance(base_prior, (str, Path)):
        b = _read(base_prior, matrix=False)
    else:
        if np.iscomplexobj(base_prior):
            raise ValueError("base_prior must be real")
        b = finite(base_prior, "base_prior")
    if b.shape != (5 * x.shape[1],):
        raise ValueError("base_prior requires five coefficients per design column")
    counts = n.astype(np.int64)
    subject = np.repeat(np.arange(n.size, dtype=np.int64), counts)
    return AnovaDDPData(
        _freeze(t),
        _freeze(y),
        np.frombuffer(subject.tobytes(), dtype=np.int64),
        _freeze(x),
        _freeze(xp),
        _freeze(theta),
        _freeze(s),
        _freeze(c),
        _freeze(b),
        np.frombuffer(counts.tobytes(), dtype=np.int64),
    )


def anovaddp_r_outputs(prediction: AnovaDDPPrediction) -> dict[str, FloatArray]:
    """The five named arrays returned by the original R wrapper, without copying."""
    return dict(
        m=prediction.common_effect,
        a0=prediction.baseline_mean,
        a02=prediction.baseline_second_moment,
        f0=prediction.prediction_mean,
        f02=prediction.prediction_second_moment,
    )


def write_anovaddp_prediction(
    prediction: AnovaDDPPrediction, directory: str | Path
) -> tuple[Path, ...]:
    """Write the seven native matrix filenames and an explicit time.txt grid.

    Matrices retain native row/column orientation and 17-digit float precision.
    Refuses existing output paths. A filesystem error may leave partial output.
    """
    destination = Path(directory)
    arrays = dict(
        comeff=prediction.common_effect,
        predstudy3=prediction.study3,
        nadir=prediction.nadir,
        base=prediction.baseline_mean,
        base2=prediction.baseline_second_moment,
        prediction=prediction.prediction_mean,
        prediction2=prediction.prediction_second_moment,
        time=prediction.time,
    )
    paths = tuple(destination / f"{name}.txt" for name in arrays)
    for path, value in zip(paths, arrays.values()):
        if path.exists():
            raise FileExistsError(path)
        if not np.isfinite(value).all():
            raise ValueError(f"{path.name}: nonfinite output")
    destination.mkdir(parents=True, exist_ok=True)
    for path, value in zip(paths, arrays.values()):
        with path.open("x") as stream:
            np.savetxt(stream, value, fmt="%.17g")
    return paths
