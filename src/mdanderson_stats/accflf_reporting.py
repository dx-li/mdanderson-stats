"""ACCFLF's population survival average and numerical reports."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .accflf import accflf_logf, accflf_shape
from .accflf_model import AccflfFit
from .accflf_search import AccflfGrid, AccflfModelResult, AccflfShapeSearch


def accflf_marginal_survival(
    time: ArrayLike,
    *,
    p: float,
    q: float,
    sigma: float,
    coefficients: ArrayLike,
    covariates: ArrayLike,
    log: bool = False,
) -> FloatArray:
    """Native SRVPRB: equal-row average over the supplied covariate distribution.

    Every requested positive time is averaged over all covariate rows. Source
    multiplicity is deliberately not used here. Duplicate linear predictors are
    grouped, and time batches bound temporary memory. log=True retains tiny tails.
    """
    if any(np.iscomplexobj(v) for v in (time, coefficients, covariates)):
        raise ValueError("time, coefficients and covariates must be real")
    t, beta, x = (
        finite(time, "time"),
        finite(coefficients, "coefficients"),
        finite(covariates, "covariates"),
    )
    scale = scalar(sigma, "sigma")
    if (
        t.ndim != 1
        or not 1 <= t.size <= 20000
        or np.any(t <= 0)
        or x.ndim != 2
        or not 1 <= x.shape[0] <= 20000
        or x.shape[1] > 16
        or beta.shape != (x.shape[1] + 1,)
        or scale <= 0
        or not isinstance(log, bool)
    ):
        raise ValueError(
            "require positive times/sigma, aligned covariates/coefficients and boolean log"
        )
    location, counts = np.unique(beta[0] + x @ beta[1:], return_counts=True)
    if location.size * t.size > 20_000_000:
        raise ValueError("at most 20 million distinct predictor/time pairs are supported")
    shape = accflf_shape(p, q)
    logweight = np.log(counts.astype(float)) - np.log(x.shape[0])
    result = np.empty(t.shape)
    batch = max(1, 100000 // location.size)
    for start in range(0, t.size, batch):
        stop = min(start + batch, t.size)
        w = (np.log(t[start:stop, None]) - location) / scale
        tails = accflf_logf(w, shape.numerator_df, shape.denominator_df).log_survival
        result[start:stop] = np.minimum(0, logsumexp(tails + logweight, axis=1))
    return _freeze(result if log else np.exp(result))


def accflf_report(
    result: AccflfFit | AccflfShapeSearch | AccflfGrid | tuple[AccflfModelResult, ...],
    *,
    covariate_names: tuple[str, ...] | None = None,
) -> str:
    """Readable fit/search/grid/six-model report, including failure diagnostics.

    Return text for printing or Path.write_text; no files are silently overwritten.
    Likelihoods use the native log-time convention. This is not bytewise console emulation.
    """
    lines = ["ACCFLF: accelerated log-F model", "Log likelihood convention: log event times"]

    def fit_lines(fit: AccflfFit) -> list[str]:
        names = (
            tuple(f"COV{i + 1}" for i in range(fit.coefficients.size - 1))
            if covariate_names is None
            else covariate_names
        )
        if len(names) != fit.coefficients.size - 1 or any(
            not isinstance(n, str) or not n.strip() for n in names
        ):
            raise ValueError("covariate_names must match the fitted coefficients")
        output = [
            f"P={fit.p:.12g} Q={fit.q:.12g} SIGMA={fit.sigma:.12g}",
            f"DFN={fit.shape.numerator_df:.12g} DFD={fit.shape.denominator_df:.12g} "
            f"TAU={fit.shape.tau:.12g}",
            f"DF clipping: lower={fit.shape.lower_clipped} upper={fit.shape.upper_clipped}",
        ]
        output.extend(
            f"{name}={value:.12g}" for name, value in zip(("MU", *names), fit.coefficients)
        )
        output.extend(
            [
                f"Log likelihood={fit.log_likelihood:.12g}",
                f"Time-density log likelihood={fit.time_log_likelihood:.12g}",
                f"Iterations={fit.iterations} Score error={fit.score_error:.6g} "
                f"Sigma fixed={fit.sigma_fixed}",
                "Covariance: conditional on p,q; coordinates log(sigma), MU, covariates",
                np.array2string(fit.covariance, precision=8, threshold=fit.covariance.size),
            ]
        )
        return output

    def search_lines(search: AccflfShapeSearch) -> list[str]:
        output = [
            f"Local shape-search convergence={search.converged}; "
            "not a global-optimality certificate",
            f"Evaluated shapes={search.evaluated_shapes} Failed shapes={len(search.failed_shapes)}",
        ]
        for run in search.runs:
            ll = "failed" if run.fit is None else f"{run.fit.log_likelihood:.12g}"
            output.append(
                f"Start ({run.start_p:.8g},{run.start_q:.8g}): LL={ll}; "
                f"converged={run.converged}; {run.message}"
            )
        output.extend(
            f"Failed ({p:.8g},{q:.8g}): {reason}" for p, q, reason in search.failed_shapes
        )
        return output

    if isinstance(result, AccflfFit):
        lines.extend(fit_lines(result))
    elif isinstance(result, AccflfShapeSearch):
        lines.extend(fit_lines(result.best))
        lines.extend(search_lines(result))
    elif isinstance(result, AccflfGrid):
        lines.append("P Q Log-likelihood / error")
        for i, p in enumerate(result.p):
            for j, q in enumerate(result.q):
                error = result.errors[i * result.q.size + j]
                text = error if error is not None else f"{result.log_likelihood[i, j]:.12g}"
                lines.append(f"{p:.12g} {q:.12g} {text}")
    elif isinstance(result, tuple) and all(isinstance(r, AccflfModelResult) for r in result):
        for model in result:
            lines.append(f"\nModel: {model.name}")
            lines.extend(
                fit_lines(model.fit) if model.fit is not None else [f"ERROR: {model.error}"]
            )
            if model.search is not None:
                lines.extend(search_lines(model.search))
    else:
        raise ValueError("unsupported ACCFLF report result")
    return "\n".join(lines) + "\n"
