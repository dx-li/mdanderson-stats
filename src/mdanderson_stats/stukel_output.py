"""STUKEL regression tables and observed-dose/modified-link plots."""

from __future__ import annotations

from math import erfc, sqrt
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .stukel import stukel_log_odds, stukel_probability
from .stukel_fit import StukelFit

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_FAMILIES = (
    "Both shapes fixed",
    "Alpha1 free; alpha2 fixed",
    "Alpha2 free; alpha1 fixed",
    "Shared shape (alpha1 = alpha2)",
    "Opposite shapes (alpha1 = -alpha2)",
    "Both shapes free",
)


def format_stukel(fit: StukelFit, *, digits: int = 4) -> str:
    """Return glim.p's coefficient/Wald and raw-data/regression tables as text.

    Values use significant digits without modifying global print settings.
    Unavailable covariance or zero standard errors produce NA Wald statistics.
    Two-sided normal-tail p-values use erfc to avoid subtracting a CDF from one.
    """
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
        raise ValueError("digits must be an integer from 1 to 17")
    extra = 0 if fit.family == 0 else 2 if fit.family == 5 else 1
    p = fit.coefficients.size - extra
    names = (["Intercept"] if fit.intercept else []) + [
        f"beta{i + 1}" for i in range(p - int(fit.intercept))
    ]
    names += (
        []
        if extra == 0
        else ["alpha1", "alpha2"]
        if extra == 2
        else ["alpha2" if fit.family == 2 else "alpha1"]
    )

    def number(value: float) -> str:
        return format(float(value), f".{digits}g")

    rows = [
        "STUKEL generalized logistic regression",
        _FAMILIES[fit.family],
        f"alpha1 = {number(fit.alpha[0])}; alpha2 = {number(fit.alpha[1])}",
        "",
        "parameter\tcoef\tse(coef)\tz\tp",
    ]
    errors = fit.standard_errors
    for i, (name, coef) in enumerate(zip(names, fit.coefficients, strict=True)):
        se = errors[i] if errors is not None else None
        if se is None or se <= 0 or not np.isfinite(se):
            fields = ["NA" if se is None else number(se), "NA", "NA"]
        else:
            z = float(coef / se)
            fields = [number(se), number(z), number(erfc(abs(z) / sqrt(2)))]
        rows.append("\t".join([name, number(coef), *fields]))
    rows += [
        "",
        "model\tdeviance\tdf\tscale",
        "\t".join(
            [
                "raw data",
                number(fit.null_deviance),
                str(fit.objective.probabilities.size - 1),
                number(fit.null_dispersion),
            ]
        ),
        "\t".join(
            [
                "regression",
                number(fit.deviance),
                str(fit.residual_df),
                number(fit.dispersion),
            ]
        ),
        "",
        fit.inference_message,
    ]
    return "\n".join(rows) + "\n"


def plot_stukel(
    fit: StukelFit,
    x: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    plot_dose: bool = False,
    plot_link: bool = True,
) -> tuple[Axes, ...]:
    """Plot observed fractions/fitted probabilities and/or the modified link.

    Return axes in dose-then-link order. Requires the optional plot extra;
    does not show, save, or alter global style/backend. Predictions are evaluated
    at the supplied data using fit's coefficients. Dose needs one covariate;
    link plots support multiple covariates. Rows stay paired during sorting.
    """
    if not isinstance(plot_dose, bool) or not isinstance(plot_link, bool):
        raise ValueError("plot_dose and plot_link must be boolean")
    if not plot_dose and not plot_link:
        raise ValueError("At least one plot must be requested")
    x = finite(x, "x")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("x must be a nonempty observations-by-covariates matrix")
    y, n = count(successes, "successes"), count(trials, "trials")
    if y.shape != (len(x),) or n.shape != y.shape or np.any(n == 0) or np.any(y > n):
        raise ValueError("Counts must match x rows with 0 <= successes <= positive trials")
    if plot_dose and x.shape[1] != 1:
        raise ValueError("Dose plot requires exactly one covariate")
    extra = 0 if fit.family == 0 else 2 if fit.family == 5 else 1
    beta = fit.coefficients[: fit.coefficients.size - extra]
    design = np.column_stack((np.ones(len(x)), x)) if fit.intercept else x
    if design.shape[1] != beta.size:
        raise ValueError("x covariates must match the fitted coefficients")
    eta = design @ beta
    h = stukel_log_odds(eta, *fit.alpha)
    if not np.all(np.isfinite(h)):
        raise ValueError("Plot log odds exceed numerical range")
    from matplotlib import pyplot as plt

    _, grid = plt.subplots(
        1,
        int(plot_dose) + int(plot_link),
        squeeze=False,
        figsize=(6 * (int(plot_dose) + int(plot_link)), 4),
        layout="constrained",
    )
    axes = tuple(grid[0])
    if plot_dose:
        ax = axes[0]
        order = np.argsort(x[:, 0], kind="stable")
        dose = x[order, 0]
        ax.plot(dose, (y / n)[order], "o", markerfacecolor="none", label="Observed")
        ax.plot(dose, stukel_probability(eta, *fit.alpha)[order], label="Fitted")
        ax.set_xlabel("Dose")
        ax.set_ylabel("Probability")
        ax.set_title("Estimated logistic probabilities")
        ax.legend()
    if plot_link:
        ax = axes[-1]
        order = np.argsort(eta, kind="stable")
        ax.plot(eta[order], h[order], label="Generalized")
        ax.plot(eta[order], eta[order], "--", label="Standard")
        ax.set_xlabel("Linear predictor eta")
        ax.set_ylabel("Transformed log odds h")
        ax.set_title("Modified link h = f(eta)")
        ax.legend()
    return axes
