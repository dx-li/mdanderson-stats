"""Independent Python implementations of MD Anderson catalog methods."""

from .goodness_of_fit import GoodnessOfFit, chi_square_gof
from .intervals import binomial_interval, bp1ci_poisson_interval, poisson_interval
from .multiplicity import (
    MultipleTestingResult,
    multiple_testing,
    rom_critical_values,
    sharpened_testing,
)
from .nonparametric import NonparametricFit, NonparametricFitError, nonparametric_pvalues
from .numerics import invert_monotone, normal_tails
from .pvalue_models import (
    OrderStatisticDiagnostics,
    clustered_pvalues,
    order_statistic_diagnostics,
)
from .ranges import RangeComparisons, kwrange, range2
from .schweder import (
    SchwederBootstrap,
    SchwederFit,
    SchwederFitError,
    schweder_bootstrap,
    schweder_fit,
)

__all__ = [
    "GoodnessOfFit",
    "MultipleTestingResult",
    "NonparametricFit",
    "NonparametricFitError",
    "OrderStatisticDiagnostics",
    "RangeComparisons",
    "SchwederBootstrap",
    "SchwederFit",
    "SchwederFitError",
    "binomial_interval",
    "bp1ci_poisson_interval",
    "chi_square_gof",
    "clustered_pvalues",
    "invert_monotone",
    "kwrange",
    "multiple_testing",
    "normal_tails",
    "nonparametric_pvalues",
    "order_statistic_diagnostics",
    "poisson_interval",
    "range2",
    "rom_critical_values",
    "schweder_bootstrap",
    "schweder_fit",
    "sharpened_testing",
]
