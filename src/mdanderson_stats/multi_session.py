"""MULTI analysis sessions and portable reports, replacing global menus/sinks."""

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from inspect import signature
from json import dumps, loads
from math import isfinite
from pathlib import Path
from typing import cast

import numpy as np

from .beta_mixture_bootstrap import beta_mixture_bootstrap
from .beta_mixture_selection import fit_beta_mixture_k, select_beta_mixture
from .beta_mixture_testing import beta_mixture_testing
from .multi_input import MultiData, parse_multi_data
from .multiplicity import multiple_testing, sharpened_testing
from .nonparametric_testing import nonparametric_testing
from .pvalue_models import order_statistic_diagnostics
from .schweder import schweder_bootstrap, schweder_fit

_PROCEDURES: dict[str, Callable[..., object]] = {
    function.__name__: function
    for function in (
        schweder_fit,
        schweder_bootstrap,
        multiple_testing,
        sharpened_testing,
        fit_beta_mixture_k,
        select_beta_mixture,
        beta_mixture_testing,
        beta_mixture_bootstrap,
        nonparametric_testing,
        order_statistic_diagnostics,
    )
}


def _record(value: object) -> object:
    """Snapshot supported scientific results into strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, np.generic):
        return _record(value.item())
    if isinstance(value, np.ndarray):
        return _record(value.tolist())
    if isinstance(value, np.random.Generator):
        return {"generator_state": _record(value.bit_generator.state)}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Report mappings require string keys")
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    raise TypeError(f"Unsupported report value type: {type(value).__name__}")


class MultiSession:
    """Run MULTI procedures on replaceable data, keeping immutable report snapshots.

    Enter data as MULTI text; use change_file for files. run returns the ordinary
    numerical result object and snapshots it in the report. Reports are explicit
    UTF-8 JSON file writes, not global print sinks. No cached fit crosses datasets.
    """

    def __init__(
        self,
        text: str,
        *,
        terminal: bool = False,
        source: str = "text",
        seed: int | None = None,
    ) -> None:
        self._datasets: list[dict[str, object]] = []
        self._events: list[dict[str, object]] = []
        self._rng = np.random.default_rng(seed)
        self.change_data(text, terminal=terminal, source=source)

    @property
    def data(self) -> MultiData:
        # Return independent arrays so callers cannot modify future session runs.
        return MultiData(self._data.pvalues.copy(), self._data.order.copy(), self._data.warnings)

    @property
    def procedures(self) -> tuple[str, ...]:
        return tuple(_PROCEDURES)

    def change_data(self, text: str, *, terminal: bool = False, source: str = "text") -> MultiData:
        """Validate replacement data before changing session state or history."""
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        data = parse_multi_data(text, terminal=terminal)
        snapshot = {
            "id": len(self._datasets) + 1,
            "source": source,
            "terminal": terminal,
            "entered_values": _record(data.entered_values),
            "sorted_values": _record(data.pvalues),
            "order": _record(data.order),
            "warnings": _record(data.warnings),
        }
        self._datasets.append(snapshot)
        self._data = data
        self._events.append({"event": "change_data", "dataset": len(self._datasets)})
        return self.data

    def change_file(self, path: str | Path) -> MultiData:
        """Read and validate a UTF-8 file before replacing observations."""
        path = Path(path)
        return self.change_data(path.read_text(encoding="utf-8"), source=str(path))

    def set_seed(self, seed: int | None) -> None:
        """Reset the session's NumPy generator and record its actual initial state."""
        generator = np.random.default_rng(seed)
        snapshot = _record(generator)
        self._rng = generator
        self._events.append({"event": "set_seed", "state": snapshot})

    def run(self, procedure: str, **settings: object) -> object:
        """Run a named procedure with explicit keyword settings on entered-order data.

        Names match the package APIs listed in procedures. Omitted RNG arguments
        use the session generator. Mixture selection defaults to desktop workflow;
        omitted sharpened null_estimate is freshly estimated from current data.
        All effective defaults are recorded before computation. Numerical failures
        are recorded and re-raised, with no fabricated successful result.
        """
        if procedure not in _PROCEDURES:
            raise ValueError(f"Unknown MULTI procedure: {procedure}")
        entry: dict[str, object] = {
            "event": "analysis",
            "dataset": len(self._datasets),
            "procedure": procedure,
        }
        function = _PROCEDURES[procedure]
        parameters = signature(function)
        if procedure == "select_beta_mixture":
            settings.setdefault("workflow", "desktop")
        if procedure == "sharpened_testing" and "null_estimate" not in settings:
            entry["null_estimate_source"] = "schweder_fit(alpha=0.05) on current dataset"
            try:
                settings["null_estimate"] = schweder_fit(self._data.entered_values).null_estimate
            except ValueError as error:
                entry.update(
                    status="failed",
                    settings=_record(settings),
                    error={"type": type(error).__name__, "message": str(error)},
                )
                self._events.append(entry)
                raise
        if "rng" in parameters.parameters and settings.get("rng") is None:
            settings["rng"] = self._rng
        bound = parameters.bind(self._data.entered_values, **settings)
        bound.apply_defaults()
        effective = {key: value for key, value in bound.arguments.items() if key != "pvalues"}
        entry["settings"] = _record(effective)
        try:
            result = function(*bound.args, **bound.kwargs)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            entry.update(
                status="failed", error={"type": type(error).__name__, "message": str(error)}
            )
            if "rng" in effective:
                entry["rng_after"] = _record(effective["rng"])
            self._events.append(entry)
            raise
        entry.update(status="returned", result=_record(result))
        if "rng" in effective:
            entry["rng_after"] = _record(effective["rng"])
        self._events.append(entry)
        return result

    def report(self) -> dict[str, object]:
        """Return a detached report of datasets, settings, results and failures.

        Named result arrays are table columns in their API's documented order.
        Internal status/message fields (e.g. component_limit) remain present;
        status=returned means a result was returned, not that selection converged.
        Nonfinite diagnostics are tagged objects, never nonstandard JSON numbers.
        """
        return cast(
            dict[str, object],
            loads(
                dumps(
                    {
                        "format": "mdanderson-stats/MULTI-session",
                        "version": 1,
                        "numpy_version": np.__version__,
                        "datasets": self._datasets,
                        "events": self._events,
                    },
                    allow_nan=False,
                )
            ),
        )

    def format_report(self, *, digits: int = 6) -> str:
        """Return Markdown tables, retaining dataset, settings and failure history.

        Observation/rank labels are one-based for reading. Rom critical alphas
        and reciprocal-density scores are labeled separately from adjusted p-values.
        """
        from .multi_report import _format_report

        return _format_report(self.report(), digits)

    def write_text_report(self, path: str | Path, *, digits: int = 6) -> Path:
        """Write the Markdown report as UTF-8, replacing path explicitly."""
        path = Path(path)
        content = self.format_report(digits=digits)
        path.write_text(content, encoding="utf-8")
        return path

    def write_report(self, path: str | Path) -> Path:
        """Write a complete JSON report, replacing path; propagate I/O failures."""
        path = Path(path)
        content = dumps(self.report(), indent=2, allow_nan=False) + "\n"
        path.write_text(content, encoding="utf-8")
        return path
