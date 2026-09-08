"""Reproducible SINGLE study specifications, revision and complete reports."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .single_search import SingleDesignSearch, single_search_design


@dataclass(frozen=True)
class SingleStudySpecification:
    """Search inputs; weighted nodes specify the actual prior approximation.

    Use parameter nodes/weights from the prior evaluators for uncertain priors.
    This records the integration measure explicitly, without trying to infer the
    original distribution family or input convention from transformed nodes.
    Validation is performed by run(). JSON contains inputs, not executable code.
    """

    parameters: ArrayLike
    dose_bounds: ArrayLike
    prior_weights: ArrayLike | None = None
    total_subjects: float = 100
    criterion: str = "quantile"
    comparison: str | None = None
    model: str = "logistic"
    form: str = "linear"
    measure: str = "sd"
    aggregation: str = "arithmetic"
    quantile: float = 0.05
    max_doses: int = 10
    scan_points: int = 10
    relative_improvement: float = 0.01
    tolerance: float = 1e-10
    max_iterations: int = 1000
    group_totals: ArrayLike | None = None

    def run(self) -> "SingleStudy":
        arguments = asdict(self)
        search = single_search_design(**arguments)
        parameters = np.asarray(arguments["parameters"], dtype=float).copy()
        if parameters.ndim == 1:
            parameters = parameters[None, :]
        bounds = np.asarray(arguments["dose_bounds"], dtype=float).copy()
        weights = (
            np.ones(1)
            if arguments["prior_weights"] is None
            else np.asarray(arguments["prior_weights"], dtype=float).copy()
        )
        weights /= weights.sum()
        for array in (parameters, bounds, weights):
            array.flags.writeable = False
        totals = (
            None
            if arguments["group_totals"] is None
            else np.asarray(arguments["group_totals"], dtype=float).copy()
        )
        if totals is not None:
            totals.flags.writeable = False
        snapshot = replace(
            self,
            parameters=parameters,
            dose_bounds=bounds,
            prior_weights=weights,
            group_totals=totals,
        )
        return SingleStudy(snapshot, search)

    def to_json(self) -> str:
        """Serialize all search settings, including explicit parameter nodes."""
        values = asdict(self)
        for name in ("parameters", "dose_bounds", "prior_weights", "group_totals"):
            if values[name] is not None:
                values[name] = np.asarray(values[name]).tolist()
        for name, value in values.items():
            if isinstance(value, np.generic):
                values[name] = value.item()
        return json.dumps(values, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "SingleStudySpecification":
        """Read a specification; run() validates its numerical and model inputs."""
        values = json.loads(text)
        if not isinstance(values, dict):
            raise ValueError("Study specification must be a JSON object")
        return cls(**values)


@dataclass(frozen=True)
class SingleStudy:
    specification: SingleStudySpecification
    search: SingleDesignSearch

    def revise(self, **changes: object) -> "SingleStudy":
        """Rerun with changed specification fields, preserving this study."""
        values = asdict(self.specification)
        values.update(changes)
        return SingleStudySpecification(**values).run()

    def report(self, *, digits: int = 8) -> str:
        """TSV settings, weighted prior nodes, search history and all stage designs."""
        best_report = self.search.best.report(digits=digits)
        specification = asdict(self.specification)
        parameters = np.asarray(specification.pop("parameters"), dtype=float)
        bounds = np.asarray(specification.pop("dose_bounds"), dtype=float)
        weights = np.asarray(specification.pop("prior_weights"), dtype=float)
        totals = specification.pop("group_totals")
        rows = [
            "SINGLE study",
            "Setting\tValue",
            "group_totals\t"
            + ("none" if totals is None else ",".join(f"{v:.{digits}g}" for v in totals)),
        ]
        for name, value in specification.items():
            rendered = (
                f"{value:.{digits}g}"
                if isinstance(value, (float, np.floating))
                else "none"
                if value is None
                else str(value)
            )
            rows.append(f"{name}\t{rendered}")
        rows.extend(
            [
                f"Minimum dose\t{bounds[0]:.{digits}g}",
                f"Maximum dose\t{bounds[1]:.{digits}g}",
                "",
                "Prior: explicit weighted model-parameter nodes",
                "Node\tWeight\t"
                + "\t".join(f"Parameter {i + 1}" for i in range(parameters.shape[1])),
            ]
        )
        for i, (weight, node) in enumerate(zip(weights, parameters, strict=True), start=1):
            rows.append(f"{i}\t{weight:.{digits}g}\t" + "\t".join(f"{v:.{digits}g}" for v in node))
        rows.extend(
            [
                "",
                "Search metric\tValue",
                f"Stop reason\t{self.search.stop_reason}",
                f"Scan evaluations\t{self.search.scan_evaluations}",
                f"Infeasible seeds\t{self.search.infeasible_seeds}",
                f"Failed local starts\t{self.search.failed_local_starts}",
                "",
                "Stage\tEntries per group\tCriterion\tRelative improvement\tAccepted",
            ]
        )
        for i, step in enumerate(self.search.steps, start=1):
            doses = step.design.doses
            counts = (
                ",".join(str(len(x)) for x in doses)
                if isinstance(doses, tuple)
                else str(len(doses))
            )
            improvement = (
                "NA"
                if step.relative_improvement is None
                else f"{step.relative_improvement:.{digits}g}"
            )
            rows.append(
                f"{i}\t{counts}\t{step.design.value:.{digits}g}\t{improvement}\t{str(step.accepted).lower()}"
            )
        rows.extend(["", "Selected design", best_report.rstrip()])
        for i, step in enumerate(self.search.steps, start=1):
            rows.extend(["", f"Stage {i} design", step.design.report(digits=digits).rstrip()])
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 8) -> Path:
        """Write UTF-8 output, replacing the specified destination."""
        destination = Path(path)
        destination.write_text(self.report(digits=digits), encoding="utf-8")
        return destination

    def write_specification(self, path: str | Path) -> Path:
        """Save full-precision JSON inputs for subsequent replay or revision."""
        destination = Path(path)
        destination.write_text(self.specification.to_json(), encoding="utf-8")
        return destination
