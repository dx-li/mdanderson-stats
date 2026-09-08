"""Reproducible SEQBIN study configuration, revision and complete numerical reports."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite
from .seqbin import SeqBinDesign, SeqBinProperties
from .seqbin_calibration import (
    SeqBinCalibration,
    SeqBinTailCalibration,
    seqbin_calibrate,
    seqbin_calibrate_tails,
)
from .seqbin_table import SeqBinBoundaryTable, _boundary_table


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in a study specification")


@dataclass(frozen=True)
class SeqBinStudySpecification:
    max_subjects: int
    prior: ArrayLike = (1, 1)
    null_probability: float = 0.2
    alternative: str = "greater"
    looks: ArrayLike | None = None
    tail_probability: ArrayLike | None = None
    significance: ArrayLike | None = None
    selection: str = "conservative"
    tail_bounds: ArrayLike = (1e-6, 1 - 1e-6)
    legacy_bounds: bool = False
    probabilities: ArrayLike = (0.1, 0.2, 0.3)

    def run(self) -> "SeqBinStudy":
        """Build a validated independent snapshot; calibrate if significance is given.

        Specify either posterior tail_probability or a significance target. With
        neither, the posterior tail cutoff is .05. A pair of significance targets
        requests separate-tail calibration and requires alternative=two-sided.
        """
        values = asdict(self)
        probabilities = finite(values["probabilities"], "probabilities")
        if (
            probabilities.ndim != 1
            or probabilities.size > 101
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("probabilities must be a vector of at most 101 values in [0,1]")
        bounds = finite(values["tail_bounds"], "tail_bounds")
        if bounds.shape != (2,) or not 0 < bounds[0] < bounds[1] < 1:
            raise ValueError("tail_bounds must increase within (0,1)")
        if values["selection"] not in ("conservative", "nearest"):
            raise ValueError("selection must be conservative or nearest")
        common = {
            name: values[name]
            for name in (
                "max_subjects",
                "prior",
                "null_probability",
                "looks",
                "legacy_bounds",
            )
        }
        calibration: SeqBinCalibration | SeqBinTailCalibration | None = None
        target: float | tuple[float, ...] | None = None
        if values["significance"] is None:
            design = SeqBinDesign(
                **common,
                alternative=values["alternative"],
                tail_probability=0.05
                if values["tail_probability"] is None
                else values["tail_probability"],
            )
        else:
            if values["tail_probability"] is not None:
                raise ValueError("Specify tail_probability or significance, not both")
            levels = finite(values["significance"], "significance")
            if levels.ndim == 0:
                target = float(levels)
                calibration = seqbin_calibrate(
                    **common,
                    significance=target,
                    alternative=values["alternative"],
                    selection=values["selection"],
                    tail_bounds=bounds,
                )
                design = calibration.chosen.design
            elif levels.shape == (2,) and values["alternative"] == "two-sided":
                target = tuple(map(float, levels))
                calibration = seqbin_calibrate_tails(
                    **common,
                    significance=target,
                    selection=values["selection"],
                    tail_bounds=bounds,
                )
                design = calibration.design
            else:
                raise ValueError(
                    "significance must be scalar, or [low, high] for a two-sided design"
                )
        snapshot = replace(
            self,
            prior=design.prior,
            looks=None if values["looks"] is None else tuple(map(int, design.looks)),
            tail_probability=design.tail_probability if target is None else None,
            significance=target,
            probabilities=tuple(map(float, probabilities)),
            tail_bounds=tuple(map(float, bounds)),
        )
        properties = design.operating_characteristics(np.r_[design.null_probability, probabilities])
        return SeqBinStudy(snapshot, design, calibration, properties)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_json_value, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "SeqBinStudySpecification":
        values = json.loads(text)
        if not isinstance(values, dict):
            raise ValueError("Study specification must be a JSON object")
        return cls(**values)


@dataclass(frozen=True)
class SeqBinStudy:
    specification: SeqBinStudySpecification
    design: SeqBinDesign
    calibration: SeqBinCalibration | SeqBinTailCalibration | None
    # Probability axis: null first, then specification.probabilities in input order.
    properties: SeqBinProperties

    def revise(self, **changes: object) -> "SeqBinStudy":
        values = asdict(self.specification)
        values.update(changes)
        return SeqBinStudySpecification(**values).run()

    def boundary_table(self, *, compact: bool = False) -> SeqBinBoundaryTable:
        return _boundary_table(
            self.design, self.properties.quit_low[0], self.properties.quit_high[0], compact
        )

    def report(self, *, digits: int = 8, compact: bool = False) -> str:
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 17
        ):
            raise ValueError("digits must be an integer from 1 to 17")
        table = self.boundary_table(compact=compact)

        def number(value: float) -> str:
            return "NA" if np.isnan(value) else f"{value:.{digits}g}"

        rows = ["SEQBIN study", "Setting\tJSON value"]
        for name, value in asdict(self.specification).items():
            rows.append(name + "\t" + json.dumps(value, default=_json_value, allow_nan=False))
        rows.append("Actual posterior tail cutoffs\t" + json.dumps(self.design.tail_probability))
        if self.calibration is not None:
            rows.extend(
                [
                    "",
                    "Calibration",
                    "Target side\tRequested level\tSelection\tPosition\t"
                    "Tail cutoff\tAchieved level",
                ]
            )
            calibrations = (
                [("low", self.calibration.low), ("high", self.calibration.high)]
                if isinstance(self.calibration, SeqBinTailCalibration)
                else [(self.design.alternative, self.calibration)]
            )
            for side, result in calibrations:
                for position in ("chosen", "lower", "upper"):
                    point = getattr(result, position)
                    cutoff, level = (
                        ("NA", "NA")
                        if point is None
                        else (
                            number(float(point.design.tail_probability)),
                            number(point.significance),
                        )
                    )
                    rows.append(
                        f"{side}\t{number(result.requested_significance)}\t{result.selection}\t{position}\t{cutoff}\t{level}"
                    )
        rows.extend(
            ["", "Null boundary table (inclusive continuation bounds)", "\t".join(table.columns)]
        )
        rows.extend("\t".join(number(v) for v in row) for row in table.rows)
        p = self.properties
        rows.extend(
            [
                "",
                "Operating characteristics",
                "Case\tProbability\tReject\tComplete\tExpected subjects\tQuit low\t"
                "Expected subjects given low\tQuit high\tExpected subjects given high",
            ]
        )
        for i, probability in enumerate(p.probability):
            row = [
                probability,
                p.rejection_probability[i],
                p.complete[i],
                p.expected_subjects[i],
                p.quit_low[i].sum(),
                p.expected_subjects_quit_low[i],
                p.quit_high[i].sum(),
                p.expected_subjects_quit_high[i],
            ]
            rows.append(
                ("null" if i == 0 else f"alternative {i}")
                + "\t"
                + "\t".join(number(v) for v in row)
            )
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 8, compact: bool = False) -> Path:
        destination = Path(path)
        destination.write_text(self.report(digits=digits, compact=compact), encoding="utf-8")
        return destination

    def write_specification(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.specification.to_json(), encoding="utf-8")
        return destination
