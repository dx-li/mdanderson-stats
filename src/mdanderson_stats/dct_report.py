"""Readable reports for decentralized clinical-trial sample-size calculations."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from pprint import pformat
from typing import Literal

import numpy as np

from .dct_binary import dct_binary_sample_size
from .dct_normal import DCTNormalSampleSize, dct_normal_sample_size

DCTEndpoint = Literal["continuous", "binary"]
DCTAllocationUnit = Literal["participants", "clusters"]
_MAX_REPORT_BYTES = 100_000
_CITATION = (
    "Tian, Lin, Liu and Yuan, ‘Sample-size determination for decentralized clinical trials,’ "
    "International Journal of Epidemiology, doi:10.1093/ije/dyaf053."
)
_PROCEDURES: dict[str, Callable[..., DCTNormalSampleSize]] = {
    "continuous": dct_normal_sample_size,
    "binary": dct_binary_sample_size,
}


def _scalar_snapshot(name: str, value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    raise TypeError(f"{name} must be a scalar setting")


def _atomic_write(path: str | Path, content: str) -> Path:
    destination = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class DCTSampleSizeReport:
    """Immutable inputs and allocation summary for one DCT planning calculation."""

    endpoint: DCTEndpoint
    allocation_unit: DCTAllocationUnit
    settings: tuple[tuple[str, object], ...]
    unrounded_total: float
    allocation: tuple[tuple[int, int], tuple[int, int]]
    achieved_power: float
    target_power: float
    alpha: float
    sides: int
    citation: str = _CITATION

    @property
    def rounded_total(self) -> int:
        return sum(sum(row) for row in self.allocation)

    def report(self) -> str:
        """Return a readable UTF-8-ready summary of this planning result."""
        control_label = "Control"
        experimental_label = "Experimental"
        unit = self.allocation_unit
        method = (
            "weighted difference-in-proportions z-test normal approximation"
            if self.endpoint == "binary"
            else "weighted two-sample z-test normal approximation"
        )
        lines = [
            "Decentralized clinical-trial sample-size report",
            "================================================",
            f"Endpoint: {self.endpoint}",
            f"Allocation unit: {unit}",
            f"Method: {method}",
            "",
            "Allocation by stratum and arm:",
            f"  Stratum     {control_label:<14} {experimental_label:<14}",
            f"  Onsite      {self.allocation[0][0]:<14d} {self.allocation[0][1]:<14d}",
            f"  Offsite     {self.allocation[1][0]:<14d} {self.allocation[1][1]:<14d}",
            f"Unrounded total requirement: {self.unrounded_total:.12g} {unit}",
            f"Rounded total allocation: {self.rounded_total} {unit}",
            f"Achieved power: {self.achieved_power:.12g} (target {self.target_power:.12g})",
            f"Type I error rate: {self.alpha:.12g}; test sides: {self.sides}",
            "",
            "Effective inputs (defaults included):",
            pformat(dict(self.settings), sort_dicts=False, width=88),
            "",
            "The allocation follows the Python planner's independent upward rounding "
            "for each active stratum-arm cell; it has no dropout inflation. Repeats "
            "are repeated-measure participants when the unit is participants and "
            "cluster sizes when the unit is clusters. Counts are reported in the "
            "selected unit; participant totals are not inferred from cluster counts.",
        ]
        inputs = dict(self.settings)
        help_example = {
            "effect": 10,
            "onsite_sd": 20,
            "offsite_sd": 20,
            "offsite_fraction": 1,
            "relative_bias": 0,
            "randomization_ratio": 1,
            "onsite_repeats": 1,
            "offsite_repeats": 1,
            "onsite_correlation": 0,
            "offsite_correlation": 0,
            "power": 0.8,
            "alpha": 0.05,
            "sides": 2,
            "onsite_experimental_sd": 20,
            "offsite_experimental_sd": 20,
        }
        if self.endpoint == "continuous" and all(
            inputs.get(key) == value for key, value in help_example.items()
        ):
            lines.append(
                "Native help lists 128 for this fully offsite example; this report preserves "
                "the Python formula result and independent upward rounding without adjustment. "
                "Source: MD Anderson DCT calculator help page."
            )
        lines.extend(["", f"Citation: {self.citation}", ""])
        return "\n".join(lines)

    def write_report(self, path: str | Path) -> Path:
        """Render and atomically write this report as UTF-8 text."""
        content = self.report()
        if len(content.encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ValueError("DCT report exceeds the size limit")
        return _atomic_write(path, content)


def dct_sample_size_report(
    endpoint: DCTEndpoint,
    *,
    allocation_unit: DCTAllocationUnit,
    **settings: object,
) -> DCTSampleSizeReport:
    """Calculate and capture a continuous or binary DCT sample-size report.

    ``allocation_unit`` must be stated because the planner supports both
    repeated-measure participants and clusters whose repeat settings represent
    cluster sizes. The returned allocation is expressed only in that unit.
    """
    if not isinstance(endpoint, str) or endpoint not in _PROCEDURES:
        raise ValueError("endpoint must be 'continuous' or 'binary'")
    if not isinstance(allocation_unit, str) or allocation_unit not in {"participants", "clusters"}:
        raise ValueError("allocation_unit must be 'participants' or 'clusters'")
    function = _PROCEDURES[endpoint]
    bound = signature(function).bind(**settings)
    bound.apply_defaults()
    effective = {name: _scalar_snapshot(name, value) for name, value in bound.arguments.items()}
    if endpoint == "continuous":
        for experimental, control in (
            ("onsite_experimental_sd", "onsite_sd"),
            ("offsite_experimental_sd", "offsite_sd"),
        ):
            if effective[experimental] is None:
                effective[experimental] = effective[control]
    result = function(**effective)
    allocation_array = np.asarray(result.allocation)
    if allocation_array.shape != (2, 2) or np.any(allocation_array < 0):
        raise ArithmeticError("DCT planner returned an invalid allocation table")
    allocation = (
        (int(allocation_array[0, 0]), int(allocation_array[0, 1])),
        (int(allocation_array[1, 0]), int(allocation_array[1, 1])),
    )
    report = DCTSampleSizeReport(
        endpoint=endpoint,
        allocation_unit=allocation_unit,
        settings=tuple(effective.items()),
        unrounded_total=result.unrounded_total,
        allocation=allocation,
        achieved_power=result.achieved_power,
        target_power=result.target_power,
        alpha=result.alpha,
        sides=result.sides,
    )
    if len(report.report().encode("utf-8")) > _MAX_REPORT_BYTES:
        raise ValueError("DCT report exceeds the size limit")
    return report
