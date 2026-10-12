"""Bounded saved inputs for the current mTPI spreadsheet's two declared tasks."""

import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass

import numpy as np

from .mtpi import MTPIDesign

_SCHEMA = "mdanderson-stats/mtpi-desktop/v1"
_MARKER = '<script id="mtpi-study" type="application/json">'
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low},{high}]")
    return int(value)


def _design(target, epsilon_lower, epsilon_upper):
    values = (target, epsilon_lower, epsilon_upper)
    if any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError("probability parameters must be real numbers")
    if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        raise ValueError("probability parameters must be real numbers")
    if not all(math.isfinite(v) for v in values) or min(epsilon_lower, epsilon_upper) <= 0:
        raise ValueError("equivalence margins must be finite and positive")
    return MTPIDesign(target=target, lower=target - epsilon_lower, upper=target + epsilon_upper)


@dataclass(frozen=True)
class MTPIDesktopTableStudy:
    """Monitoring-table inputs; no scenario or invented dose grid is required."""

    target: float = 0.3
    epsilon_lower: float = 0.05
    epsilon_upper: float = 0.05
    maximum_patients: int = 30

    def __post_init__(self):
        self.design()
        for name in ("target", "epsilon_lower", "epsilon_upper"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(
            self, "maximum_patients", _integer(self.maximum_patients, "maximum_patients", 1, 200)
        )

    def design(self):
        return _design(self.target, self.epsilon_lower, self.epsilon_upper)

    def to_json(self):
        return _serialize(self, "table")


@dataclass(frozen=True)
class MTPIDesktopStudy:
    """Current spreadsheet simulation inputs, with immutable ordered scenarios.

    Uniform allocation priors and native fixed safety cutoffs are not editable
    spreadsheet inputs. Paper-facing informative-prior APIs remain separate.
    """

    target: float
    epsilon_lower: float
    epsilon_upper: float
    maximum_patients: int
    cohort_size: int
    starting_dose: int
    scenarios: tuple[tuple[float, ...], ...]
    trials: int = 2000

    def __post_init__(self):
        self.design()
        for name in ("target", "epsilon_lower", "epsilon_upper"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name, low, high in (
            ("maximum_patients", 1, 200),
            ("cohort_size", 1, 200),
            ("starting_dose", 1, 100),
            ("trials", 1, 100000),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, low, high))
        try:
            p = np.asarray(self.scenarios, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("scenarios must be a rectangular probability matrix") from exc
        if (
            p.ndim != 2
            or not 1 <= len(p) <= 100
            or not 2 <= p.shape[1] <= 100
            or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
        ):
            raise ValueError("require 1..100 scenarios with 2..100 finite probabilities each")
        if self.starting_dose > p.shape[1]:
            raise ValueError("starting dose exceeds the scenario grid")
        object.__setattr__(self, "scenarios", tuple(tuple(float(v) for v in row) for row in p))

    def design(self):
        return _design(self.target, self.epsilon_lower, self.epsilon_upper)

    def to_json(self):
        return _serialize(self, "simulation")


def _serialize(study, mode):
    return json.dumps({"schema": _SCHEMA, "mode": mode, **asdict(study)}, allow_nan=False, indent=2)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate saved-input field")
        result[key] = value
    return result


def read_mtpi_desktop_project(text):
    """Read this package's versioned JSON or a captured-input HTML report."""
    if not isinstance(text, str) or len(text) > 10_000_000:
        raise ValueError("saved input must be a string of at most10 million characters")
    if _MARKER in text:
        if text.count(_MARKER) != 1:
            raise ValueError("duplicate saved-input envelope")
        text = text.split(_MARKER, 1)[1].split("</script>", 1)[0]
    try:
        value = json.loads(text, object_pairs_hook=_unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid mTPI saved JSON") from exc
    if not isinstance(value, dict) or value.pop("schema", None) != _SCHEMA:
        raise ValueError("unsupported mTPI saved-input schema")
    mode = value.pop("mode", None)
    constructor = {"table": MTPIDesktopTableStudy, "simulation": MTPIDesktopStudy}.get(mode)
    if constructor is None:
        raise ValueError("unsupported saved mTPI task")
    try:
        return constructor(**value)
    except TypeError as exc:
        raise ValueError("missing or unsupported saved mTPI fields") from exc


def _xml(archive, name):
    with archive.open(name) as stream:
        raw = stream.read(10_000_001)
    if (
        len(raw) > 10_000_000
        or b"<!doctype" in raw.lower()
        or b"<!entity" in raw.lower()
        or b"\x00" in raw
    ):
        raise ValueError("spreadsheet XML exceeds limits or declares entities")
    return ET.fromstring(raw)


def _column(index):
    result = ""
    while index:
        index, digit = divmod(index - 1, 26)
        result = chr(65 + digit) + result
    return result


def read_mtpi_desktop_workbook(data: bytes, *, sheet_name: str):
    """Import one native XLSM task using stored numeric inputs; never execute macros.

    Select a worksheet explicitly. Formulas in active input cells are rejected
    because their cached values could be stale. Unrelated cells, styles, VBA and
    cached simulation outputs are not imported or redistributed.
    """
    if not isinstance(data, bytes) or len(data) > 20_000_000:
        raise ValueError("workbook must be at most20 MB of bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 1000 or sum(v.file_size for v in entries) > 20_000_000:
                raise ValueError("uncompressed workbook exceeds20 MB or1000 members")
            if len({v.filename for v in entries}) != len(entries):
                raise ValueError("duplicate workbook members")
            workbook = _xml(archive, "xl/workbook.xml")
            sheets = workbook.findall("m:sheets/m:sheet", _NS)
            matches = [s for s in sheets if s.attrib.get("name") == sheet_name]
            if len(matches) != 1:
                raise ValueError("sheet_name must identify exactly one worksheet")
            rid = matches[0].attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            rels = _xml(archive, "xl/_rels/workbook.xml.rels")
            targets = [r for r in rels if r.attrib.get("Id") == rid]
            if len(targets) != 1 or targets[0].attrib.get("TargetMode") == "External":
                raise ValueError("invalid worksheet relationship")
            target = targets[0].attrib.get("Target", "")
            if ".." in target.split("/") or "\\" in target:
                raise ValueError("invalid worksheet target")
            name = target.lstrip("/") if target.startswith("/") else "xl/" + target
            sheet = _xml(archive, name)
            cells = {}
            for cell in sheet.findall("m:sheetData/m:row/m:c", _NS):
                address = cell.attrib.get("r")
                if address in cells:
                    raise ValueError("duplicate worksheet cells")
                cells[address] = cell

            def number(address):
                cell = cells.get(address)
                if (
                    cell is None
                    or cell.attrib.get("t", "n") != "n"
                    or cell.find("m:f", _NS) is not None
                ):
                    raise ValueError("active input must be a numeric non-formula cell: " + address)
                value = float(cell.findtext("m:v", default="nan", namespaces=_NS))
                if not math.isfinite(value):
                    raise ValueError("active input must be finite: " + address)
                return value

            def integer(address, low, high):
                value = number(address)
                if value != int(value):
                    raise ValueError("active integer input is fractional: " + address)
                return _integer(int(value), address, low, high)

            mode = integer("L1", 0, 1)
            if mode == 0:
                return MTPIDesktopTableStudy(
                    number("B1"), number("C1"), number("D1"), integer("A1", 1, 200)
                )
            doses = integer("C1", 2, 100)
            scenarios = integer("G1", 1, 100)
            truth = tuple(
                tuple(number(_column(j + 2) + str(i + 3)) for j in range(doses))
                for i in range(scenarios)
            )
            return MTPIDesktopStudy(
                number("B1"),
                number("H1"),
                number("I1"),
                integer("D1", 1, 200),
                integer("E1", 1, 200),
                integer("F1", 1, doses),
                truth,
                integer("A1", 1, 100000),
            )
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError("invalid mTPI workbook container/XML") from exc
