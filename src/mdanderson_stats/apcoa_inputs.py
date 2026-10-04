"""Explicit, ID-aligned input preparation for covariate-adjusted PCoA."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .apcoa import AdjustedPCoA, adjusted_pcoa
from .boin import _owned

_MAX_SAMPLES = 2000
_MAX_METADATA_COLUMNS = 256
_MAX_DISTANCE_FILE_BYTES = 128 * 1024 * 1024
_MAX_METADATA_FILE_BYTES = 16 * 1024 * 1024
_DELIMITERS = {",", "\t"}


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must contain nonempty text identifiers of at most 256 characters")
    return value


def _unique_ids(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or not 2 <= values.size <= _MAX_SAMPLES:
            raise ValueError(f"{name} must contain 2..{_MAX_SAMPLES} identifiers")
        raw = values.tolist()
    elif isinstance(values, (list, tuple)):
        if not 2 <= len(values) <= _MAX_SAMPLES:
            raise ValueError(f"{name} must contain 2..{_MAX_SAMPLES} identifiers")
        raw = values
    else:
        raise TypeError(f"{name} must be a bounded list, tuple, or one-dimensional array")
    ids = tuple(_identifier(value, name) for value in raw)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must be unique")
    return ids


def _distance_array(value: ArrayLike, n: int) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape != (n, n) or np.iscomplexobj(value):
            raise ValueError("distances must be a real n-by-n matrix matching the sample IDs")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("distances must contain real numeric values") from error
    elif isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError("distances must be an n-by-n matrix matching the sample IDs")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != n:
                raise ValueError("distances must be a flat n-by-n matrix")
            if isinstance(row, np.ndarray) and (row.ndim != 1 or np.iscomplexobj(row)):
                raise ValueError("distances must be a flat real n-by-n matrix")
            if any(
                isinstance(cell, (list, tuple, np.ndarray, complex, np.complexfloating))
                for cell in row
            ):
                raise ValueError("distances must contain real scalar values")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("distances must contain real numeric values") from error
    else:
        raise TypeError("distances must be a bounded n-by-n array or nested list/tuple")
    if not np.isfinite(matrix).all():
        raise ValueError("distances must contain only finite values")
    return _owned(matrix)


def _metadata_column(value: object, n: int, name: str) -> tuple[str | int | float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != n or np.iscomplexobj(value):
            raise ValueError(f"metadata column {name!r} must be a real vector of length {n}")
        raw = value.tolist()
    elif isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"metadata column {name!r} must have length {n}")
        raw = value
    else:
        raise TypeError(f"metadata column {name!r} must be a bounded vector")
    normalized: list[str | int | float] = []
    for item in raw:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(f"metadata column {name!r} must contain real numeric or text scalars")
        normalized.append(item)
    return tuple(normalized)


def _delimiter(value: str) -> str:
    if value not in _DELIMITERS:
        raise ValueError("delimiter must be comma or tab")
    return value


def _check_file(path: str | Path, maximum: int, kind: str) -> Path:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise OSError(f"cannot inspect {kind} file {source}") from error
    if size > maximum:
        raise ValueError(f"{kind} file exceeds the {maximum}-byte input limit")
    return source


@dataclass(frozen=True)
class APCoADistanceTable:
    """A numeric distance matrix in canonical sample-ID order."""

    sample_ids: tuple[str, ...]
    distances: FloatArray


@dataclass(frozen=True)
class APCoAMetadataTable:
    """Metadata columns keyed by unique names; CSV fields are kept as text."""

    sample_ids: tuple[str, ...]
    columns: Mapping[str, tuple[str | int | float, ...]]


@dataclass(frozen=True)
class CategoricalEncoding:
    """Explicit treatment coding for one categorical nuisance covariate."""

    levels: tuple[str, ...]
    reference: str


@dataclass(frozen=True)
class CovariateEncoding:
    """Provenance for one selected numeric or categorical metadata column."""

    name: str
    kind: Literal["numeric", "categorical"]
    columns: tuple[str, ...]
    levels: tuple[str, ...] = ()
    reference: str | None = None


@dataclass(frozen=True)
class PreparedAPCoAInput:
    """Immutable aligned inputs and explicit covariate-encoding provenance."""

    sample_ids: tuple[str, ...]
    distances: FloatArray
    covariates: FloatArray
    covariate_columns: tuple[str, ...]
    encodings: tuple[CovariateEncoding, ...]
    group_labels: tuple[str, ...] | None

    def fit(
        self,
        *,
        components: int | None = 2,
        intercept: bool = False,
        eigenvalue_tolerance: float = 1e-12,
    ) -> AdjustedPCoA:
        """Run the existing numeric core with this prepared input."""
        return adjusted_pcoa(
            self.distances,
            self.covariates,
            components=components,
            intercept=intercept,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )


def read_apcoa_distance_csv(path: str | Path, *, delimiter: str = ",") -> APCoADistanceTable:
    """Read a bounded square CSV/TSV distance table with row and column IDs.

    The first header cell is empty, followed by column sample IDs. Each data
    row begins with its row sample ID and then has one numeric distance per
    column. Rows are reordered to header order; no ID normalization occurs.
    """
    sep = _delimiter(delimiter)
    source = _check_file(path, _MAX_DISTANCE_FILE_BYTES, "distance")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=sep, strict=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("distance file is empty") from error
        if not 3 <= len(header) <= _MAX_SAMPLES + 1 or header[0] != "":
            raise ValueError(
                f"distance header must start empty and contain 2..{_MAX_SAMPLES} sample IDs"
            )
        column_ids = _unique_ids(header[1:], "distance column IDs")
        n = len(column_ids)
        matrix = np.empty((n, n), dtype=np.float64)
        row_ids: list[str] = []
        for index, row in enumerate(reader):
            if index >= n:
                raise ValueError("distance file has more rows than column IDs")
            if len(row) != n + 1:
                raise ValueError("each distance row must contain one ID and n numeric values")
            row_ids.append(_identifier(row[0], "distance row IDs"))
            try:
                matrix[index] = [float(item) for item in row[1:]]
            except ValueError as error:
                raise ValueError(f"distance row {index + 1} contains a nonnumeric value") from error
            if not np.isfinite(matrix[index]).all():
                raise ValueError(f"distance row {index + 1} contains a nonfinite value")
        ids = _unique_ids(row_ids, "distance row IDs")
        if set(ids) != set(column_ids):
            raise ValueError("distance row and column IDs must match exactly")
        row_position = {sample_id: index for index, sample_id in enumerate(ids)}
        aligned = matrix[[row_position[sample_id] for sample_id in column_ids]]
        aligned.setflags(write=False)
    return APCoADistanceTable(column_ids, aligned)


def read_apcoa_metadata_csv(path: str | Path, *, delimiter: str = ",") -> APCoAMetadataTable:
    """Read a bounded metadata CSV/TSV whose first column contains sample IDs."""
    sep = _delimiter(delimiter)
    source = _check_file(path, _MAX_METADATA_FILE_BYTES, "metadata")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=sep, strict=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("metadata file is empty") from error
        if not 2 <= len(header) <= _MAX_METADATA_COLUMNS + 1:
            raise ValueError(f"metadata must contain 1..{_MAX_METADATA_COLUMNS} columns plus IDs")
        if any(not name or not name.strip() or len(name) > 256 for name in header) or len(
            set(header)
        ) != len(header):
            raise ValueError("metadata column names must be nonempty and unique")
        ids: list[str] = []
        values: list[list[str]] = [[] for _ in header[1:]]
        for index, row in enumerate(reader):
            if index >= _MAX_SAMPLES:
                raise ValueError(f"metadata may contain at most {_MAX_SAMPLES} samples")
            if len(row) != len(header):
                raise ValueError(f"metadata row {index + 1} has the wrong number of columns")
            ids.append(_identifier(row[0], "metadata sample IDs"))
            if any(value == "" for value in row[1:]):
                raise ValueError(f"metadata row {index + 1} contains a blank value")
            for column, value in zip(values, row[1:], strict=True):
                column.append(value)
    sample_ids = _unique_ids(ids, "metadata sample IDs")
    columns = MappingProxyType(
        {name: tuple(column) for name, column in zip(header[1:], values, strict=True)}
    )
    return APCoAMetadataTable(sample_ids, columns)


def prepare_apcoa_input(
    distance_table: APCoADistanceTable,
    metadata: APCoAMetadataTable,
    *,
    numeric_covariates: Sequence[str] = (),
    categorical_covariates: Mapping[str, CategoricalEncoding] | None = None,
    main_group: str | None = None,
) -> PreparedAPCoAInput:
    """Align labeled data and encode only explicitly selected nuisance columns.

    Categorical columns use treatment coding in the caller-specified level order,
    omitting the explicit reference. Numeric columns retain their supplied scale.
    Main-group labels are returned separately and are never automatically fitted.
    """
    if not isinstance(distance_table, APCoADistanceTable) or not isinstance(
        metadata, APCoAMetadataTable
    ):
        raise TypeError("distance_table and metadata must use the aPCoA table types")
    ids = _unique_ids(distance_table.sample_ids, "distance sample IDs")
    n = len(ids)
    distances = _distance_array(distance_table.distances, n)
    metadata_ids = _unique_ids(metadata.sample_ids, "metadata sample IDs")
    if set(ids) != set(metadata_ids):
        raise ValueError("distance and metadata sample IDs must match exactly")
    if len(metadata.columns) > _MAX_METADATA_COLUMNS:
        raise ValueError(f"metadata may contain at most {_MAX_METADATA_COLUMNS} columns")
    position = {sample_id: index for index, sample_id in enumerate(metadata_ids)}
    order = tuple(position[sample_id] for sample_id in ids)

    if not isinstance(numeric_covariates, (list, tuple)) or len(numeric_covariates) > n:
        raise ValueError("numeric_covariates must be a bounded list or tuple of column names")
    numeric_names = tuple(numeric_covariates)
    if any(not isinstance(name, str) or not name or len(name) > 256 for name in numeric_names):
        raise ValueError(
            "numeric covariate names must be nonempty strings of at most 256 characters"
        )
    if len(set(numeric_names)) != len(numeric_names):
        raise ValueError("numeric covariates must not repeat a column")
    if categorical_covariates is not None and (
        not isinstance(categorical_covariates, Mapping)
        or len(categorical_covariates) > _MAX_METADATA_COLUMNS
    ):
        raise ValueError(
            f"categorical_covariates must be a mapping of at most {_MAX_METADATA_COLUMNS} columns"
        )
    categorical_specs = dict(categorical_covariates or {})
    if any(not isinstance(name, str) or not name or len(name) > 256 for name in categorical_specs):
        raise ValueError(
            "categorical covariate names must be nonempty strings of at most 256 characters"
        )
    if set(numeric_names).intersection(categorical_specs):
        raise ValueError("a metadata column cannot be both numeric and categorical")
    if main_group is not None and (
        not isinstance(main_group, str) or not main_group or len(main_group) > 256
    ):
        raise ValueError("main_group must be a nonempty metadata column name or None")
    selected = (*numeric_names, *categorical_specs)
    if len(selected) > _MAX_METADATA_COLUMNS or len(set(selected)) != len(selected):
        raise ValueError("selected nuisance covariate columns must be unique and bounded")
    required = set(selected) | ({main_group} if main_group is not None else set())
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"metadata is missing selected columns: {sorted(missing_columns)}")

    if len(numeric_names) > n:
        raise ValueError("encoded covariate count cannot exceed sample count")
    total_encoded = len(numeric_names)
    for name, spec in categorical_specs.items():
        if not isinstance(spec, CategoricalEncoding):
            raise TypeError(f"categorical covariate {name!r} needs CategoricalEncoding")
        if not isinstance(spec.levels, tuple) or not 2 <= len(spec.levels) <= n + 1:
            raise ValueError(f"categorical covariate {name!r} needs 2..n+1 explicit levels")
        if not isinstance(spec.reference, str) or spec.reference not in spec.levels:
            raise ValueError(
                f"reference for categorical covariate {name!r} must be one of its levels"
            )
        total_encoded += len(spec.levels) - 1
    if total_encoded > n:
        raise ValueError("encoded covariate count cannot exceed sample count")

    encoded_columns: list[FloatArray] = []
    column_names: list[str] = []
    encodings: list[CovariateEncoding] = []
    for name in numeric_names:
        raw = _metadata_column(metadata.columns[name], n, name)
        try:
            values = np.asarray([float(raw[index]) for index in order], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"numeric covariate {name!r} must contain finite numbers") from error
        if not np.isfinite(values).all():
            raise ValueError(f"numeric covariate {name!r} must contain finite numbers")
        encoded_columns.append(values)
        column_names.append(name)
        encodings.append(CovariateEncoding(name, "numeric", (name,)))

    for name, spec in categorical_specs.items():
        if not isinstance(spec, CategoricalEncoding):
            raise TypeError(f"categorical covariate {name!r} needs CategoricalEncoding")
        levels = tuple(_identifier(level, f"levels for {name!r}") for level in spec.levels)
        if len(levels) < 2 or len(set(levels)) != len(levels):
            raise ValueError(f"categorical covariate {name!r} requires at least two unique levels")
        if not isinstance(spec.reference, str) or spec.reference not in levels:
            raise ValueError(
                f"reference for categorical covariate {name!r} must be one of its levels"
            )
        raw = _metadata_column(metadata.columns[name], n, name)
        category_values = tuple(raw[index] for index in order)
        if any(not isinstance(value, str) for value in category_values):
            raise ValueError(f"categorical covariate {name!r} must contain text values")
        unknown = set(category_values) - set(levels)
        if unknown:
            raise ValueError(
                f"categorical covariate {name!r} contains undeclared levels: {sorted(unknown)}"
            )
        generated = tuple(f"{name}[{level}]" for level in levels if level != spec.reference)
        if any(len(column) > 256 for column in generated):
            raise ValueError("encoded covariate names must not exceed 256 characters")
        if set(generated).intersection(column_names):
            raise ValueError("encoded covariate column names must be unique")
        for level in levels:
            if level != spec.reference:
                encoded_columns.append(
                    np.fromiter(
                        (value == level for value in category_values),
                        dtype=np.float64,
                        count=n,
                    )
                )
                column_names.append(f"{name}[{level}]")
        encodings.append(CovariateEncoding(name, "categorical", generated, levels, spec.reference))

    if len(column_names) > n:
        raise ValueError("encoded covariate count cannot exceed sample count")
    covariates = (
        _owned(np.column_stack(encoded_columns))
        if encoded_columns
        else _owned(np.empty((n, 0), dtype=np.float64))
    )
    labels = None
    if main_group is not None:
        raw_group = _metadata_column(metadata.columns[main_group], n, main_group)
        group_list: list[str] = []
        for index in order:
            label = raw_group[index]
            if not isinstance(label, str):
                raise ValueError("main-group labels must be text")
            group_list.append(label)
        aligned_group = tuple(group_list)
        if any(not label or not label.strip() for label in aligned_group):
            raise ValueError("main-group labels must be nonempty")
        labels = aligned_group
    return PreparedAPCoAInput(
        ids,
        distances,
        covariates,
        tuple(column_names),
        tuple(encodings),
        labels,
    )
