"""RANLIST list specifications and immutable per-stratum enrollment state."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count
from .ranlist_random import _DEFAULT
from .ranlist_restricted import ranlist_restricted
from .ranlist_unrestricted import ranlist_unrestricted


def _labels(values: tuple[str, ...], width: int, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if any(
        not isinstance(value, str)
        or not value.isascii()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or len(value) > width
        for value in result
    ):
        raise ValueError(
            f"{name} must contain printable ASCII strings of at most {width} characters"
        )
    return tuple(value.rstrip(" ") for value in result)


def _coordinates(
    patients: ArrayLike, strata: ArrayLike, number_strata: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    patients_array, strata_array = np.broadcast_arrays(
        count(patients, "patients"), count(strata, "strata")
    )
    patient = patients_array.astype(np.int64)
    stratum = strata_array.astype(np.int64)
    if np.any(patient < 1):
        raise ValueError("patients must be positive one-based numbers")
    if np.any((stratum < 1) | (stratum > number_strata)):
        raise ValueError("strata must be one-based indices within this list")
    return patient, stratum


@dataclass(frozen=True)
class RanlistAssignments:
    patients: NDArray[np.int64]
    strata: NDArray[np.int64]
    treatments: NDArray[np.int64]


@dataclass(frozen=True)
class RanlistSpecification:
    """Reusable list definition; strata correspond to RNG streams 1..20.

    ``weights`` are positive integer treatment counts when restricted, or
    positive relative frequencies otherwise. Labels follow the source's ASCII
    field widths. Blank names designate unnamed strata/treatments. The seed
    is authoritative; ``phrase`` is retained descriptive metadata.
    """

    weights: tuple[float, ...]
    restricted: bool = False
    balance: tuple[int, int] = (1, 1)
    seed: tuple[int, int] = _DEFAULT
    strata: tuple[str, ...] = ("",)
    treatments: tuple[str, ...] = ()
    title: tuple[str, ...] = ("Randomization list",)
    phrase: str = ""
    legacy: bool = False
    max_blocks: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.restricted, (bool, np.bool_)):
            raise ValueError("restricted must be boolean")
        # Validate the numerical contract even for an empty list, reusing the
        # allocation kernels rather than duplicating their rules.
        if self.restricted:
            result = ranlist_restricted(
                [],
                self.weights,
                balance=self.balance,
                seed=self.seed,
                legacy=self.legacy,
                max_blocks=self.max_blocks,
            )
            weights = tuple(map(float, result.counts))
            bounds = result.balance
        else:
            unrestricted = ranlist_unrestricted(
                [], self.weights, seed=self.seed, legacy=self.legacy
            )
            weights = tuple(map(float, unrestricted.weights))
            # Balance settings have no meaning for unrestricted allocation.
            bounds_array = count(self.balance, "balance")
            if bounds_array.shape != (2,) or not np.array_equal(bounds_array, [1, 1]):
                raise ValueError("unrestricted lists use balance=(1, 1)")
            bounds = (1, 1)
            if (
                isinstance(self.max_blocks, (bool, np.bool_))
                or not isinstance(self.max_blocks, (int, np.integer))
                or self.max_blocks < 1
            ):
                raise ValueError("max_blocks must be a positive integer")
        strata = _labels(self.strata, 30, "strata")
        treatments = _labels(self.treatments, 30, "treatments") or ("",) * len(weights)
        title = _labels(self.title, 80, "title")
        phrase = _labels((self.phrase,), 31, "phrase")[0]
        if not 1 <= len(strata) <= 20:
            raise ValueError("there must be 1..20 strata")
        if len(treatments) != len(weights):
            raise ValueError("treatment names must match the number of weights")
        for name, labels in [("strata", strata), ("treatments", treatments)]:
            if any(labels) and not all(labels):
                raise ValueError(f"{name} must either all be named or all be blank")
        if not 1 <= len(title) <= 9 or not all(title):
            raise ValueError("title must contain 1..9 nonblank lines")
        for name, value in {
            "weights": weights,
            "balance": bounds,
            "seed": tuple(map(int, self.seed)),
            "strata": strata,
            "treatments": treatments,
            "title": title,
            "phrase": phrase,
            "restricted": bool(self.restricted),
            "legacy": bool(self.legacy),
            "max_blocks": int(self.max_blocks),
        }.items():
            object.__setattr__(self, name, value)

    def allocate(self, patients: ArrayLike, *, strata: ArrayLike = 1) -> RanlistAssignments:
        """Query any patient positions without advancing enrollment counters."""
        patient, stratum = _coordinates(patients, strata, len(self.strata))
        assigned = np.empty(patient.shape, dtype=np.int64)
        for stream in np.unique(stratum):
            mask = stratum == stream
            if self.restricted:
                values = ranlist_restricted(
                    patient[mask],
                    self.weights,
                    balance=self.balance,
                    seed=self.seed,
                    stream=int(stream),
                    legacy=self.legacy,
                    max_blocks=self.max_blocks,
                ).treatments
            else:
                values = ranlist_unrestricted(
                    patient[mask],
                    self.weights,
                    seed=self.seed,
                    stream=int(stream),
                    legacy=self.legacy,
                ).treatments
            assigned[mask] = values
        for value in (patient, stratum, assigned):
            value.flags.writeable = False
        return RanlistAssignments(patient, stratum, assigned)


@dataclass(frozen=True)
class RanlistSession:
    """Enrollment counts with transactional, immutable batch updates.

    Retain the returned session after enrollment. Failed batches leave the
    original session unchanged. This is an in-process workflow, not concurrent
    enrollment storage or a database transaction.
    """

    specification: RanlistSpecification
    current_patients: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.specification, RanlistSpecification):
            raise ValueError("specification must be a RanlistSpecification")
        counters = count(self.current_patients, "current_patients")
        if counters.shape == (0,):
            counters = np.zeros(len(self.specification.strata))
        if counters.shape != (len(self.specification.strata),):
            raise ValueError("current_patients must contain one count per stratum")
        object.__setattr__(self, "current_patients", tuple(map(int, counters)))

    def enroll(self, strata: ArrayLike = 1) -> tuple["RanlistSession", RanlistAssignments]:
        """Assign successive patients within each stratum in C-order arrival order."""
        stratum = count(strata, "strata").astype(np.int64)
        if np.any((stratum < 1) | (stratum > len(self.current_patients))):
            raise ValueError("strata must be one-based indices within this list")
        patients = np.empty(stratum.shape, dtype=np.int64)
        updated = list(self.current_patients)
        for stream in np.unique(stratum):
            mask = stratum == stream
            index = int(stream) - 1
            amount = int(np.count_nonzero(mask))
            if updated[index] + amount >= 2**53:
                raise ValueError("enrollment would exceed the patient number limit")
            patients[mask] = np.arange(updated[index] + 1, updated[index] + amount + 1)
            updated[index] += amount
        assignments = self.specification.allocate(patients, strata=stratum)
        return RanlistSession(self.specification, tuple(updated)), assignments

    def inquire(self, patients: ArrayLike, *, strata: ArrayLike = 1) -> RanlistAssignments:
        """Retrieve only already-enrolled subjects, without changing counters."""
        patient, stratum = _coordinates(patients, strata, len(self.current_patients))
        if np.any(patient > np.asarray(self.current_patients)[stratum - 1]):
            raise ValueError("a requested patient has not been randomized yet")
        return self.specification.allocate(patient, strata=stratum)
