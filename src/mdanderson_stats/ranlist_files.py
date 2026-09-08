"""Validated session persistence and archived RANLIST parameter records."""

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ranlist_session import RanlistSession, RanlistSpecification

_HEADER = "Parameter file for RANLST"


def ranlist_parameter_text(session: RanlistSession, *, allow_rounding: bool = False) -> str:
    """Export legacy sessions using the archived fixed-width record layout.

    The format cannot store modern algorithm choices. Weights use source
    float32 F6.3 rounding; changes to effective weights require allow_rounding.
    Counts above nine and counters above 999999 cannot be represented.
    """
    if not isinstance(session, RanlistSession):
        raise ValueError("session must be a RanlistSession")
    if not isinstance(allow_rounding, bool):
        raise ValueError("allow_rounding must be boolean")
    spec = session.specification
    if not spec.legacy:
        raise ValueError(
            "original parameter files require legacy=True; use JSON for modern sessions"
        )
    lines = [_HEADER.ljust(25), str(len(spec.title)), *(line.ljust(80) for line in spec.title)]
    lines.append(
        f"{spec.phrase:31}{spec.seed[0]:10d}{spec.seed[1]:10d}"
        f"{len(spec.strata):2d}{len(spec.weights):2d}{'T' if spec.restricted else 'F'}"
    )
    for names in (spec.strata, spec.treatments):
        lines.extend(name.ljust(30) for name in (names if any(names) else ("",)))
    if spec.restricted:
        if any(weight > 9 for weight in spec.weights):
            raise ValueError("original I1 treatment counts cannot exceed nine")
        lines.extend(str(int(weight)) for weight in spec.weights)
    else:
        for weight in spec.weights:
            source_weight = np.float32(weight)
            field = f"{float(source_weight):6.3f}"
            if len(field) > 6 or float(field) <= 0:
                raise ValueError("weights must remain positive and fit the original F6.3 field")
            if not allow_rounding and np.float32(float(field)) != source_weight:
                raise ValueError(
                    "F6.3 changes a weight; set allow_rounding=True to export deliberately"
                )
            lines.append(field)
    lines.append(f"{spec.balance[0]:3d}{spec.balance[1]:3d}")
    if any(value > 999999 for value in session.current_patients):
        raise ValueError("original I6 patient counters cannot exceed 999999")
    lines.extend(f"{value:6d}" for value in session.current_patients)
    return "\n".join(lines) + "\n"


def read_ranlist_parameters(
    text: str, *, max_blocks: int = 10_000, repair_unrestricted_balance: bool = False
) -> RanlistSession:
    """Parse canonical original records into a source-compatible session.

    Trailing padding and CRLF are accepted. Malformed, truncated, overflowing
    or additional nonblank records are rejected. Unrestricted balance fields
    are read but normalized to (1,1), since MKLST did not initialize them.
    Explicit repair_unrestricted_balance permits ignoring corrupt unused balance
    fields from MKLST. Restricted balance fields are always validated.
    The file's explicit seed pair is authoritative, irrespective of its phrase.
    """
    if not isinstance(repair_unrestricted_balance, bool):
        raise ValueError("repair_unrestricted_balance must be boolean")
    if not isinstance(text, str) or not text.isascii():
        raise ValueError("parameter text must be ASCII")
    if any(ord(char) < 32 and char not in "\r\n" or ord(char) == 127 for char in text):
        raise ValueError("parameter records must contain printable ASCII")
    records = iter(text.splitlines())

    def record(width: int, name: str) -> str:
        try:
            line = next(records)
        except StopIteration as exc:
            raise ValueError(f"missing parameter record: {name}") from exc
        if line[width:].strip():
            raise ValueError(f"extra data in parameter record: {name}")
        return line[:width].ljust(width)

    def integer(field: str, name: str) -> int:
        if not re.fullmatch(r" *[+-]?\d+ *", field):
            raise ValueError(f"invalid integer field: {name}")
        return int(field)

    if record(25, "header").rstrip() != _HEADER:
        raise ValueError("not a RANLST parameter file")
    title_count = integer(record(1, "title count"), "title count")
    if not 1 <= title_count <= 9:
        raise ValueError("title count must be 1..9")
    title = tuple(record(80, "title").rstrip() for _ in range(title_count))
    settings = record(56, "settings")
    phrase = settings[:31].rstrip()
    seed = (integer(settings[31:41], "seed 1"), integer(settings[41:51], "seed 2"))
    nstrata = integer(settings[51:53], "strata count")
    ntreat = integer(settings[53:55], "treatment count")
    if not 1 <= nstrata <= 20 or not 1 <= ntreat <= 20:
        raise ValueError("strata and treatment counts must be 1..20")
    if settings[55].upper() not in ("T", "F"):
        raise ValueError("restriction flag must be T or F")
    restricted = settings[55].upper() == "T"

    def names(size: int, name: str) -> tuple[str, ...]:
        first = record(30, name).rstrip()
        return (
            (first, *(record(30, name).rstrip() for _ in range(size - 1)))
            if first
            else ("",) * size
        )

    strata, treatments = names(nstrata, "stratum name"), names(ntreat, "treatment name")
    weights = []
    for _ in range(ntreat):
        if restricted:
            weights.append(float(integer(record(1, "count"), "count")))
        else:
            field = record(6, "weight").strip()
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", field):
                raise ValueError("invalid F6.3 weight")
            value = float(field) if "." in field else int(field) / 1000
            weights.append(float(np.float32(value)))
    balance_record = record(6, "balance")
    balance = (
        (1, 1)
        if not restricted and repair_unrestricted_balance
        else (
            integer(balance_record[:3], "minimum balance"),
            integer(balance_record[3:], "maximum balance"),
        )
    )
    counters = tuple(
        integer(record(6, "patient counter"), "patient counter") for _ in range(nstrata)
    )
    if any(line.strip() for line in records):
        raise ValueError("unexpected trailing parameter records")
    specification = RanlistSpecification(
        tuple(weights),
        restricted=restricted,
        balance=balance if restricted else (1, 1),
        seed=seed,
        strata=strata,
        treatments=treatments,
        title=title,
        phrase=phrase,
        legacy=True,
        max_blocks=max_blocks,
    )
    return RanlistSession(specification, counters)


def save_ranlist_session(session: RanlistSession, path: str | Path) -> None:
    """Atomically replace a versioned JSON snapshot after full serialization.

    Retains numerical precision and modern/legacy settings. Does not coordinate
    concurrent writers. The destination's parent directory must already exist.
    """
    if not isinstance(session, RanlistSession):
        raise ValueError("session must be a RanlistSession")
    payload = {"format": "mdanderson-stats/ranlist", "version": 1, **asdict(session)}
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_ranlist_session(path: str | Path) -> RanlistSession:
    """Load and validate a version-one JSON snapshot; never infer missing fields."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(payload, dict) or set(payload) != {
        "format",
        "version",
        "specification",
        "current_patients",
    }:
        raise ValueError("invalid RANLIST snapshot fields")
    if (
        payload["format"] != "mdanderson-stats/ranlist"
        or type(payload["version"]) is not int
        or payload["version"] != 1
    ):
        raise ValueError("unsupported RANLIST snapshot format or version")
    specification = payload["specification"]
    if not isinstance(specification, dict) or set(specification) != set(
        RanlistSpecification.__dataclass_fields__
    ):
        raise ValueError("invalid or missing specification fields")
    for name, kinds in [
        ("weights", (int, float)),
        ("balance", (int,)),
        ("seed", (int,)),
        ("strata", (str,)),
        ("treatments", (str,)),
        ("title", (str,)),
    ]:
        values = specification[name]
        if not isinstance(values, list) or any(type(value) not in kinds for value in values):
            raise ValueError(f"invalid snapshot array: {name}")
    if len(specification["treatments"]) != len(specification["weights"]):
        raise ValueError("snapshot must explicitly contain every treatment label")
    try:
        spec = RanlistSpecification(**specification)
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid specification field types") from exc
    counters = payload["current_patients"]
    if (
        not isinstance(counters, list)
        or len(counters) != len(spec.strata)
        or any(type(value) is not int for value in counters)
    ):
        raise ValueError("snapshot must contain one integer counter per stratum")
    return RanlistSession(spec, tuple(counters))
