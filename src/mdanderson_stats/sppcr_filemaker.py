"""SPPCR FileMaker-style numeric-row parsing and canonical CSV output."""

import csv
import re

from .cdflib_strings import QlexToken, qlex
from .sppcr_batch import _positive
from .sppcr_data import SPPCRData, sppcr_data


def _tokens(line: str, number: int) -> list[QlexToken]:
    if "," in line:
        try:
            fields = next(csv.reader([line], skipinitialspace=True, strict=True))
        except csv.Error as exc:
            raise ValueError(f"SPPCR FileMaker line {number}: malformed CSV") from exc
        tokens = []
        for field in fields:
            values = list(qlex(field.strip()))
            if len(values) != 1:
                raise ValueError(f"SPPCR FileMaker line {number}: missing or compound CSV field")
            tokens.extend(values)
    else:
        tokens = list(qlex(line.replace("\t", " ")))
    result = []
    for token in tokens:
        if token.kind == "QS":
            inner = list(qlex(token.text))
            if len(inner) != 1 or inner[0].kind not in ("IN", "RL"):
                raise ValueError(f"SPPCR FileMaker line {number}: quoted field must be numeric")
            token = inner[0]
        result.append(token)
    return result


def parse_sppcr_filemaker(
    text: str,
    *,
    unseen_alleles: str = "drop",
    max_alleles: int = 25,
    max_runs: int = 50,
    max_characters: int = 1_000_000,
) -> SPPCRData:
    """Parse rows: parent1 parent2 wells genome_DNA (allele_size seen)...

    Accept whitespace-separated or numeric CSV rows, with / or # comments.
    Parent identities and allele order must agree across rows. Empty/comment rows
    are ignored. Missing CSV fields and malformed numbers raise, never shift data.
    Defaults match the source's 54-value row and 50-run limits.
    """
    for value, name in [
        (max_alleles, "max_alleles"),
        (max_runs, "max_runs"),
        (max_characters, "max_characters"),
    ]:
        _positive(value, name)
    if not isinstance(text, str):
        raise TypeError("SPPCR FileMaker input must be text")
    if len(text) > max_characters:
        raise ValueError("SPPCR FileMaker input exceeds max_characters")
    rows: list[list[float]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"[/#]", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        if len(rows) >= max_runs:
            raise ValueError(f"SPPCR FileMaker line {number}: exceeds max_runs")
        tokens = _tokens(line, number)
        if len(tokens) < 6 or (len(tokens) - 4) % 2 or (len(tokens) - 4) // 2 > max_alleles:
            raise ValueError(f"SPPCR FileMaker line {number}: invalid number of allele/count pairs")
        values = []
        for i, token in enumerate(tokens):
            if i == 3:
                if token.kind not in ("IN", "RL") or token.real is None or token.underflow:
                    raise ValueError(f"SPPCR FileMaker line {number}: invalid DNA amount")
                values.append(token.real)
            else:
                if token.kind != "IN" or token.integer is None or token.overflow:
                    raise ValueError(
                        f"SPPCR FileMaker line {number}: expected signed-int32 integer"
                    )
                values.append(float(token.integer))
        if rows and (values[:2] != rows[0][:2] or values[4::2] != rows[0][4::2]):
            raise ValueError(f"SPPCR FileMaker line {number}: progenitors or allele order changed")
        rows.append(values)
    if not rows:
        raise ValueError("SPPCR FileMaker input contains no data rows")
    first = rows[0]
    return sppcr_data(
        [r[3] for r in rows],
        [r[5::2] for r in rows],
        [r[2] for r in rows],
        first[4::2],
        (int(first[0]), int(first[1])),
        unseen_alleles=unseen_alleles,
    )


def format_sppcr_filemaker(data: SPPCRData) -> str:
    """Write quoted numeric CSV rows within native 25-allele/50-run/500-char limits.

    Retained columns are written in their existing order using genome DNA units.
    No header is emitted. Missing columns, comments and source layout are not rebuilt.
    """
    d = sppcr_data(
        data.genome_dna,
        data.seen,
        data.wells,
        data.allele_sizes,
        data.progenitor_sizes,
        unseen_alleles="retain",
    )
    if len(d.allele_sizes) > 25 or len(d.dna) > 50:
        raise ValueError("native FileMaker limits are 25 alleles and 50 runs")
    if any(x >= 2**31 for x in (*d.allele_sizes, *d.progenitor_sizes, *d.wells, *d.seen.flat)):
        raise ValueError("native FileMaker integers must fit signed int32")
    lines = []
    for dna, n, seen in zip(d.genome_dna, d.wells, d.seen, strict=True):
        fields = [
            str(d.progenitor_sizes[0]),
            str(d.progenitor_sizes[1]),
            str(int(n)),
            format(dna, ".17g"),
        ]
        for size, value in zip(d.allele_sizes, seen, strict=True):
            fields.extend([str(size), str(int(value))])
        line = ",".join('"' + x + '"' for x in fields)
        if len(line) > 500:
            raise ValueError("native FileMaker row exceeds 500 characters")
        lines.append(line)
    return "\n".join(lines) + "\n"
