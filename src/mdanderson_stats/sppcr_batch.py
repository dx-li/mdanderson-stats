"""Strict SPPCR batch input parsing and canonical formatting."""

import re

from .cdflib_strings import QlexToken, qlex
from .sppcr_data import SPPCRData, sppcr_data


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class _Reader:
    def __init__(self, text: str):
        self.lines = iter(enumerate(text.splitlines(), 1))
        self.number = 0

    def line(self) -> list[QlexToken] | None:
        for self.number, raw in self.lines:
            value = re.split(r"[/#]", raw, maxsplit=1)[0].replace("\t", " ")
            tokens = list(qlex(value))
            if tokens:
                return tokens
        return None

    def error(self, message: str) -> ValueError:
        return ValueError(f"SPPCR batch line {self.number}: {message}")

    def record(self, name: str, size: int, *, dna: bool = False) -> list[float]:
        tokens = self.line()
        if tokens is None:
            raise self.error(f"expected {name}, reached end of input")
        if tokens[0].kind != "ID" or tokens[0].text.lower() != name:
            raise self.error(f"expected {name} record")
        tokens = tokens[1:]
        result = []
        if dna:
            if (
                not tokens
                or tokens[0].kind not in ("IN", "RL")
                or tokens[0].real is None
                or tokens[0].underflow
            ):
                raise self.error("run requires a representable DNA amount on the same line")
            token = tokens.pop(0)
            assert token.real is not None
            result.append(token.real)
        numbers: list[float] = []
        while len(numbers) < size:
            if not tokens:
                tokens = self.line()
                if tokens is None:
                    raise self.error(f"incomplete {name} values at end of input")
            for token in tokens:
                if token.kind != "IN" or token.integer is None or token.overflow:
                    raise self.error(f"{name} requires signed-int32 integer tokens")
                numbers.append(float(token.integer))
            if len(numbers) > size:
                raise self.error(f"extra values after {name}")
            tokens = []
        return result + numbers


def parse_sppcr_batch(
    text: str,
    *,
    unseen_alleles: str = "drop",
    max_alleles: int = 50,
    max_runs: int = 50,
    max_characters: int = 1_000_000,
) -> SPPCRData:
    """Parse ordered native batch records, continuations and / or # comments.

    Native array limits default to 50 runs/alleles and may be increased explicitly.
    Integer tokens must fit signed int32; scientific notation is allowed for DNA.
    Tabs are accepted as whitespace. Invalid data raises, never prints and resumes.
    """
    for value, name in [
        (max_alleles, "max_alleles"),
        (max_runs, "max_runs"),
        (max_characters, "max_characters"),
    ]:
        _positive(value, name)
    if not isinstance(text, str):
        raise TypeError("SPPCR batch input must be text")
    if len(text) > max_characters:
        raise ValueError("SPPCR batch input exceeds max_characters")
    reader = _Reader(text)
    alleles = int(reader.record("nallele", 1)[0])
    if not 1 <= alleles <= max_alleles:
        raise reader.error("nallele outside allowed range")
    runs = int(reader.record("nrun", 1)[0])
    if not 1 <= runs <= max_runs:
        raise reader.error("nrun outside allowed range")
    wells = reader.record("nwell", runs)
    sizes = reader.record("allelesizes", alleles)
    parents = reader.record("progenitor", 2)
    values = [reader.record("run", alleles, dna=True) for _ in range(runs)]
    if reader.line() is not None:
        raise reader.error("unexpected record after final run")
    return sppcr_data(
        [row[0] for row in values],
        [row[1:] for row in values],
        wells,
        sizes,
        (int(parents[0]), int(parents[1])),
        unseen_alleles=unseen_alleles,
    )


def format_sppcr_batch(data: SPPCRData) -> str:
    """Format retained alleles in canonical native records, with lines <=240 chars.

    Genome-equivalent DNA units are written, avoiding a second conversion on read.
    Output contains retained columns; dropped zero columns are not reconstructed.
    Integer values must fit signed int32 for the native parser.
    """
    d = sppcr_data(
        data.genome_dna,
        data.seen,
        data.wells,
        data.allele_sizes,
        data.progenitor_sizes,
        unseen_alleles="retain",
    )
    records: list[tuple[str, list[str]]] = [
        ("nallele", [str(len(d.allele_sizes))]),
        ("nrun", [str(len(d.dna))]),
        ("nwell", [str(int(n)) for n in d.wells]),
        ("allelesizes", [str(n) for n in d.allele_sizes]),
        ("progenitor", [str(n) for n in d.progenitor_sizes]),
    ]
    if any(x >= 2**31 for x in (*d.allele_sizes, *d.progenitor_sizes, *d.wells, *d.seen.flat)):
        raise ValueError("native batch integer values must fit signed int32")
    records.extend(
        ("run", [format(dna, ".17g"), *[str(int(x)) for x in seen]])
        for dna, seen in zip(d.genome_dna, d.seen, strict=True)
    )
    lines = []
    for name, values in records:
        line = name
        for value in values:
            if len(line) + 1 + len(value) > 240:
                lines.append(line)
                line = value
            else:
                line += " " + value
        lines.append(line)
    return "\n".join(lines) + "\n"
