"""MISCLIB format_specs pages as immutable Python message templates."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from .cdflib_console import CDFConsole


@dataclass(frozen=True)
class MisclibMessage:
    """One named page; substitution widths follow percent runs in reading order."""

    name: str
    template: str
    substitution_widths: tuple[int, ...]
    source_line: int

    def render(self, substitutions: Sequence[str] = ()) -> str:
        """Truncate/pad fixed-width substitutions and return text without final newline."""
        if isinstance(substitutions, str) or len(substitutions) != len(self.substitution_widths):
            raise ValueError("Supply exactly one string per substitution field")
        if any(not isinstance(s, str) for s in substitutions):
            raise ValueError("Substitutions must be strings")
        values = (
            s[:width].ljust(width)
            for s, width in zip(substitutions, self.substitution_widths, strict=True)
        )
        return self.template.format(*values)


def _page(name: str, lines: list[str], start: int) -> MisclibMessage:
    widths: list[int] = []
    rendered = []
    for line in lines:
        parts = []
        position = 0
        for match in re.finditer(r"%+", line):
            parts.append(line[position : match.start()].replace("{", "{{").replace("}", "}}"))
            parts.append("{" + str(len(widths)) + "}")
            widths.append(len(match[0]))
            position = match.end()
        parts.append(line[position:].replace("{", "{{").replace("}", "}}"))
        rendered.append("     " + "".join(parts) if line else "")
    return MisclibMessage(name, "\n".join(rendered), tuple(widths), start)


def compile_misclib_messages(source: str) -> tuple[MisclibMessage, ...]:
    """Compile >>BEGIN [name], >>CONTINUE and >>END into ordered message pages.

    Nonblank lines gain five spaces; trailing whitespace is removed. Percent
    runs define fixed-width fields. Names are lowercased; default name is message.
    This returns Python templates, replacing the original Fortran code generator.
    """
    if not isinstance(source, str) or len(source) > 2_000_000:
        raise ValueError("source must be a string of at most 2M characters")
    pages = []
    name: str | None = None
    lines: list[str] = []
    start = 0
    for number, line in enumerate(source.splitlines(), 1):
        if line.startswith(">>BEGIN"):
            if name is not None:
                raise ValueError(f"line {number}: nested BEGIN")
            match = re.match(r">>BEGIN\s*([A-Za-z][A-Za-z0-9_]*)", line)
            name = match[1].lower() if match else "message"
            lines, start = [], number
        elif line.startswith((">>END", ">>CONTINUE")):
            if name is None:
                raise ValueError(f"line {number}: page ending without BEGIN")
            if not lines:
                raise ValueError(f"line {number}: empty page")
            pages.append(_page(name, lines, start))
            lines, start = [], number
            if line.startswith(">>END"):
                name = None
        elif name is not None:
            lines.append(line.rstrip())
    if name is not None:
        raise ValueError("Unterminated message block")
    return tuple(pages)


def print_misclib_message(
    message: MisclibMessage,
    substitutions: Sequence[str] = (),
    *,
    console: CDFConsole | None = None,
    force: bool = False,
    unit: TextIO | None = None,
    unit_only: bool = False,
) -> str | None:
    """Render a page through CDFConsole's display/routing controls.

    The supplied console retains its template and substitutions; format_printed
    records this operation. Streams remain caller-owned. No global message state.
    """
    target = CDFConsole() if console is None else console
    target.format_printed = False
    text = message.render(substitutions)
    previous = target.message_format, target.substitutions
    try:
        target.message_format = "{0}"
        target.substitutions = (text,)
        return target.print_message_format(force=force, unit=unit, unit_only=unit_only)
    finally:
        target.message_format, target.substitutions = previous
