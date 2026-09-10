"""Alphabetize explicit Fortran program units using SORTF90's layout."""

import re
from dataclasses import dataclass, field

_NAME = r"[a-z][a-z0-9_]*"
_BEGIN = re.compile(rf"\b(program|module|subroutine|function)\s+({_NAME})\b", re.I)
_END = re.compile(rf"end\s+(program|module|subroutine|function)(?:\s+({_NAME}))?\s*", re.I)
_STRINGS = re.compile(r"'([^']|'')*'|\"([^\"]|\"\")*\"")


@dataclass
class _Unit:
    kind: str
    name: str
    header: list[str] = field(default_factory=list)
    children: list["_Unit"] = field(default_factory=list)
    contains: bool = False
    end: str = ""


def _render(units: list[_Unit], level: int, separators: tuple[str, str]) -> str:
    chunks = []
    for unit in sorted(units, key=lambda item: item.name):
        if level:
            chunks.append(separators[min(level, 2) - 1])
        chunks.extend(unit.header)
        if unit.contains:
            chunks.append(_render(unit.children, level + 1, separators))
            chunks.append(separators[min(level + 1, 2) - 1])
        chunks.append(unit.end)
    return "".join(chunks)


def sortf90(source: str, *, separators: tuple[str, str] | None = None) -> str:
    """Sort named units and contained procedures, returning source without file I/O.

    Supports the original explicit PROGRAM/MODULE/SUBROUTINE/FUNCTION subset;
    each opening/closing statement occupies one line. Interfaces remain opaque.
    Like SORTF90, comments outside unit bodies are discarded. This is not a
    general Fortran parser; see docs/sortf90.md before using it on source files.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if separators is None:
        separators = ("\n!" + "*" * 69 + "\n\n", "\n!" + "." * 69 + "\n\n")
    if len(separators) != 2 or any(not isinstance(s, str) for s in separators):
        raise ValueError("separators must contain two strings")
    roots: list[_Unit] = []
    stack: list[_Unit] = []
    interface = False
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.rstrip(" ") + "\n"
        code = _STRINGS.sub("''", raw).split("!", 1)[0].strip().lower()
        error = f"line {number}: "
        if interface:
            stack[-1].header.append(line)
            if re.fullmatch(r"end\s*interface(?:\s+.*)?", code):
                interface = False
            continue
        if re.match(r"(?:abstract\s+)?interface\b", code):
            if not stack or stack[-1].contains:
                raise ValueError(error + "interface must be inside a unit body")
            interface = True
            stack[-1].header.append(line)
            continue
        if re.match(r"(?:submodule\b|type\s*(?:::|,)|type\s+" + _NAME + r"\s*$)", code):
            raise ValueError(error + "derived-type definitions and submodules are unsupported")
        if code.startswith("#"):
            raise ValueError(error + "preprocess source before sorting")
        begin = _BEGIN.search(code)
        end = _END.fullmatch(code)
        contains = code == "contains"
        if (begin or code.startswith("end") or code.startswith("contains")) and (
            ";" in code or "&" in code
        ):
            raise ValueError(error + "unit statements must occupy one complete line")
        if end:
            if not stack:
                raise ValueError(error + "END without an opening unit")
            unit = stack.pop()
            if end[1] != unit.kind or (end[2] is not None and end[2] != unit.name):
                raise ValueError(error + "END does not match the opening unit")
            unit.end = line
        elif begin:
            if stack and not stack[-1].contains:
                raise ValueError(error + "nested unit requires CONTAINS")
            siblings = stack[-1].children if stack else roots
            if any(unit.name == begin[2] for unit in siblings):
                raise ValueError(error + "duplicate unit name")
            unit = _Unit(begin[1], begin[2], [line])
            siblings.append(unit)
            stack.append(unit)
        elif contains:
            if not stack or stack[-1].contains:
                raise ValueError(error + "unexpected CONTAINS")
            stack[-1].header.append(line)
            stack[-1].contains = True
        elif stack and not stack[-1].contains:
            if code == "end":
                raise ValueError(error + "use an explicit END with the unit kind")
            stack[-1].header.append(line)
        elif code:
            raise ValueError(error + "unexpected code outside a unit body")
    if stack or interface:
        raise ValueError("unterminated unit or interface")
    return _render(roots, 0, separators)
