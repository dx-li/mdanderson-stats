"""ASCII case conversion and a reentrant, checked CDFLIB command lexer."""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_UPPER = str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DELIMITERS = frozenset("()[]{},:;")
_OPERATORS = frozenset("+-*/<>=")
_RECOGNIZED = (
    frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._$ \"'")
    | _DELIMITERS
    | _OPERATORS
)
_MANTISSA = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_NUMBER = re.compile(r"[+-]?" + _MANTISSA + r"(?:[eEdD][+-]?[0-9]+)?\Z")
_EXP_PREFIX = re.compile(r"[+-]?" + _MANTISSA + r"[eEdD]\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9._$]*\Z")
type TokenKind = Literal["IN", "RL", "ID", "QS", "DL", "OP", "OS", "UC"]


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("input must be a string")
    return value


def lower_case_string(text: str) -> str:
    """Return ASCII-lowercased text; preserve other characters and length."""
    return _text(text).translate(_LOWER)


def upper_case_string(text: str) -> str:
    """Return ASCII-uppercased text; preserve other characters and length."""
    return _text(text).translate(_UPPER)


def lower_case_char(char: str) -> str:
    """Lowercase one ASCII character, leaving other Unicode characters alone."""
    if len(_text(char)) != 1:
        raise ValueError("char must contain exactly one character")
    return char.translate(_LOWER)


def upper_case_char(char: str) -> str:
    """Uppercase one ASCII character, leaving other Unicode characters alone."""
    if len(_text(char)) != 1:
        raise ValueError("char must contain exactly one character")
    return char.translate(_UPPER)


@dataclass(frozen=True)
class QlexToken:
    """Decoded token, zero-based source span and defined numeric results.

    overflow=0: numeric fields fit; 1: integer exceeds signed int32;
    2: real exceeds binary64. Unavailable fields are None, never placeholders.
    Non-numeric tokens have both numeric fields None and overflow zero.
    """

    kind: TokenKind
    text: str
    start: int
    stop: int
    integer: int | None = None
    real: float | None = None
    overflow: int = 0
    underflow: bool = False

    @property
    def length(self) -> int:
        """Length of decoded text, excluding quotes and spaces after unary signs."""
        return len(self.text)


def _number(text: str, start: int, stop: int) -> QlexToken:
    kind: TokenKind = "RL" if any(c in text for c in ".eEdD") else "IN"
    value = float(text.replace("D", "e").replace("d", "e"))
    if not math.isfinite(value):
        return QlexToken(kind, text, start, stop, overflow=2)
    # At finite binary64 magnitude, truncation cannot allocate an unbounded
    # integer, irrespective of how many digits/exponent characters were read.
    integer = math.trunc(value)
    overflow = 0 if -(2**31) <= integer <= 2**31 - 1 else 1
    mantissa = re.split("[eEdD]", text, maxsplit=1)[0]
    underflow = value == 0 and any(c in "123456789" for c in mantissa)
    return QlexToken(
        kind, text, start, stop, None if overflow else integer, value, overflow, underflow
    )


def _numeric_start(text: str, pos: int) -> bool:
    return pos < len(text) and (
        "0" <= text[pos] <= "9"
        or (text[pos] == "." and pos + 1 < len(text) and "0" <= text[pos + 1] <= "9")
    )


def _after_sign(text: str, pos: int) -> int | None:
    after = pos + 1
    while after < len(text) and text[after] == " ":
        after += 1
    return after if _numeric_start(text, after) else None


def _scan(text: str) -> Iterator[QlexToken]:
    pos, previous = 0, ""
    size = len(text)
    while pos < size:
        separated = text[pos] == " "
        while pos < size and text[pos] == " ":
            pos += 1
        if pos == size:
            return
        start, char = pos, text[pos]
        if char in "\"'":
            pos += 1
            pieces = []
            while pos < size:
                if text[pos] == char:
                    if pos + 1 < size and text[pos + 1] == char:
                        pieces.append(char)
                        pos += 2
                        continue
                    pos += 1
                    token = QlexToken("QS", "".join(pieces), start, pos)
                    break
                pieces.append(text[pos])
                pos += 1
            else:
                token = QlexToken("OS", "".join(pieces), start, pos)
        elif char in _DELIMITERS:
            pos += 1
            token = QlexToken("DL", char, start, pos)
        elif char not in _RECOGNIZED:
            pos += 1
            while pos < size and text[pos] not in _RECOGNIZED:
                pos += 1
            token = QlexToken("UC", text[start:pos], start, pos)
        else:
            sign = ""
            if char in "+-" and (separated or previous in ("", "OP", "DL")):
                after = _after_sign(text, pos)
                if after is not None:
                    sign, pos = char, after
            if char in _OPERATORS and not sign:
                pos += 1
                while pos < size and text[pos] in _OPERATORS:
                    # A following numeric sign starts a new token after this OP.
                    if text[pos] in "+-" and _after_sign(text, pos) is not None:
                        break
                    pos += 1
                token = QlexToken("OP", text[start:pos], start, pos)
            else:
                body_start = pos
                while pos < size:
                    current = text[pos]
                    if current in _OPERATORS:
                        if current in "+-" and _EXP_PREFIX.fullmatch(sign + text[body_start:pos]):
                            pos += 1
                            continue
                        break
                    if current == " " or current in _DELIMITERS or current not in _RECOGNIZED:
                        break
                    pos += 1
                body = sign + text[body_start:pos]
                if _NUMBER.fullmatch(body):
                    token = _number(body, start, pos)
                else:
                    kind: TokenKind = "ID" if _IDENTIFIER.fullmatch(body) else "OS"
                    token = QlexToken(kind, body, start, pos)
        previous = token.kind
        yield token


def qlex(text: str) -> Iterator[QlexToken]:
    """Iterate CDFLIB command tokens with independent per-stream state.

    Spaces separate tokens; other controls and non-ASCII characters outside
    quotes form UC tokens. Signs are unary at start, after a space, operator
    or delimiter when followed by a numeric mantissa. Restart by calling qlex
    again. Malformed numeric/quoted input becomes OS, with no fabricated value.
    """
    return _scan(_text(text))
