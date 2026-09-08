"""MULTI QLEX token states adapted under the original MULTI terms.

See THIRD_PARTY_NOTICES.md and notices/mdanderson-multi-LEGALITIES.txt.
Numeric conversion retains decimal digit accumulation, with signed integers
corrected and bounded exponent processing to avoid pathological input runtimes.
"""

from collections.abc import Iterator
from dataclasses import dataclass

_TRANTB = (
    (3, 4, 12, 12, 9, 9, 13, 1, 0, 11, 12, 13, 14),
    (3, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
    (3, 4, 0, 0, 13, 13, 13, 0, 0, 13, 0, 13, 0),
    (5, 13, 0, 0, 6, 13, 13, 0, 0, 13, 0, 13, 0),
    (5, 13, 0, 0, 6, 13, 13, 0, 0, 13, 0, 13, 0),
    (8, 13, 7, 7, 13, 13, 13, 0, 13, 13, 13, 13, 0),
    (8, 13, 13, 13, 13, 13, 13, 0, 13, 13, 13, 13, 0),
    (8, 13, 0, 0, 13, 13, 13, 0, 0, 13, 0, 13, 0),
    (9, 9, 0, 0, 9, 9, 9, 0, 0, 13, 0, 13, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11),
    (0, 0, 12, 12, 0, 0, 0, 0, 0, 0, 12, 0, 0),
    (13, 13, 13, 13, 13, 13, 13, 0, 13, 13, 13, 13, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14),
)

_ACTNTB = (
    (1, 5, 13, 13, 11, 11, 6, 12, 20, 15, 13, 6, 14),
    (1, 5, 24, 24, 24, 24, 24, 12, 24, 24, 24, 24, 24),
    (1, 5, 17, 17, 6, 6, 6, 17, 17, 6, 17, 6, 17),
    (2, 6, 18, 18, 4, 6, 6, 25, 18, 6, 18, 6, 18),
    (2, 6, 18, 18, 5, 6, 6, 18, 18, 6, 18, 6, 18),
    (3, 6, 9, 10, 6, 6, 6, 22, 6, 6, 6, 6, 22),
    (3, 6, 6, 6, 6, 6, 6, 22, 6, 6, 6, 6, 22),
    (3, 6, 18, 18, 6, 6, 6, 18, 18, 6, 18, 6, 18),
    (4, 4, 19, 19, 4, 4, 4, 19, 19, 6, 19, 6, 19),
    (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 16, 4, 4, 4),
    (21, 21, 4, 4, 21, 21, 21, 21, 21, 21, 4, 21, 21),
    (4, 4, 4, 4, 4, 4, 4, 22, 4, 4, 4, 4, 22),
    (23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 4),
)


@dataclass(frozen=True)
class Token:
    text: str
    kind: str
    column: int
    value: float | None


def _class(char: str) -> int:
    if "0" <= char <= "9":
        return 0
    for i, chars in enumerate((".", "+", "-", "EeDd"), 1):
        if char in chars:
            return i
    if char.isascii() and char.isalpha():
        return 5
    if char in "$_":
        return 6
    if char == " ":
        return 7
    if char in ",(){}:;[]":
        return 8
    if char in "\"'":
        return 9
    if char in "*/<>=":
        return 10
    return 12


def tokens(line: str) -> Iterator[Token]:
    # A sentinel makes the final token well defined even on long physical lines.
    line += " "
    pos, previous_type = 0, ""
    while pos < len(line):
        state, kind, value, first, old_char = 1, "", "", True, " "
        start = pos
        integer = fraction = exponent = 0.0
        power, places, sign, exponent_sign = 1.0, 0, 1, 1
        quote = ""
        while pos < len(line):
            char = line[pos]
            cls = _class(char)
            next_char = line[pos + 1] if pos + 1 < len(line) else " "
            unary = (
                first
                and cls in (2, 3)
                and (previous_type in ("", "OP", "DL") or _class(next_char) == 0)
            )
            action = (7 if cls == 2 else 8) if unary else _ACTNTB[state - 1][cls]
            if first and cls != 7:
                start = pos
            if action in (17, 18, 19, 21, 22, 23):
                break
            if action == 20:
                value += char
                kind = "DL"
                pos += 1
                break
            if action == 24:
                if old_char == " ":
                    pos -= 1
                    kind = "OP"
                else:
                    value += char
                    kind = "OS"
                # Original action 24 falls through to action 25.
                break
            if action == 25:
                if value.startswith("."):
                    kind = "OS"
                break
            if action == 16:
                if char == quote:
                    if next_char == quote:
                        value += char
                        pos += 1
                    else:
                        pos += 1
                        break
                else:
                    value += char
            elif action == 15:
                quote, kind = char, "QS"
            elif action != 12:
                value += char
                if action == 1:
                    integer = 10 * integer + int(char)
                    kind = "IN"
                elif action == 2:
                    places += 1
                    power = float(f"1e-{places}") if places <= 20 else power * 0.1
                    fraction += int(char) * power
                    kind = "RL"
                elif action == 3:
                    exponent = min(2147483647.0, 10 * exponent + int(char))
                elif action in (5, 6, 7, 8, 11, 13, 14):
                    kind = {5: "RL", 6: "OS", 7: "IN", 8: "IN", 11: "ID", 13: "OP", 14: "UC"}[
                        action
                    ]
                    if action in (7, 8):
                        sign = 1 if action == 7 else -1
                elif action in (9, 10):
                    exponent_sign = 1 if action == 9 else -1
            state = 2 if unary else _TRANTB[state - 1][cls]
            pos += 1
            if cls != 7:
                first = False
            old_char = char
        else:
            if kind == "QS":
                kind = "OS"
            elif state == 2:
                kind = "OP"
            else:
                return
        if not kind:
            return
        number = None
        if kind in ("IN", "RL"):
            number = integer + fraction
            if kind == "RL":
                if exponent >= 2147483647:
                    number = float("inf")
                else:
                    # After at most 700 decimal shifts binary64 has saturated;
                    # remaining shifts cannot change zero or infinity.
                    for _ in range(min(int(exponent), 700)):
                        number = number * 10 if exponent_sign == 1 else number / 10
                        if number == 0 or number == float("inf"):
                            break
            number *= sign
        previous_type = kind
        yield Token(value, kind, start + 1, number)
