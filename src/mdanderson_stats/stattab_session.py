"""Checked STATTAB requests and transactional per-session parameter reuse."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from numpy.typing import ArrayLike

from ._validation import finite
from .cdflib_number_list import CDFNumberList
from .cdflib_strings import qlex
from .stattab_results import STATTAB_DISTRIBUTIONS, STATTABResult, stattab_solve


@dataclass(frozen=True)
class STATTABRequest:
    """Parsed command; numerical domains and reused values are checked at execution."""

    distribution: str
    action: Literal["solve", "help", "menu"]
    compute: str | None
    parameters: Mapping[str, float]
    reuse: tuple[str, ...] = ()
    table_parameter: str | None = None


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _distribution(name: str) -> str:
    if not isinstance(name, str) or name not in STATTAB_DISTRIBUTIONS:
        raise ValueError("unknown STATTAB distribution")
    return name


def parse_stattab_request(
    distribution: str,
    line: str,
    *,
    max_length: int = 4096,
) -> STATTABRequest:
    """Parse source-order numbers, ?, ., T and = without reading or mutating state.

    Exactly one ? and at most one T are required for a solve. Each complementary
    pair has exactly one dot. HELP is an exact case-insensitive command; an empty
    or comment-only line requests the distribution menu. Spaces, tabs and commas
    separate fields. Numeric D exponents are accepted; overflow/underflow fail.
    """
    name = _distribution(distribution)
    _positive_integer(max_length, "max_length")
    if not isinstance(line, str) or len(line) > max_length:
        raise ValueError("request must be a string within max_length")
    if "\n" in line or "\r" in line:
        raise ValueError("a request must contain one line")
    text = line.split("#", 1)[0].replace("\t", " ").strip()
    if not text or text.upper() == "HELP":
        return STATTABRequest(name, "help" if text else "menu", None, MappingProxyType({}))
    tokens: list[str | float] = []
    descriptor = STATTAB_DISTRIBUTIONS[name]
    for token in qlex(text):
        if token.kind == "DL" and token.text == ",":
            continue
        if len(tokens) == len(descriptor.parameters):
            raise ValueError("too many parameters")
        if token.kind in ("IN", "RL"):
            if token.real is None or token.overflow == 2 or token.underflow:
                raise ValueError("numeric input overflows or underflows float64")
            tokens.append(token.real)
        elif (
            token.kind != "QS"
            and token.text.upper() in ("?", ".", "T", "=")
            and text[token.start : token.stop].upper() == token.text.upper()
        ):
            tokens.append(token.text.upper())
        else:
            raise ValueError(f"invalid parameter token: {token.text}")
    if len(tokens) != len(descriptor.parameters):
        raise ValueError(f"{name} requires {len(descriptor.parameters)} parameters")
    if tokens.count("?") != 1 or tokens.count("T") > 1:
        raise ValueError("require exactly one ? and at most one T")
    fields = dict(zip(descriptor.parameters, tokens, strict=True))
    pairs = [group for group in descriptor.groups if len(group) == 2]
    for pair in pairs:
        if sum(fields[key] == "." for key in pair) != 1:
            raise ValueError("each complementary pair requires exactly one omitted member (.)")
    paired = {key for pair in pairs for key in pair}
    if any(value == "." and key not in paired for key, value in fields.items()):
        raise ValueError("a required parameter cannot be omitted")
    compute = next(key for key, value in fields.items() if value == "?")
    if not any(compute in group for group in descriptor.groups):
        raise ValueError(f"{name} does not support computing {compute}")
    values = {key: value for key, value in fields.items() if isinstance(value, float)}
    reuse = tuple(key for key, value in fields.items() if value == "=")
    table = next((key for key, value in fields.items() if value == "T"), None)
    return STATTABRequest(name, "solve", compute, MappingProxyType(values), reuse, table)


class STATTABSession:
    """One selected distribution and its last completed row, without global state.

    execute returns a numerical result or a help/menu request for the caller to
    render. Successful nonempty tables retain their last row for =. Errors leave
    prior values intact. Blank/menu commands and select clear reuse state.
    """

    def __init__(
        self, distribution: str, *, max_table_size: int = 100, max_length: int = 4096
    ) -> None:
        self._distribution = _distribution(distribution)
        self._max_table_size = _positive_integer(max_table_size, "max_table_size")
        self._max_length = _positive_integer(max_length, "max_length")
        self._previous: Mapping[str, float] = MappingProxyType({})
        self._last_result: STATTABResult | None = None

    @property
    def distribution(self) -> str:
        return self._distribution

    @property
    def previous(self) -> Mapping[str, float]:
        """Immutable snapshot of the last completed nonempty row in input order."""
        return self._previous

    @property
    def last_result(self) -> STATTABResult | None:
        return self._last_result

    def select(self, distribution: str) -> None:
        """Select a distribution and clear prior values, including when reselected."""
        name = _distribution(distribution)
        self._distribution = name
        self._previous = MappingProxyType({})
        self._last_result = None

    def execute(
        self,
        line: str,
        *,
        table: ArrayLike | CDFNumberList | None = None,
        df_bracket: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> STATTABResult | STATTABRequest:
        """Parse and solve atomically; table must be a finite one-dimensional list.

        Supply table only with T. A CDFNumberList is snapshotted, not mutated.
        A scalar df bracket may select a noncentral-t root. Parser/session calls
        do not broadcast brackets into extra rows; use stattab_solve for that.
        """
        request = parse_stattab_request(self._distribution, line, max_length=self._max_length)
        if request.action != "solve":
            if table is not None or df_bracket is not None:
                raise ValueError("help/menu commands do not accept numerical inputs")
            if request.action == "menu":
                self.select(self._distribution)
            return request
        if (table is None) != (request.table_parameter is None):
            raise ValueError("supply table exactly when the request contains T")
        inputs: dict[str, ArrayLike] = dict(request.parameters)
        for key in request.reuse:
            if key not in self._previous:
                raise ValueError(f"{key} has no previous value in this session")
            inputs[key] = self._previous[key]
            # Preserve the smaller saved member even if the selected one rounded
            # to one. The request still names only one member of this pair.
            pair = next(
                (
                    g
                    for g in STATTAB_DISTRIBUTIONS[self._distribution].groups
                    if key in g and len(g) == 2
                ),
                (),
            )
            for partner in pair:
                inputs[partner] = self._previous[partner]
        if request.table_parameter is not None:
            assert table is not None
            data = table.values if isinstance(table, CDFNumberList) else finite(table, "table")
            if data.ndim != 1 or data.size > self._max_table_size:
                raise ValueError("table must be one-dimensional and within max_table_size")
            inputs[request.table_parameter] = data
        if df_bracket is not None:
            if len(df_bracket) != 2 or any(finite(v, "df_bracket").ndim != 0 for v in df_bracket):
                raise ValueError("session df_bracket requires two scalar endpoints")
        assert request.compute is not None
        result = stattab_solve(
            self._distribution, compute=request.compute, df_bracket=df_bracket, **inputs
        )
        if result.values.size:
            previous = {
                key: float(value.reshape(-1)[-1]) for key, value in result.parameters.items()
            }
            self._previous = MappingProxyType(previous)
        self._last_result = result
        return result
