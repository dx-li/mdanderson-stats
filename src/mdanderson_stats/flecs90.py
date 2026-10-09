"""Python implementation of the public-domain MD Anderson FLECS90 translator.

FLECS control structures become fixed-form Fortran. This translates source; it
neither executes Fortran nor translates statistical algorithms to Python.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_MAX_SOURCE = 4 * 1024 * 1024
_MAX_OUTPUT = 16 * 1024 * 1024
_MAX_DEPTH = 512
_MAX_COMMAND = 32768
_LOOPS = {"DO", "FOR", "WHILE", "UNTIL", "REPEAT WHILE", "REPEAT UNTIL", "LOOP"}
_CHOICES = {"CONDITIONAL", "SELECT"}
_KEYS = (
    _LOOPS
    | _CHOICES
    | {"IF", "UNLESS", "WHEN", "ELSE", "TO", "FIN", "CYCLE", "EXIT", "REVERT", "ENDIF"}
)


class FLECS90SyntaxError(ValueError):
    """An invalid FLECS structure, with its physical source line."""

    def __init__(self, line_number: int, message: str) -> None:
        self.line_number = line_number
        self.message = message
        super().__init__(f"line {line_number}: {message}")


@dataclass(frozen=True)
class FLECS90Translation:
    fortran_source: str
    source_line_numbers: tuple[int, ...]
    """Physical input line corresponding to each output line; comments included."""


@dataclass
class _Line:
    number: int
    text: str

    @property
    def label(self) -> str:
        return self.text[:5].ljust(5)

    @property
    def continuation(self) -> bool:
        return len(self.text) > 5 and self.text[5] not in " 0"


@dataclass
class _Frame:
    key: str
    line: int
    condition: str = ""
    loop: int = 0
    branches: int = 0


def _word(text: str) -> tuple[str, str]:
    match = re.match(r"([^\s(]+)(.*)", text.strip(), re.DOTALL)
    return (match[1], match[2].strip()) if match else ("", text.strip())


def _procedure(word: str) -> str | None:
    if "-" in word and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", word):
        return word.upper().replace("-", "_")
    return None


def _classify(text: str) -> tuple[str, str]:
    word, rest = _word(text)
    upper = word.upper()
    if text.lstrip().startswith("("):
        return "INSTANCE", text.strip()
    if _procedure(word) is not None:
        return "PROCEDURE", word
    if upper == "REPEAT":
        second, tail = _word(rest)
        if second.upper() in {"WHILE", "UNTIL"}:
            return "REPEAT " + second.upper(), tail
    if upper == "DO" and not rest.startswith("("):
        return "NONE", text
    if upper == "CASE":
        return ("SELECT", rest) if rest else ("CONDITIONAL", rest)
    if upper == "END":
        return (
            ("ENDIF", "") if rest.upper() == "IF" else (("END", "") if not rest else ("NONE", text))
        )
    return (upper, rest) if upper in _KEYS else ("NONE", text)


def _instance(text: str, number: int) -> tuple[str, str]:
    text = text.lstrip()
    if not text.startswith("("):
        raise FLECS90SyntaxError(number, "missing parenthesized control specification")
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 1
                else:
                    quote = ""
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[: i + 1], text[i + 1 :].strip()
        i += 1
    raise FLECS90SyntaxError(number, "unbalanced control specification")


def _collapse(text: str) -> str:
    """Collapse outside whitespace while preserving quoted Fortran strings."""
    output: list[str] = []
    quote = ""
    space = False
    i = 0
    while i < len(text):
        char = text[i]
        if quote:
            output.append(char)
            if char == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    output.append(text[i + 1])
                    i += 1
                else:
                    quote = ""
        elif char.isspace():
            space = True
        else:
            if space and output:
                output.append(" ")
            space = False
            output.append(char)
            if char in "\"'":
                quote = char
        i += 1
    return "".join(output)


class _Translator:
    def __init__(
        self,
        source: str,
        echo_comments: bool,
        line_numbers: bool,
        select_case: bool,
        label_loops: bool,
    ) -> None:
        self.lines = source.split("\n")
        if self.lines[-1] == "":
            self.lines.pop()
        if len(self.lines) > 99999:
            raise ValueError("source exceeds 99999 physical lines")
        self.echo_comments = echo_comments
        self.line_numbers = line_numbers
        self.select_case = select_case
        self.label_loops = label_loops
        self.index = 0
        self.current: _Line | None = None
        self.stack: list[_Frame] = []
        self.state = "standard"
        self.loop_count = 0
        self.contains = False
        self.ended = False
        self.output: list[str] = []
        self.mapping: list[int] = []
        self.output_bytes = 0

    def append(self, text: str, number: int, *, numbered: bool = True) -> None:
        if self.line_numbers and numbered:
            text = text.ljust(73) + f"{number:05d}"
        self.output_bytes += len(text) + 1
        if self.output_bytes > _MAX_OUTPUT:
            raise ValueError("translated output exceeds 16 MiB")
        self.output.append(text)
        self.mapping.append(number)

    def emit(self, text: str, number: int, label: str = "     ") -> None:
        text = _collapse(text)
        self.append(label + " " + text[:66], number)
        for start in range(66, len(text), 66):
            self.append("     +" + text[start : start + 66], number)

    def label_continue(self, label: str, number: int) -> None:
        if label.strip():
            self.emit("CONTINUE", number, label)

    def advance(self) -> None:
        blanks: list[int] = []
        self.current = None
        while self.index < len(self.lines):
            raw = self.lines[self.index][:72].replace("\t", " ").rstrip("\r ")
            self.index += 1
            number = self.index
            if not raw:
                blanks.append(number)
                continue
            if self.echo_comments:
                for blank in blanks:
                    self.append("", blank, numbered=False)
            blanks.clear()
            if raw[0] in "Cc*":
                if self.echo_comments:
                    self.append(raw, number)
                continue
            if any(char not in " 0123456789" for char in raw[:5]):
                raise FLECS90SyntaxError(number, "fixed-form labels must be in columns 1..5")
            self.current = _Line(number, raw)
            return

    def collect(self) -> tuple[_Line, str]:
        assert self.current is not None
        first = self.current
        text = first.text[6:]
        self.advance()
        while self.current is not None and self.current.continuation:
            text += " " + self.current.text[6:]
            if len(text) > _MAX_COMMAND:
                raise FLECS90SyntaxError(first.number, "logical statement exceeds 32768 characters")
            self.advance()
        return first, text

    def push(self, frame: _Frame) -> None:
        if len(self.stack) >= _MAX_DEPTH:
            raise FLECS90SyntaxError(frame.line, "structure nesting exceeds 512")
        self.stack.append(frame)

    def loop_name(self, number: int) -> str:
        return f" FLECS_LOOP_{number}" if self.label_loops else ""

    def action(self, text: str, number: int) -> str:
        word, rest = _word(text)
        name = _procedure(word)
        if name is not None:
            if rest:
                raise FLECS90SyntaxError(number, "FLECS procedure calls take no arguments")
            return "CALL " + name
        if text.upper() in {"CYCLE", "EXIT"}:
            frame = next((f for f in reversed(self.stack) if f.key in _LOOPS), None)
            if frame is None:
                raise FLECS90SyntaxError(number, f"nothing to {text.upper()}")
            return text.upper() + self.loop_name(frame.loop)
        if text.upper() == "REVERT":
            if not any(f.key == "TO" for f in self.stack):
                raise FLECS90SyntaxError(number, "nothing to REVERT")
            return "RETURN"
        return text

    def finish(self, number: int, label: str) -> None:
        if not self.stack or self.stack[-1].key == "IFTHEN":
            raise FLECS90SyntaxError(number, "FIN has no matching FLECS structure")
        frame = self.stack[-1]
        if frame.key == "WHEN":
            self.state = "when_fin"
            self.label_continue(label, number)
            return
        self.stack.pop()
        if frame.key in {"INSTANCE", "OTHERWISE"}:
            self.state = "otherwise_fin" if frame.key == "OTHERWISE" else "branch_fin"
            self.label_continue(label, number)
        elif frame.key == "TO":
            self.emit("END SUBROUTINE", number, label)
            self.state = "to_fin"
        elif frame.key in _LOOPS:
            if frame.key.startswith("REPEAT"):
                condition = (
                    "(.NOT." + frame.condition + ")"
                    if frame.key == "REPEAT WHILE"
                    else frame.condition
                )
                self.emit("IF " + condition + " EXIT" + self.loop_name(frame.loop), number, label)
                label = "     "
            self.emit("END DO" + self.loop_name(frame.loop), number, label)
        elif frame.key == "SELECT" and self.select_case:
            self.emit("END SELECT", number, label)
        elif frame.key in _CHOICES and frame.branches == 0:
            self.label_continue(label, number)
        else:
            self.emit("END IF", number, label)

    def process(self, first: _Line, key: str, rest: str) -> None:
        number, label = first.number, first.label
        if self.state == "to_fin" and key != "TO":
            raise FLECS90SyntaxError(number, "only TO or END is valid after a procedure")
        if self.state == "when_fin" and key != "ELSE":
            raise FLECS90SyntaxError(number, "WHEN requires ELSE after its body")
        if self.state in {"branch_fin", "otherwise_fin"} and key not in (
            {"INSTANCE", "FIN"} if self.state == "branch_fin" else {"FIN"}
        ):
            raise FLECS90SyntaxError(number, "expected INSTANCE or FIN after a branch")
        self.state = "standard"
        if key == "FIN":
            self.finish(number, label)
        elif key in {"CYCLE", "EXIT", "REVERT"}:
            self.emit(self.action(key, number), number, label)
        elif key == "PROCEDURE":
            self.emit(self.action(rest, number), number, label)
        elif key == "ENDIF":
            if not self.stack or self.stack[-1].key != "IFTHEN":
                raise FLECS90SyntaxError(number, "ENDIF has no matching Fortran IF THEN")
            self.stack.pop()
            self.emit("END IF", number, label)
        elif key == "ELSE":
            if not self.stack or self.stack[-1].key not in {"WHEN", "IFTHEN"}:
                raise FLECS90SyntaxError(number, "ELSE has no matching WHEN or IF THEN")
            frame = self.stack[-1]
            if frame.key == "IFTHEN":
                self.emit("ELSE" + (" " + rest if rest else ""), number, label)
            else:
                frame.key = "ELSE"
                self.emit("ELSE", number, label)
                if rest:
                    self.emit(self.action(rest, number), number)
                    self.finish(number, "     ")
        elif key in _CHOICES:
            condition = ""
            if key == "SELECT":
                condition, tail = _instance(rest, number)
                if tail:
                    raise FLECS90SyntaxError(number, "unexpected text after SELECT specification")
            elif rest:
                raise FLECS90SyntaxError(number, "CONDITIONAL takes no specification")
            self.push(_Frame(key, number, condition))
            if key == "SELECT" and self.select_case:
                self.emit("SELECT CASE " + condition, number, label)
            else:
                self.label_continue(label, number)
            self.state = "branch_fin"
        elif key == "INSTANCE":
            if not self.stack or self.stack[-1].key not in _CHOICES:
                raise FLECS90SyntaxError(
                    number, "INSTANCE has no matching CASE, CONDITIONAL or SELECT"
                )
            frame = self.stack[-1]
            condition, tail = _instance(rest, number)
            otherwise = condition[1:-1].strip().upper() == "OTHERWISE"
            if frame.key == "SELECT" and self.select_case:
                statement = "CASE DEFAULT" if otherwise else "CASE" + condition
            elif otherwise:
                statement = "ELSE" if frame.branches else "IF (.TRUE.) THEN"
            else:
                test = (
                    "(" + condition + ".EQ." + frame.condition + ")"
                    if frame.key == "SELECT"
                    else condition
                )
                statement = ("ELSE IF " if frame.branches else "IF ") + test + " THEN"
            frame.branches += 1
            self.emit(statement, number, label)
            if tail:
                self.emit(self.action(tail, number), number)
                self.state = "otherwise_fin" if otherwise else "branch_fin"
            else:
                self.push(_Frame("OTHERWISE" if otherwise else "INSTANCE", number))
        elif key == "TO":
            if self.stack:
                raise FLECS90SyntaxError(number, "TO cannot be nested in an unfinished structure")
            name, tail = _word(rest)
            procedure = _procedure(name)
            if procedure is None:
                raise FLECS90SyntaxError(number, "TO requires a hyphenated procedure name")
            if not self.contains:
                self.emit("CONTAINS", number)
                self.contains = True
            self.emit("SUBROUTINE " + procedure, number, label)
            self.push(_Frame("TO", number))
            if tail:
                self.emit(self.action(tail, number), number)
                self.finish(number, "     ")
        elif key in {"IF", "UNLESS", "WHEN"} | _LOOPS:
            condition, tail = ("", "") if key == "LOOP" else _instance(rest, number)
            if key in {"IF", "UNLESS", "WHEN"}:
                test = "(.NOT." + condition + ")" if key == "UNLESS" else condition
                if key == "IF" and tail.upper() == "THEN":
                    self.emit("IF " + test + " THEN", number, label)
                    self.push(_Frame("IFTHEN", number))
                elif key == "WHEN" or not tail:
                    self.emit("IF " + test + " THEN", number, label)
                    self.push(_Frame(key, number))
                    if tail:
                        self.emit(self.action(tail, number), number)
                        self.state = "when_fin"
                else:
                    self.emit("IF " + test + " " + self.action(tail, number), number, label)
            else:
                self.loop_count += 1
                loop = self.loop_count
                frame = _Frame(key, number, condition, loop)
                self.push(frame)
                if key in {"DO", "FOR"}:
                    statement = "DO " + condition[1:-1]
                elif key in {"WHILE", "UNTIL"}:
                    test = "(.NOT." + condition + ")" if key == "UNTIL" else condition
                    statement = "DO WHILE " + test
                else:
                    statement = "DO"
                prefix = f"FLECS_LOOP_{loop}: " if self.label_loops else ""
                self.emit(prefix + statement, number, label)
                if tail:
                    self.emit(self.action(tail, number), number)
                    self.finish(number, "     ")
        else:
            raise FLECS90SyntaxError(number, f"unsupported control keyword: {key}")

    def run(self) -> FLECS90Translation:
        self.advance()
        if self.current is None:
            raise FLECS90SyntaxError(1, "no program statements")
        while self.current is not None:
            first = self.current
            if first.continuation:
                raise FLECS90SyntaxError(first.number, "continuation has no preceding statement")
            key, rest = _classify(first.text[6:])
            if key == "END":
                if self.stack:
                    frame = self.stack[-1]
                    raise FLECS90SyntaxError(
                        first.number, f"{frame.key} on line {frame.line} is unfinished before END"
                    )
                self.append(first.text, first.number)
                self.advance()
                self.state, self.contains, self.loop_count = "standard", False, 0
                self.ended = True
                continue
            self.ended = False
            if key == "NONE":
                # Native Fortran statements and their continuation cards pass through.
                if self.state != "standard":
                    expected = {
                        "when_fin": "WHEN requires ELSE after its body",
                        "to_fin": "only TO or END is valid after a procedure",
                        "branch_fin": "expected INSTANCE or FIN after a branch",
                        "otherwise_fin": "expected FIN after OTHERWISE",
                    }
                    raise FLECS90SyntaxError(first.number, expected[self.state])
                self.append(first.text, first.number)
                self.advance()
                while self.current is not None and self.current.continuation:
                    self.append(self.current.text, self.current.number)
                    self.advance()
                continue
            if key in {"FIN", "CYCLE", "EXIT", "REVERT", "ENDIF", "LOOP", "PROCEDURE"}:
                self.process(first, key, rest)
                self.advance()
                while self.current is not None and self.current.continuation:
                    self.advance()
            else:
                first, text = self.collect()
                key, rest = _classify(text)
                self.process(first, key, rest)
        if self.stack:
            frame = self.stack[-1]
            raise FLECS90SyntaxError(frame.line, f"unfinished {frame.key} at end of file")
        if not self.ended:
            raise FLECS90SyntaxError(len(self.lines), "missing END statement")
        return FLECS90Translation("\n".join(self.output) + "\n", tuple(self.mapping))


def translate_flecs90(
    source: str,
    *,
    echo_comments: bool = False,
    line_numbers: bool = False,
    select_case: bool = False,
    label_loops: bool = True,
) -> FLECS90Translation:
    """Translate bounded fixed-form FLECS source into Fortran 90 source.

    Input columns after 72 are ignored, as in the native translator. Output
    sequence numbers occupy columns 74..78 when requested. Only FLECS loops
    receive generated names; existing Fortran loops pass through unchanged.
    Malformed structures raise FLECS90SyntaxError instead of emitting a repaired
    program. Quoted strings preserve whitespace and literal parentheses.
    """
    if not isinstance(source, str) or len(source) > _MAX_SOURCE or not source.isascii():
        raise ValueError("source must be ASCII text of at most 4 MiB")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in source):
        raise ValueError("source contains unsupported control characters")
    if any(
        type(option) is not bool
        for option in (echo_comments, line_numbers, select_case, label_loops)
    ):
        raise ValueError("translation options must be boolean")
    return _Translator(source, echo_comments, line_numbers, select_case, label_loops).run()


def translate_flecs90_file(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    echo_comments: bool = False,
    line_numbers: bool = False,
    select_case: bool = False,
    label_loops: bool = True,
) -> FLECS90Translation:
    """Translate a source file and atomically save a separate Fortran file."""
    source = Path(source_path)
    destination = source.with_suffix(".f") if output_path is None else Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("output must differ from input")
    if source.stat().st_size > _MAX_SOURCE:
        raise ValueError("source file exceeds 4 MiB")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    result = translate_flecs90(
        source.read_text(encoding="ascii"),
        echo_comments=echo_comments,
        line_numbers=line_numbers,
        select_case=select_case,
        label_loops=label_loops,
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=destination.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(result.fortran_source)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # An atomic no-clobber save also protects files created during translation.
            os.link(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate fixed-form FLECS to Fortran 90")
    parser.add_argument("files", nargs="+")
    parser.add_argument("-f", "--check-only", action="store_true")
    parser.add_argument("-c", "--echo-comments", action="store_true")
    parser.add_argument("-n", "--line-numbers", action="store_true")
    parser.add_argument("-s", "--select-case", action="store_true")
    parser.add_argument("-l", "--unlabelled-loops", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for filename in args.files:
        options = dict(
            echo_comments=args.echo_comments,
            line_numbers=args.line_numbers,
            select_case=args.select_case,
            label_loops=not args.unlabelled_loops,
        )
        try:
            if args.check_only:
                path = Path(filename)
                if path.stat().st_size > _MAX_SOURCE:
                    raise ValueError("source file exceeds 4 MiB")
                translate_flecs90(path.read_text(encoding="ascii"), **options)
            else:
                translate_flecs90_file(filename, overwrite=args.overwrite, **options)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{filename}: {error}\n")


if __name__ == "__main__":
    main()
