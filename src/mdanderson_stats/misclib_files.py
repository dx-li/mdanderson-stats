"""Interactive MISCLIB file selection with confirmation before filesystem changes."""

from codecs import lookup
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from .cdflib_console import CDFConsole, _CharacterAttemptsExhausted


@dataclass(frozen=True)
class MisclibFileSelection:
    """Status 0 opens a caller-owned stream; 1 exhausts attempts, 2 quits, 3 goes back."""

    status: Literal[0, 1, 2, 3]
    stream: TextIO | None
    path: Path | None
    action: str | None
    delimiter: str
    error: str | None = None


def _filename(line: str) -> str:
    text = line.strip(" ")
    if not text:
        return ""
    if text[0] in "\"'":
        end = text.find(text[0], 1)
        if end < 0:
            raise ValueError("Missing closing filename quote")
        if text[end + 1 :] and not text[end + 1 :].startswith(" "):
            raise ValueError("Separate a filename comment with a space")
        return text[1:end]
    return text.partition(" ")[0]


def misclib_open_file(
    *,
    console: CDFConsole | None = None,
    message: str = "",
    read_only: bool = False,
    appendable: bool = True,
    delimiter: str = "none",
    max_attempts: int = 3,
    encoding: str = "utf-8",
) -> MisclibFileSelection:
    """Select/read/create/overwrite/append a text file, returning an explicit result.

    Filename input accepts a first space-delimited word or a quoted path with
    spaces, followed by an optional comment. Back/quit are case-insensitive.
    Confirmation precedes opening or truncation. The caller must close the stream.
    Delimiter records a legacy list-directed formatting preference; raw Python
    stream writes remain literal and are not transformed into Fortran NAMELIST.
    """
    if not isinstance(message, str) or not isinstance(encoding, str):
        raise ValueError("message and encoding must be strings")
    if not isinstance(read_only, bool) or not isinstance(appendable, bool):
        raise ValueError("read_only and appendable must be boolean")
    if not isinstance(delimiter, str) or delimiter.lower() not in ("none", "quote", "apostrophe"):
        raise ValueError("delimiter must be none, quote or apostrophe")
    delimiter = delimiter.lower()
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    lookup(encoding)  # Validate an unknown encoding before any file is opened.
    target = CDFConsole() if console is None else console
    last_error: str | None = None
    for _ in range(max_attempts):
        if message:
            target.output.write(message + "\n")
        target.output.write("Enter a filename:")
        target.prompt()
        line = target._read()
        try:
            filename = _filename(line)
        except ValueError as error:
            last_error = str(error)
            target.output.write(last_error + ". Please try again.\n")
            continue
        if filename.lower() in ("back", "quit"):
            return MisclibFileSelection(
                3 if filename.lower() == "back" else 2, None, None, None, delimiter
            )
        if not filename or "\x00" in filename:
            last_error = "Filename must be nonblank and cannot contain NUL"
            target.output.write(last_error + ". Please try again.\n")
            continue
        path = Path(filename)
        try:
            exists = path.exists()
        except OSError as error:
            last_error = str(error)
            target.output.write(last_error + "\n")
            continue
        action = "read" if read_only else "create"
        if read_only and not exists:
            last_error = f"File is not available: {path}"
            target.output.write(last_error + "\n")
            continue
        try:
            if exists and not read_only:
                choices = "qroa" if appendable else "qro"
                choice = target.get_character(
                    choices,
                    f"File exists: {path}. Quit, retry, overwrite"
                    + (" or append?" if appendable else "?"),
                )
                if choice == 1:
                    return MisclibFileSelection(2, None, None, None, delimiter)
                if choice == 2:
                    continue
                action = "overwrite" if choice == 3 else "append"
            choice = target.get_character(
                "qrp", f"{action.capitalize()} {path}: quit, retry or proceed?"
            )
        except _CharacterAttemptsExhausted as error:
            last_error = str(error)
            continue
        if choice == 1:
            return MisclibFileSelection(2, None, None, None, delimiter)
        if choice == 2:
            continue
        try:
            # NEW must not overwrite a path created by another process after
            # confirmation; existing files must still exist for r+ operations.
            mode: Literal["r", "x+", "r+"] = (
                "r" if action == "read" else "x+" if action == "create" else "r+"
            )
            with ExitStack() as stack:
                stream = stack.enter_context(path.open(mode, encoding=encoding))
                if action == "overwrite":
                    stream.truncate(0)
                elif action == "append":
                    stream.seek(0, 2)
                result = MisclibFileSelection(0, stream, path, action, delimiter)
                stack.pop_all()
                return result
        except OSError as error:
            last_error = str(error)
            target.output.write(f"Could not {action} {path}: {error}\n")
    return MisclibFileSelection(1, None, None, None, delimiter, last_error)
