"""STATTAB file selection with explicit outcomes and Python stream ownership."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from .cdflib_console import CDFConsole, CDFConsoleError, _positive


@dataclass(frozen=True)
class STATTABFile:
    """An opened caller-owned stream, or an explicit non-opening outcome."""

    status: Literal["opened", "quit", "back", "declined"]
    path: Path | None = None
    stream: TextIO | None = None


def stattab_open_file(
    console: CDFConsole,
    message: str = "Enter file name",
    *,
    read: bool = True,
    confirm: bool = False,
    append_ok: bool = True,
    max_attempts: int = 3,
) -> STATTABFile:
    """Select a UTF-8 file; return a stream whose caller must close it.

    Existing write targets require q/r/o/a (quit/retry/overwrite/append).
    Optional q/r/p confirmation occurs BEFORE opening or truncating a file.
    New files use exclusive creation. Read requests open read-only streams.
    Filename comments start at #; exact quit/back commands cancel selection.
    """
    if not isinstance(console, CDFConsole) or not isinstance(message, str):
        raise ValueError("require a CDFConsole and plain prompt string")
    if any(not isinstance(v, bool) for v in (read, confirm, append_ok)):
        raise ValueError("read, confirm and append_ok must be boolean")
    _positive(max_attempts, "max_attempts")
    cause: OSError | None = None
    for _ in range(max_attempts):
        console.write_message(message)
        console.prompt()
        name = console._read().split("#", 1)[0].strip()
        if name.lower() in ("quit", "back"):
            return STATTABFile("quit" if name.lower() == "quit" else "back")
        if not name or any(ord(c) < 32 or ord(c) == 127 for c in name):
            console.write_message("Invalid file name. Please try again.")
            continue
        path = Path(name)
        try:
            path.stat()
            exists = True
        except FileNotFoundError:
            exists = False
        except OSError as error:
            cause = error
            console.write_message(f"Cannot inspect file: {error}")
            continue
        in_use = False
        try:
            for active in (console.input, console.output, console.report):
                active_name = getattr(active, "name", None)
                if isinstance(active_name, (str, Path)) and exists:
                    try:
                        in_use |= path.samefile(active_name)
                    except FileNotFoundError:
                        pass  # Streams such as <stdin> need not name a filesystem entry.
        except OSError as error:
            cause = error
            console.write_message(f"Cannot check active file: {error}")
            continue
        if in_use:
            console.write_message("File is already used by this console. Please try again.")
            continue
        mode: Literal["r", "x", "w", "a"] = "r" if read else "x"
        if read and not exists:
            cause = FileNotFoundError(f"File does not exist: {path}")
            console.write_message(str(cause))
            continue
        if not read and exists:
            choices = "qroa" if append_ok else "qro"
            action = console.get_character(
                choices,
                "File exists: q quit; r retry; o overwrite" + ("; a append" if append_ok else ""),
            )
            if action == 1:
                return STATTABFile("quit")
            if action == 2:
                continue
            mode = "w" if action == 3 else "a"
        if confirm:
            action = console.get_character(
                "qrp", f"Use {path} in mode {mode}? q quit; r retry; p proceed"
            )
            if action == 1:
                return STATTABFile("quit")
            if action == 2:
                continue
        try:
            stream = path.open(mode, encoding="utf-8", newline="")
        except OSError as error:
            cause = error
            console.write_message(f"Cannot open file: {error}")
            continue
        return STATTABFile("opened", path, stream)
    raise CDFConsoleError("File selection exhausted max_attempts") from cause


def stattab_report_file_dialogue(console: CDFConsole) -> STATTABFile:
    """Ask whether to create a report; opened streams remain caller-owned."""
    if not console.get_yn("Do you want a report file?"):
        return STATTABFile("declined")
    return stattab_open_file(console, "Enter report file name", read=False)
