"""Interactive SPPCR output choices made before any report file is modified."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .cdflib_console import CDFConsole, CDFConsoleError, _positive
from .sppcr_analysis import SPPCRReports
from .sppcr_files import _publish


@dataclass(frozen=True)
class SPPCRSavedFiles:
    """Saved absolute paths, or an explicit decision not to save this analysis."""

    status: Literal["saved", "declined"]
    report: Path | None = None
    simulations: Path | None = None


def _same(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def _select(
    console: CDFConsole,
    default: Path,
    protected: list[Path],
    max_attempts: int,
) -> tuple[Path, str] | None:
    for _ in range(max_attempts):
        name = console.get_string(
            f"Output path [blank uses {default}; quit/back cancels saving]:",
            allow_blank=True,
        ).strip()
        if name.lower() in ("quit", "back"):
            return None
        path = (Path(name) if name else default).absolute()
        if any(_same(path, other) for other in protected):
            console.write_message(
                "Output conflicts with an input, active stream or other output. Retry."
            )
            continue
        if path.is_dir() or not path.parent.is_dir():
            console.write_message("Output must name a file in an existing directory. Retry.")
            continue
        if os.path.lexists(path):
            action = console.get_character(
                "qroa", f"{path} exists: q cancel; r retry; o overwrite; a append"
            )
            if action == 1:
                return None
            if action == 2:
                continue
            return path, "w" if action == 3 else "a"
        return path, "x"
    raise CDFConsoleError("SPPCR output selection exhausted max_attempts")


def sppcr_output_dialogue(
    console: CDFConsole,
    reports: SPPCRReports,
    *,
    default_name: str | Path = "sppcr",
    protected_paths: tuple[str | Path, ...] = (),
    max_attempts: int = 3,
    max_append_characters: int = 10000000,
) -> SPPCRSavedFiles:
    """Choose report and optional replicate paths, then publish staged files.

    Defaults use the basename of default_name with .ans/.sim extensions, in cwd.
    Existing files offer native q/r/o/a choices. Neither file changes until both
    choices are complete and all contents are staged. Append preserves existing
    UTF-8 text literally, without adding separators; it uses bounded reads.
    Cancellation or EOF before publication changes no files. Publication errors
    propagate; the pair is not a transaction and concurrent writers are not locked.
    """
    _positive(max_attempts, "max_attempts")
    _positive(max_append_characters, "max_append_characters")
    if not isinstance(console, CDFConsole) or not isinstance(reports, SPPCRReports):
        raise TypeError("require a CDFConsole and SPPCRReports")
    if not isinstance(reports.report, str) or (
        reports.simulations is not None and not isinstance(reports.simulations, str)
    ):
        raise TypeError("reports must contain text")
    stem = Path(default_name).stem
    if not stem or stem in (".", ".."):
        raise ValueError("default_name must have a nonempty basename")
    protected = [Path(path) for path in protected_paths]
    for stream in (console.input, console.output, console.report):
        name = getattr(stream, "name", None)
        if isinstance(name, (str, Path)):
            protected.append(Path(name))
    if not console.get_yn("Save this analysis to report files? (y/n)"):
        return SPPCRSavedFiles("declined")
    paths: list[Path] = []
    modes: list[str] = []
    texts = [reports.report]
    if reports.simulations is not None:
        texts.append(reports.simulations)
    for extension in (".ans", ".sim")[: len(texts)]:
        selection = _select(console, Path(stem + extension), protected + paths, max_attempts)
        if selection is None:
            return SPPCRSavedFiles("declined")
        path, mode = selection
        paths.append(path)
        modes.append(mode)
    payloads: list[bytes] = []
    for path, mode, text in zip(paths, modes, texts, strict=True):
        if mode == "a":
            with path.open("r", encoding="utf-8", newline="") as stream:
                previous = stream.read(max_append_characters + 1)
            if len(previous) > max_append_characters:
                raise ValueError("existing report exceeds max_append_characters")
            text = previous + text
        payloads.append(text.encode("utf-8"))
    _publish(paths, payloads, [mode != "x" for mode in modes])
    console.write_message("Saved: " + ", ".join(map(str, paths)))
    return SPPCRSavedFiles("saved", paths[0], paths[1] if len(paths) == 2 else None)
