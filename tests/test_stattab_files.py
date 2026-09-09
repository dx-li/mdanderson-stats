"""Real temporary files verify modes, cancellation and stream ownership."""

import io
from pathlib import Path

import pytest

from mdanderson_stats import (
    CDFConsole,
    CDFConsoleError,
    run_stattab,
    stattab_open_file,
    stattab_report_file_dialogue,
)


def console(text):
    return CDFConsole(io.StringIO(text), io.StringIO())


def test_read_only_existing_file_and_caller_ownership(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("data\n")
    opened = stattab_open_file(console(str(path) + "\n"))
    assert opened.status == "opened" and opened.path == path
    with opened.stream as stream:
        assert stream.read() == "data\n" and not stream.writable()
    assert path.read_text() == "data\n"


@pytest.mark.parametrize("action,expected", [("a", "originalnew"), ("o", "new")])
def test_explicit_append_and_overwrite(tmp_path, action, expected):
    path = tmp_path / "report.txt"
    path.write_text("original")
    opened = stattab_open_file(console(f"{path}\n{action}\n"), read=False)
    with opened.stream as stream:
        assert stream.writable() and not stream.readable()
        stream.write("new")
    assert path.read_text() == expected


def test_new_unicode_filename_spaces_and_comment(tmp_path):
    path = tmp_path / "résumé report.txt"
    opened = stattab_open_file(console(f"{path} # note\n"), read=False)
    with opened.stream as stream:
        stream.write("test")
    assert path.read_text() == "test"


@pytest.mark.parametrize("command", ["quit", "QUIT", "back", "BACK"])
def test_exact_cancel_commands(command):
    opened = stattab_open_file(console(command + "\n"), read=False)
    assert opened.status == command.lower() and opened.stream is None and opened.path is None


def test_existing_file_retry_selects_new_target(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.write_text("keep")
    opened = stattab_open_file(console(f"{old}\nr\n{new}\n"), read=False)
    opened.stream.close()
    assert old.read_text() == "keep" and new.exists()


def test_confirmation_precedes_any_creation_or_truncation(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.write_text("keep")
    canceled = stattab_open_file(console(f"{old}\no\nq\n"), read=False, confirm=True)
    assert canceled.status == "quit" and old.read_text() == "keep"
    canceled = stattab_open_file(console(f"{new}\nq\n"), read=False, confirm=True)
    assert canceled.status == "quit" and not new.exists()
    opened = stattab_open_file(console(f"{old}\no\nr\n{new}\np\n"), read=False, confirm=True)
    opened.stream.close()
    assert old.read_text() == "keep" and new.exists()


def test_append_can_be_disabled(tmp_path):
    path = tmp_path / "report"
    path.write_text("keep")
    c = console(f"{path}\na\nq\n")
    opened = stattab_open_file(c, read=False, append_ok=False)
    assert opened.status == "quit" and path.read_text() == "keep"
    assert "Invalid choice" in c.output.getvalue()


def test_missing_read_file_and_directory_fail_with_bounded_retries(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(CDFConsoleError, match="max_attempts") as error:
        stattab_open_file(console((str(missing) + "\n") * 3))
    assert isinstance(error.value.__cause__, FileNotFoundError)
    with pytest.raises(CDFConsoleError):
        stattab_open_file(console((str(tmp_path) + "\n") * 3))
    assert not missing.exists()


def test_invalid_names_eof_and_file_configuration():
    with pytest.raises(CDFConsoleError):
        stattab_open_file(console("\n# comment\nbad\x00name\n"))
    with pytest.raises(EOFError):
        stattab_open_file(console(""))
    for kwargs in [dict(read=1), dict(confirm="yes"), dict(append_ok=None), dict(max_attempts=0)]:
        with pytest.raises(ValueError):
            stattab_open_file(console("quit\n"), **kwargs)


def test_report_dialogue_decline_and_caller_owned_file(tmp_path):
    assert stattab_report_file_dialogue(console("n\n")).status == "declined"
    path = tmp_path / "report"
    opened = stattab_report_file_dialogue(console(f"y\n{path}\n"))
    assert opened.status == "opened" and not opened.stream.closed
    opened.stream.close()


@pytest.mark.parametrize("ending,reason", [("\n0\n", "exit"), ("", "eof")])
def test_application_report_dialogue_closes_owned_real_file(tmp_path, monkeypatch, ending, reason):
    path = tmp_path / "report.txt"
    original = Path.open
    opened = []

    def track(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        if self == path:
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", track)
    result = run_stattab(
        io.StringIO(f"y\n{path}\n9\n1 0 1 ? .\n" + ending), io.StringIO(), ask_report=True
    )
    assert result.reason == reason and result.completed == 1
    assert len(opened) == 1 and opened[0].closed
    assert "normal: computed cum/ccum" in path.read_text()


def test_owned_report_closes_on_formatter_failure(tmp_path, monkeypatch):
    path = tmp_path / "report.txt"
    opened = []
    original = Path.open

    def track(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", track)
    with pytest.raises(ValueError, match="max_output"):
        run_stattab(
            io.StringIO(f"y\n{path}\n9\n1 0 1 ? .\n"), io.StringIO(), ask_report=True, max_output=5
        )
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize(
    "text,reason",
    [("n\n0\n", "exit"), ("y\nquit\n", "exit"), ("y\nback\n", "exit"), ("y\n", "eof")],
)
def test_application_optional_report_outcomes(text, reason):
    outcome = run_stattab(io.StringIO(text), io.StringIO(), ask_report=True)
    assert outcome.reason == reason and outcome.completed == 0
    with pytest.raises(ValueError):
        run_stattab(io.StringIO(), io.StringIO(), ask_report=True, report_stream=io.StringIO())


@pytest.mark.parametrize("hard_link", [False, True])
def test_file_selection_cannot_overwrite_its_active_input(tmp_path, hard_link):
    path = tmp_path / "commands"
    target = tmp_path / "alias" if hard_link else path
    text = (str(target) + "\n") * 3
    path.write_text(text)
    if hard_link:
        target.hardlink_to(path)
    with path.open() as source:
        c = CDFConsole(source, io.StringIO())
        with pytest.raises(CDFConsoleError):
            stattab_open_file(c, read=False)
        assert not source.closed
        assert "already used by this console" in c.output.getvalue()
    assert path.read_text() == text
