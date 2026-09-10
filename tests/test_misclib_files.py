from io import StringIO

import pytest

from mdanderson_stats import CDFConsole, misclib_open_file


def choose(text, **options):
    return misclib_open_file(console=CDFConsole(StringIO(text), StringIO()), **options)


def test_create_read_append_and_overwrite_real_files(tmp_path):
    path = tmp_path / "report with spaces.txt"
    result = choose(f'"{path}" comment\np\n', delimiter="QUOTE")
    assert result.status == 0 and result.path == path and result.action == "create"
    assert result.delimiter == "quote" and result.stream is not None
    with result.stream as stream:
        stream.write("first\n")
    result = choose(f'"{path}"\np\n', read_only=True)
    assert result.stream is not None
    with result.stream as stream:
        assert stream.read() == "first\n"
        assert not stream.writable()
    result = choose(f'"{path}"\na\np\n')
    assert result.stream is not None and result.action == "append"
    with result.stream as stream:
        stream.write("second\n")
    assert path.read_text() == "first\nsecond\n"
    result = choose(f'"{path}"\no\np\n', appendable=False)
    assert result.stream is not None and result.action == "overwrite"
    with result.stream as stream:
        assert stream.read() == ""
        stream.write("replacement")
    assert path.read_text() == "replacement"


def test_cancel_and_retry_do_not_mutate_files(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("valuable contents")
    result = choose(f"{path}\no\nq\n")
    assert result.status == 2 and result.stream is None
    assert path.read_text() == "valuable contents"
    new_path = tmp_path / "new.txt"
    result = choose(f"{new_path}\nr\nBACK\n")
    assert result.status == 3 and not new_path.exists()
    result = choose(f"{path}\nr\nquit\n")
    assert result.status == 2 and path.read_text() == "valuable contents"
    assert choose("quit\n").status == 2
    result = choose("\n\n\n")
    assert result.status == 1 and result.error is not None


def test_failed_open_and_invalid_input_have_explicit_outcomes(tmp_path):
    absent = tmp_path / "missing"
    result = choose(f"{absent}\n{absent}\n{absent}\n", read_only=True)
    assert result.status == 1 and result.error is not None and not absent.exists()
    # Opening a directory as a text file fails; a later command can exit normally.
    result = choose(f"{tmp_path}\np\nback\n", read_only=True)
    assert result.status == 3
    with pytest.raises(ValueError):
        choose("", delimiter="bad")
    with pytest.raises(LookupError):
        choose("", encoding="unknown-misclib-codec")
    with pytest.raises(EOFError):
        choose("")
