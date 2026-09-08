"""Defined Fortran/C PHRTSD character mappings, with unsafe C reads rejected."""

from .ranlist_random import _phrase_seed_codes, ranlist_seeds


def phrase_seeds(phrase: str, source: str) -> tuple[int, int]:
    if not isinstance(source, str) or source not in ("fortran", "c"):
        raise ValueError("source must be fortran or c")
    if source == "fortran":
        return ranlist_seeds(phrase)
    if not isinstance(phrase, str) or not phrase.isascii():
        raise ValueError("phrase must be an ASCII string")
    table = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+[];:'"
        + '\\"'
        + "<>?,./"
    )
    phrase = phrase.rstrip(" ")
    if any(character not in table for character in phrase):
        raise ValueError("C PHRTSD reads beyond its table for unknown characters")
    # C increments the matching index before testing for the terminating NUL;
    # the last table character therefore gets fallback code 63.
    return _phrase_seed_codes(
        (table.index(c) + 1) % 64 or 63 if c != table[-1] else 63 for c in phrase
    )
