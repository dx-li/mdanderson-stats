"""CDFLIB constants; kind and unit integers are legacy identifiers, not Python handles."""

from typing import Final

# Kind codes from the audited GNU Fortran build. Numerical APIs use float64;
# these opaque identifiers are not NumPy dtype arguments or portable Fortran kinds.
dpkind: Final = 8
spkind: Final = 4
# Fortran logical units, not operating-system file descriptors.
stdin: Final = 5
stdout: Final = 6

eight: Final = 8.0
five: Final = 5.0
four: Final = 4.0
hundred: Final = 100.0
nine: Final = 9.0
one: Final = 1.0
seven: Final = 7.0
six: Final = 6.0
sixth: Final = one / six
ten: Final = 10.0
tenth: Final = one / ten
thousand: Final = 1000.0
thousandth: Final = one / thousand
three: Final = 3.0
twelve: Final = 12.0
two: Final = 2.0
zero: Final = 0.0
eighth: Final = one / eight
fifth: Final = one / five
fourth: Final = one / four
half: Final = one / two
hundredth: Final = one / hundred
third: Final = one / three

__all__ = [
    "dpkind",
    "spkind",
    "stdin",
    "stdout",
    "eight",
    "five",
    "four",
    "hundred",
    "nine",
    "one",
    "seven",
    "six",
    "sixth",
    "ten",
    "tenth",
    "thousand",
    "thousandth",
    "three",
    "twelve",
    "two",
    "zero",
    "eighth",
    "fifth",
    "fourth",
    "half",
    "hundredth",
    "third",
]
