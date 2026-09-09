"""Independent mathematical and session-state checks classify STATTAB evidence."""

import json
import math
import re
from pathlib import Path

import pytest

REFERENCE = json.loads((Path(__file__).parent / "fixtures/stattab.json").read_text())
CASES = {c["name"]: c for c in REFERENCE["cases"]}
NUMBER = r"[+-]?\d+\.\d*(?:(?:[EeDd])?[+-]\d+)?"


def rows(name):
    result = []
    for line in CASES[name]["stdout"].splitlines():
        tokens = re.findall(NUMBER, line)
        if len(tokens) >= 4 and not re.sub(NUMBER, "", line).strip():
            result.append(
                [
                    float(re.sub(r"(?<=\d)([+-]\d+)$", r"e\1", s))
                    if not re.search("[EeDd]", s)
                    else float(s.replace("D", "e"))
                    for s in tokens
                ]
            )
    return result


def normal(x):
    return math.erfc(-x / math.sqrt(2)) / 2


def t5(x):
    theta = math.atan(x / math.sqrt(5))
    return 0.5 + (theta + 2 * math.sin(2 * theta) / 3 + math.sin(4 * theta) / 12) / math.pi


def nct5(x, nc, intervals=4096):
    # Condition on a chi(5) denominator. Its tail beyond 12 is below 1e-27.
    h = 12 / intervals

    def term(i):
        v = h * i
        return (
            normal(x * v / math.sqrt(5) - nc)
            * math.sqrt(2 / math.pi)
            / 3
            * v**4
            * math.exp(-v * v / 2)
        )

    return (
        h
        / 3
        * (
            term(0)
            + term(intervals)
            + math.fsum((4 if i % 2 else 2) * term(i) for i in range(1, intervals))
        )
    )


def probability(family, row):
    x = row[0]
    if family == "beta":
        assert row[2:4] == [2, 2]
        return x * x * (3 - 2 * x), 4
    if family == "chisq":
        assert row[1] == 2
        return -math.expm1(-x / 2), 2
    if family == "f":
        assert row[1:3] == [4, 4]
        z = x / (1 + x)
        return z * z * (3 - 2 * z), 3
    if family == "nc_f":
        assert row[1:4] == [4, 4, 2]
        z = x / (1 + x)
        # Poisson(1) mixture of Beta(2+j,2); omitted weights < 1e-150.
        terms = [
            math.exp(-1) / math.factorial(j) * z ** (2 + j) * (3 + j - (2 + j) * z)
            for j in range(100)
        ]
        return math.fsum(terms), 4
    if family == "gamma":
        assert row[1:3] == [3, 2]  # Printed order is shape, rate, unlike input.
        z = 2 * x
        return 1 - math.exp(-z) * (1 + z + z * z / 2), 3
    if family == "normal":
        assert row[1:3] == [0, 1]
        return normal(x), 3
    if family == "t":
        assert row[1] == 5
        return t5(x), 2
    if family == "nc_t":
        assert row[1:3] == [5, 0.5]
        p = nct5(x, 0.5)
        assert p == pytest.approx(nct5(x, 0.5, 8192), abs=2e-11)
        return p, 3
    raise AssertionError(family)


ORDINARY = [
    c
    for c in REFERENCE["cases"]
    if c.get("which") == 1
    and c["family"] in {"beta", "chisq", "f", "nc_f", "gamma", "normal", "t", "nc_t"}
    and c["name"] != "normal_boundary_2"
]


@pytest.mark.parametrize("case", ORDINARY, ids=lambda c: c["name"])
def test_native_ordinary_probabilities_against_independent_identities(case):
    values = rows(case["name"])
    assert values and case["returncode"] == 0
    for row in values:
        p, index = probability(case["family"], row)
        assert row[index] == pytest.approx(p, abs=5.1e-7)
        assert row[index + 1] == pytest.approx(1 - p, abs=5.1e-7)
        if case["family"] in {"normal", "t"}:
            assert row[-1] == pytest.approx(2 * min(p, 1 - p), abs=5.1e-7)


@pytest.mark.parametrize("name", ["binomial_which_1", "binomial_upper_query", "complement_chance"])
def test_native_integer_binomial_mass_and_cdf_are_exact_sums(name):
    s, n, p, q, cdf, ccdf, mass = rows(name)[0]
    terms = [math.comb(int(n), k) * p**k * q ** (int(n) - k) for k in range(int(s) + 1)]
    assert mass == pytest.approx(terms[-1], abs=5.1e-7)
    assert cdf == pytest.approx(math.fsum(terms), abs=5.1e-7)
    assert ccdf == pytest.approx(1 - math.fsum(terms), abs=5.1e-7)


def test_fractional_zero_count_native_mass_is_the_wrong_continuous_cdf():
    b = rows("binomial_boundary_0")[0]
    assert b[-1] == b[-3] and abs(b[-1] - 0.5**4) > 0.03
    p = rows("poisson_boundary_0")[0]
    expected_continuous = math.erfc(math.sqrt(3)) + 2 * math.sqrt(3 / math.pi) * math.exp(-3)
    assert p[-1] == pytest.approx(expected_continuous, abs=5.1e-7)
    assert abs(p[-1] - math.exp(-3)) > 0.05
    # Above one, STATTAB truncates the count (and binomial trial count).
    assert rows("binomial_boundary_1")[0][-1] == 6 / 16
    assert rows("poisson_boundary_1")[0][-1] == pytest.approx(4.5 * math.exp(-3), abs=5.1e-7)


def test_failed_negative_binomial_calls_print_zero_instead_of_a_valid_mass():
    row = rows("neg_binomial_which_1")[0]
    exact = math.comb(4, 2) * 0.5**5
    assert row[-1] == 0 and exact == 0.1875
    assert "BELOW the LOWER search bound" in CASES["neg_binomial_which_1"]["stdout"]


def test_false_success_inverses_violate_complements_and_exact_probabilities():
    beta = rows("beta_which_2")[0]
    assert beta[-2:] == [0.75, 0.5]
    assert sum(beta[-2:]) != 1 and 0.75**2 * (3 - 2 * 0.75) != beta[0]
    binomial = rows("binomial_which_4")[0]
    assert binomial[-2:] == [0, 1]
    assert binomial[0] == 11 / 16 and binomial[0] != 1  # P(S<=2 | n=4,p=0)=1.
    f = rows("f_which_2")[0]
    z = f[-1] / (1 + f[-1])
    assert abs(z * z * (3 - 2 * z) - f[0]) > 1e-4


def test_poisson_inverse_displays_continuous_and_adjacent_integer_answers():
    continuous, lower, upper = rows("poisson_which_2")
    assert lower[-1] == math.floor(continuous[-1]) == 2
    assert upper[-1] == lower[-1] + 1
    for row in (lower, upper):
        cdf = math.exp(-3) * sum(3**k / math.factorial(k) for k in range(int(row[-1]) + 1))
        assert row[0] == pytest.approx(cdf, abs=5.1e-7)


def test_reuse_preserves_normal_inputs_but_native_poisson_overwrites_previous_tail():
    first, second = rows("reuse")
    assert first[:3] == [1, 0, 1] and second[:3] == [2, 0, 1]
    p = rows("poisson_reuse_tail")
    assert p[0][2] == pytest.approx(8.5 * math.exp(-3), abs=5.1e-7)
    assert p[1][0] == pytest.approx(4 * math.exp(-3), abs=5.1e-7)
    assert p[1][0] != p[0][2]  # '=' reuses the internal count-minus-one tail.


def test_table_state_leaks_into_later_scalar_request():
    case = CASES["table_then_scalar"]
    assert len(rows(case["name"])) == 3  # The requested scalar x=3 is never computed.
    assert case["stdout"].count("The list currently contains 0 values.") == 2


@pytest.mark.parametrize(
    "name",
    [
        "bad_double_complement",
        "bad_missing_complement",
        "bad_two_queries",
        "bad_no_query",
        "bad_short_input",
        "bad_required_omission",
        "bad_two_tables",
    ],
)
def test_invalid_parameter_requests_are_rejected_without_answer_rows(name):
    assert not rows(name)
    assert "Bad input ... Please try again" in CASES[name]["stdout"]
    assert CASES[name]["returncode"] == 0


def test_native_input_bounds_eof_and_overflow_defects_remain_explicit():
    assert "above upper bound" in CASES["bad_extra_input"]["stderr"]
    assert CASES["bad_extra_input"]["returncode"] == 2
    assert "End of file" in CASES["parameter_eof"]["stderr"]
    assert CASES["parameter_eof"]["returncode"] == 2
    assert rows("bad_numeric_overflow")[0][0] == 0  # 1e999 was silently replaced.


def test_report_file_receives_the_same_computed_answer():
    c = CASES["report_file"]
    assert c["report"] is not None
    assert "1 0 1 ? ." in c["report"]
    answer = next(line for line in c["stdout"].splitlines() if "0.841345" in line)
    assert answer in c["report"]
