"""STATTAB's 42 computed groups, source columns and independent probability checks."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from test_stattab_reference import nct5, normal, probability, rows, t5

from mdanderson_stats import STATTAB_DISTRIBUTIONS, stattab_solve

# Source menu inputs; gamma A is a rate. Independent probabilities below use
# these particular shapes, not an alternative call to the production kernels.
INPUTS = {
    "beta": dict(x=0.5, cx=0.5, a=2, b=2),
    "binomial": dict(s=2, n=4, pr=0.5, cpr=0.5),
    "neg_binomial": dict(f=2, s=3, pr=0.5, cpr=0.5),
    "chisq": dict(x=2, df=2),
    "nc_chisq": dict(x=4, df=2, pnonc=2),
    "f": dict(f=1, dfn=4, dfd=4),
    "nc_f": dict(f=2, dfn=4, dfd=4, pnonc=2),
    "gamma": dict(x=1, rate=2, shape=3),
    "normal": dict(x=1, mean=0, sd=1),
    "poisson": dict(s=2, mean=3),
    "t": dict(t=1, df=5),
    "nc_t": dict(t=1, df=5, pnonc=0.5),
}


def nc_chisq4():
    # Poisson(1) mixture of chi-square(2+2j), evaluated by integer gamma sums.
    return math.fsum(
        math.exp(-1)
        / math.factorial(j)
        * (1 - math.exp(-2) * math.fsum(2**k / math.factorial(k) for k in range(j + 1)))
        for j in range(40)
    )


TARGETS = {
    "beta": 0.5,
    "binomial": 11 / 16,
    "neg_binomial": 0.5,
    "chisq": -math.expm1(-1),
    "nc_chisq": nc_chisq4(),
    "f": 0.5,
    "nc_f": probability("nc_f", [2, 4, 4, 2])[0],
    "gamma": 1 - 5 * math.exp(-2),
    "normal": normal(1),
    "poisson": 8.5 * math.exp(-3),
    "t": t5(1),
    "nc_t": nct5(1, 0.5, 8192),
}
# Literal source output columns for each computed group, in kernel selector order.
COLUMNS = {
    "beta": ["x cx a b cum ccum", "cum ccum a b x cx", "cum ccum x cx b a", "cum ccum x cx a b"],
    "binomial": [
        "s n pr cpr cum ccum term",
        "cum ccum n pr cpr s",
        "cum ccum s pr cpr n",
        "cum ccum s n pr cpr",
    ],
    "neg_binomial": [
        "f s pr cpr cum ccum term",
        "cum ccum s pr cpr f",
        "cum ccum f pr cpr s",
        "cum ccum f s pr cpr",
    ],
    "chisq": ["x df cum ccum", "cum ccum df x", "cum ccum x df"],
    "nc_chisq": [
        "x df pnonc cum ccum",
        "cum ccum df pnonc x",
        "cum ccum x pnonc df",
        "cum ccum x df pnonc",
    ],
    "f": ["f dfn dfd cum ccum", "cum ccum dfn dfd f"],
    "nc_f": ["f dfn dfd pnonc cum ccum", "cum ccum dfn dfd pnonc f", "cum ccum f dfn dfd pnonc"],
    "gamma": [
        "x shape rate cum ccum",
        "cum ccum shape rate x",
        "cum ccum x rate shape",
        "cum ccum x shape rate",
    ],
    "normal": [
        "x mean sd cum ccum two_sided_p",
        "cum ccum mean sd x",
        "cum ccum x sd mean",
        "cum ccum x mean sd",
    ],
    "poisson": ["s mean cum ccum term", "cum ccum mean s", "cum ccum s mean"],
    "t": ["t df cum ccum two_sided_p", "cum ccum df t", "cum ccum t df"],
    "nc_t": [
        "t df pnonc cum ccum",
        "cum ccum df pnonc t",
        "cum ccum t pnonc df",
        "cum ccum t df pnonc",
    ],
}
GROUPS = [
    (name, group, which)
    for name, descriptor in STATTAB_DISTRIBUTIONS.items()
    for which, group in enumerate(descriptor.groups)
]


def test_source_menu_and_all_42_groups():
    assert tuple(STATTAB_DISTRIBUTIONS) == tuple(INPUTS)
    assert [d.menu for d in STATTAB_DISTRIBUTIONS.values()] == list(range(1, 13))
    assert len(GROUPS) == 42
    assert STATTAB_DISTRIBUTIONS["gamma"].parameters == ("x", "rate", "shape", "cum", "ccum")


@pytest.mark.parametrize("name,group,which", GROUPS)
def test_every_group_recovers_independent_known_parameters_in_source_columns(name, group, which):
    expected = INPUTS[name] | dict(cum=TARGETS[name], ccum=1 - TARGETS[name])
    inputs = {k: v for k, v in expected.items() if k not in group}
    kwargs = dict(df_bracket=(4.9, 5.1)) if name == "nc_t" and group == ("df",) else {}
    result = stattab_solve(name, compute=group[0], **inputs, **kwargs)
    assert result.computed == group
    assert result.columns == tuple(COLUMNS[name][which].split())
    assert tuple(result.parameters) == STATTAB_DISTRIBUTIONS[name].parameters
    for key, target in expected.items():
        assert float(result.parameters[key]) == pytest.approx(target, rel=2e-8, abs=2e-9)
    for i, key in enumerate(result.columns):
        if key in expected:
            assert result.values[i] == result.parameters[key]
    assert bool(result.neighbors) == (
        name in ("binomial", "neg_binomial") and which in (1, 2) or name == "poisson" and which == 1
    )


@pytest.mark.parametrize("name", INPUTS)
def test_forward_native_sessions_and_independent_extra_columns(name):
    result = stattab_solve(name, **INPUTS[name])
    native = rows(name + "_which_1")[0]
    # Native negative-binomial stale status leaves its final term at zero.
    # Native noncentral chi-square failed with uninitialized/stale tails.
    end = 3 if name == "nc_chisq" else (-1 if name == "neg_binomial" else None)
    np.testing.assert_allclose(result.values[:end], native[:end], rtol=0, atol=5.1e-7)
    if name in ("normal", "t"):
        assert result.values[-1] == pytest.approx(2 * (1 - TARGETS[name]), rel=2e-13)
    elif name in ("binomial", "neg_binomial", "poisson"):
        target = {"binomial": 0.375, "neg_binomial": 0.1875, "poisson": 4.5 * math.exp(-3)}[name]
        assert result.values[-1] == pytest.approx(target, rel=3e-13)
    else:
        assert len(result.columns) == len(result.parameters)


@pytest.mark.parametrize("name", INPUTS)
def test_broadcast_batches_and_empty_results(name):
    key = next(iter(INPUTS[name]))
    inputs = INPUTS[name] | {key: np.array([[0.25], [0.5]])}
    second = list(INPUTS[name])[1]
    # Preserve both beta coordinates rather than giving inconsistent complements.
    if name == "beta":
        inputs.pop("cx")
        second = "a"
    inputs[second] = np.array([INPUTS[name][second], INPUTS[name][second] + 1])
    result = stattab_solve(name, **inputs)
    assert result.values.shape == (2, 2, len(result.columns))
    for i, x in enumerate([0.25, 0.5]):
        scalar = INPUTS[name] | {key: x}
        if name == "beta":
            scalar.pop("cx")
        for j in range(2):
            scalar[second] = INPUTS[name][second] + j
            np.testing.assert_array_equal(result.values[i, j], stattab_solve(name, **scalar).values)
    inputs[key] = np.empty((0, 1))
    assert stattab_solve(name, **inputs).values.shape == (0, 2, len(result.columns))


def test_complement_queries_and_tiny_two_sided_probabilities():
    direct = stattab_solve("beta", cx=0.25, a=2, b=2)
    upper = stattab_solve("beta", compute="ccum", cx=0.25, a=2, b=2)
    np.testing.assert_array_equal(direct.values, upper.values)
    inv = stattab_solve("beta", compute="cx", a=2, b=2, cum=0.5)
    assert tuple(inv.parameters[k] for k in ("x", "cx")) == (0.5, 0.5)
    for name, kwargs, key in [("normal", dict(mean=0, sd=1), "x"), ("t", dict(df=5), "t")]:
        a = stattab_solve(name, **{key: np.array([-9.0, 0.0, 9.0])}, **kwargs)
        assert a.values[0, -1] == a.values[2, -1]
        assert a.values[1, -1] == 1
        inv = stattab_solve(name, compute=key, ccum=1e-20, **kwargs)
        out = stattab_solve(
            name, **{k: v for k, v in inv.parameters.items() if k not in ("cum", "ccum")}
        )
        assert out.values[-1] == pytest.approx(2e-20, rel=1e-12)


def test_zero_mean_and_fractional_counts_do_not_overwrite_cdf():
    result = stattab_solve("poisson", s=[0, 0.5, 2], mean=[0, 3, 3])
    assert result.values[0].tolist() == [0, 0, 1, 0, 1]
    assert result.values[1, -1] == pytest.approx(math.exp(-3))
    assert result.parameters["cum"][2] == pytest.approx(8.5 * math.exp(-3))
    assert result.parameters["cum"][2] != pytest.approx(4 * math.exp(-3))
    assert stattab_solve("neg_binomial", f=0, s=0, pr=0).values[-1] == 1


def test_count_neighbors_preserve_continuous_solution_and_mask_invalid_rows():
    forward = stattab_solve("binomial", s=[2.4, 1.4], n=[2.5, 4], pr=0.5)
    inverse = stattab_solve(
        "binomial", compute="s", n=[2.5, 4], pr=0.5, cum=forward.parameters["cum"]
    )
    np.testing.assert_allclose(inverse.parameters["s"], [2.4, 1.4], rtol=2e-8)
    lower, upper = inverse.neighbors
    assert lower.valid.tolist() == [True, True]
    assert upper.valid.tolist() == [False, True]
    assert upper.source_indices.tolist() == [1]
    assert lower.values[:, -1].tolist() == [2, 1]
    assert upper.values[:, -1].tolist() == [2]
    assert upper.values[0, 0] == pytest.approx(11 / 16)
    # Inverting trial count can instead put the lower neighbor below successes.
    forward = stattab_solve("binomial", s=2.5, n=2.75, pr=0.5)
    inverse = stattab_solve("binomial", compute="n", s=2.5, pr=0.5, cum=forward.parameters["cum"])
    assert not inverse.neighbors[0].valid
    assert inverse.neighbors[0].values.shape == (0, 6)
    assert inverse.neighbors[1].valid
    # Native always emits floor and floor+1, even when the solution is integral.
    inverse = stattab_solve("binomial", compute="s", n=0, pr=0.5, cum=1)
    assert inverse.neighbors[0].values[0, -1] == 0
    assert not inverse.neighbors[1].valid


@pytest.mark.parametrize(
    "name,key",
    [
        ("binomial", "s"),
        ("binomial", "n"),
        ("neg_binomial", "f"),
        ("neg_binomial", "s"),
        ("poisson", "s"),
    ],
)
def test_all_five_count_inversions_have_independently_checked_neighbor_tails(name, key):
    base = INPUTS[name] | {key: INPUTS[name][key] + 0.25}
    forward = stattab_solve(name, **base)
    inverse = stattab_solve(
        name,
        compute=key,
        **{k: v for k, v in base.items() if k != key},
        cum=forward.parameters["cum"],
    )
    for neighbor in inverse.neighbors:
        assert neighbor.valid
        row = dict(zip(inverse.columns, neighbor.values[0], strict=True))
        if name == "poisson":
            target = math.exp(-row["mean"]) * math.fsum(
                row["mean"] ** i / math.factorial(i) for i in range(int(row["s"]) + 1)
            )
        elif name == "binomial":
            # Other count stays integral in these cases.
            target = math.fsum(
                math.comb(int(row["n"]), i) * row["pr"] ** i * row["cpr"] ** (int(row["n"]) - i)
                for i in range(int(row["s"]) + 1)
            )
        else:
            target = math.fsum(
                math.comb(i + int(row["s"]) - 1, i) * row["pr"] ** int(row["s"]) * row["cpr"] ** i
                for i in range(int(row["f"]) + 1)
            )
        assert row["cum"] == pytest.approx(target, rel=3e-13)
        assert row["ccum"] == pytest.approx(1 - target, rel=3e-13)


def test_noncentral_t_two_root_brackets():
    result = stattab_solve(
        "nc_t", compute="df", t=1, pnonc=3, cum=0.03, df_bracket=([0.001, 0.2], [0.2, 100])
    )
    assert result.parameters["df"][0] < 0.2 < result.parameters["df"][1]
    forward = stattab_solve("nc_t", t=1, pnonc=3, df=result.parameters["df"])
    np.testing.assert_allclose(forward.parameters["cum"], 0.03, rtol=2e-8)


def test_result_ownership_and_descriptor_immutability():
    values = np.array([1.0, 2.0])
    result = stattab_solve("poisson", s=values, mean=3)
    values[:] = 9
    assert result.parameters["s"].tolist() == [1, 2]
    for array in [result.values, *result.parameters.values()]:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.parameters["s"] = values
    with pytest.raises(TypeError):
        STATTAB_DISTRIBUTIONS["normal"] = None
    with pytest.raises(FrozenInstanceError):
        result.distribution = "normal"
    inverse = stattab_solve("poisson", compute="s", mean=3, cum=0.5)
    for neighbor in inverse.neighbors:
        for array in [neighbor.values, neighbor.valid, neighbor.source_indices, neighbor.candidate]:
            with pytest.raises(ValueError):
                array.setflags(write=True)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("bad", {}),
        ("normal", dict(compute="bad")),
        ("normal", dict(compute=1)),
        ("normal", dict(x=1)),
        ("normal", dict(x=1, mean=None, sd=1)),
        ("normal", dict(x=1, mean=0, sd=1, cum=0.5)),
        ("normal", dict(x=1, mean=0, sd=1, extra=2)),
        ("normal", dict(x=[1, 2], mean=[0, 1, 2], sd=1)),
        ("normal", dict(x=math.inf, mean=0, sd=1)),
        ("normal", dict(x=1, mean=0, sd=-1)),
        ("normal", dict(x=1, mean=0, sd=1, df_bracket=(1, 2))),
        ("f", dict(compute="dfn", f=1, dfd=4, cum=0.5)),
        ("nc_f", dict(compute="dfd", f=1, dfn=4, pnonc=2, cum=0.5)),
        ("poisson", dict(s=0, mean=-1)),
        ("poisson", dict(s=0, mean=1e-11)),
        ("poisson", dict(s=1e101, mean=0)),
        ("poisson", dict(compute="s", mean=0, cum=0.5)),
    ],
)
def test_invalid_requests_fail_explicitly(name, kwargs):
    with pytest.raises(ValueError):
        stattab_solve(name, **kwargs)


@pytest.mark.parametrize("name,key", [(n, k) for n, inputs in INPUTS.items() for k in inputs])
def test_table_at_every_forward_input_position_including_complements(name, key):
    inputs = INPUTS[name].copy()
    for pair in [("x", "cx"), ("pr", "cpr")]:
        if key in pair:
            inputs.pop(next(k for k in pair if k != key), None)
    base = inputs[key]
    table = np.array([base * 0.75, base])
    inputs[key] = table
    result = stattab_solve(name, **inputs)
    for i, value in enumerate(table):
        np.testing.assert_array_equal(
            result.values[i], stattab_solve(name, **(inputs | {key: value})).values
        )


@pytest.mark.parametrize("name", INPUTS)
@pytest.mark.parametrize("tail", ["cum", "ccum"])
def test_inverse_table_at_each_probability_position(name, tail):
    key = STATTAB_DISTRIBUTIONS[name].groups[1][0]
    inputs = {
        k: v for k, v in INPUTS[name].items() if k not in STATTAB_DISTRIBUTIONS[name].groups[1]
    }
    targets = np.array([TARGETS[name], TARGETS[name] * 0.9])
    if tail == "ccum":
        targets = 1 - targets
    result = stattab_solve(name, compute=key, **inputs, **{tail: targets})
    for i, target in enumerate(targets):
        np.testing.assert_array_equal(
            result.values[i], stattab_solve(name, compute=key, **inputs, **{tail: target}).values
        )


def test_unrepresentable_next_integer_is_not_reported_as_a_duplicate_row():
    result = stattab_solve("poisson", s=1e100, mean=1e100)
    inverse = stattab_solve(
        "poisson",
        compute="s",
        mean=1e100,
        cum=result.parameters["cum"],
        ccum=result.parameters["ccum"],
    )
    assert inverse.parameters["s"] == 1e100
    assert inverse.neighbors[0].valid
    assert not inverse.neighbors[1].valid
    assert inverse.neighbors[1].values.shape == (0, 4)
    assert inverse.neighbors[1].source_indices.size == 0


def test_forward_probabilities_match_independent_values_at_tighter_precision():
    for name, inputs in INPUTS.items():
        result = stattab_solve(name, **inputs)
        assert result.parameters["cum"] == pytest.approx(TARGETS[name], rel=3e-13, abs=3e-14)
        assert result.parameters["ccum"] == pytest.approx(1 - TARGETS[name], rel=3e-13, abs=3e-14)


@pytest.mark.parametrize("s,pr", [(0, 0.25), (0, 0.75), (3, 0.25), (3, 0.75)])
@pytest.mark.parametrize("tail", ["cum", "ccum"])
def test_binomial_chance_inverse_across_native_branch_conditions(s, pr, tail):
    # The archived program chooses its branch from C <= 1/2 and C(1/2) <= C.
    # These four cases exercise every combination. Three native branches omit
    # output assignments; verify the mathematical answer, not native garbage.
    cum = math.fsum(math.comb(4, k) * pr**k * (1 - pr) ** (4 - k) for k in range(s + 1))
    result = stattab_solve(
        "binomial", compute="pr", s=s, n=4, **{tail: cum if tail == "cum" else 1 - cum}
    )
    assert result.parameters["pr"] == pytest.approx(pr, rel=2e-13)
    assert result.parameters["cpr"] == pytest.approx(1 - pr, rel=2e-13)
