"""Independent Decimal likelihood/derivative references for neural survival.

Uses explicit transformed observations and scalar probability formulas, with
85-digit arithmetic and central differences. No NumPy, SciPy, or package code
is imported. Run from the repository root to regenerate the JSON fixture.
"""

import json
from decimal import Decimal, localcontext
from pathlib import Path

D = Decimal
TIME = ["0", "0.4", "1", "1.7", "3.5"]
EVENT = [0, 1, 0, 1, 1]
CUTS = ["0", "1", "3"]
# Events round upward, censorings downward, exact cuts stay exact. The last
# observation is administratively censored at the last cut, despite event=1.
INDEX = [0, 1, 1, 2, 2]
TRANSFORMED_EVENT = [0, 1, 0, 1, 0]
LOGITS = [
    ["0.2", "-0.7", "1.1"],
    ["-1.2", "0.4", "0.1"],
    ["0.9", "-0.3", "0.6"],
    ["0", "0.8", "-0.6"],
    ["-1", "0.2", "0.5"],
]


def objective(family: str, z: list[list[Decimal]]) -> Decimal:
    n = D(len(z))
    total = D(0)
    if family == "loghaz":
        for row, event, index in zip(z, TRANSFORMED_EVENT, INDEX, strict=True):
            h = [D(1) / (D(1) + (-value).exp()) for value in row]
            probability = D(1)
            for j in range(index):
                probability *= 1 - h[j]
            probability *= h[index] if event else 1 - h[index]
            total -= probability.ln()
        return total / n
    if family == "deephit":
        probabilities = []
        for row in z:
            mass = [value.exp() for value in row] + [D(1)]
            denominator = sum(mass)
            probabilities.append([value / denominator for value in mass])
        for mass, event, index in zip(probabilities, TRANSFORMED_EVENT, INDEX, strict=True):
            probability = mass[index] if event else sum(mass[index + 1 :])
            total -= probability.ln()
        ranking = D(0)
        for i in range(len(z)):
            for j in range(len(z)):
                if TRANSFORMED_EVENT[i] and (
                    INDEX[i] < INDEX[j] or (INDEX[i] == INDEX[j] and not TRANSFORMED_EVENT[j])
                ):
                    f_i = sum(probabilities[i][: INDEX[i] + 1])
                    f_j = sum(probabilities[j][: INDEX[i] + 1])
                    ranking += ((f_j - f_i) / D("0.4")).exp()
        return D("0.2") * total / n + D("0.8") * ranking / (n * n)
    if family == "pchazard":
        # Integrated hazard increments: widths are NOT exposure multipliers.
        # Initial-cut censor has no data and is excluded from the source mean.
        exposures = [(D(0), D(0)), (D("0.4"), D(0)), (D(1), D(0)), (D(1), D("0.35")), (D(1), D(1))]
        interval = [None, 0, 0, 1, 1]
        for i in range(1, len(z)):
            increments = [(D(1) + value.exp()).ln() for value in z[i]]
            total += sum(
                value * fraction for value, fraction in zip(increments, exposures[i], strict=True)
            )
            if TRANSFORMED_EVENT[i]:
                total -= increments[interval[i]].ln()
        return total / D(4)
    raise ValueError(family)


def reference(family: str) -> dict:
    values = [row[:2] if family == "pchazard" else row[:] for row in LOGITS]
    z = [[D(value) for value in row] for row in values]
    step = D("1e-25")
    gradient = []
    for i, row in enumerate(z):
        derivatives = []
        for j, value in enumerate(row):
            z[i][j] = value + step
            above = objective(family, z)
            z[i][j] = value - step
            below = objective(family, z)
            z[i][j] = value
            derivatives.append(str((above - below) / (2 * step)))
        gradient.append(derivatives)
    return dict(
        family=family,
        time=TIME,
        event=EVENT,
        cuts=CUTS,
        logits=values,
        alpha="0.2",
        rank_sigma="0.4",
        loss=str(objective(family, z)),
        gradient=gradient,
    )


def cox_reference(family: str) -> dict:
    times = [D(x) for x in (1, 1, 2, 4, 4, 5)]
    events = [1, 0, 1, 1, 1, 0]
    x = [D(s) for s in ("-1", "-0.3", "0.2", "0.5", "1", "1.3")]
    first_layer = [[D("0.7"), D("-0.4")]]
    if family == "coxtime":
        first_layer.append([D("-0.3"), D("0.8")])
    first_layer.append([D("0.15"), D("-0.25")])
    weights = [first_layer, [[D("0.6")], [D("-0.7")], [D("0.1")]]]
    event_times = sorted({t for t, e in zip(times, events, strict=True) if e})

    def forward(value: Decimal, time: Decimal) -> Decimal:
        row = [value]
        if family == "coxtime":
            row.append((1 + time).ln())
        for layer_no, layer in enumerate(weights):
            inputs = row + [D(1)]
            row = [sum(a * layer[k][j] for k, a in enumerate(inputs)) for j in range(len(layer[0]))]
            if layer_no == 0:
                row = [max(D(0), a) for a in row]
        return row[0]

    def loss_baseline() -> tuple[Decimal, list[Decimal]]:
        loss, baseline = D(0), []
        for time in event_times:
            scores = [forward(value, time) for value in x]
            denominator = sum(
                score.exp() for score, t in zip(scores, times, strict=True) if t >= time
            )
            deaths = [
                i for i, (t, e) in enumerate(zip(times, events, strict=True)) if t == time and e
            ]
            loss += len(deaths) * denominator.ln() - sum(scores[i] for i in deaths)
            baseline.append(D(len(deaths)).ln() - denominator.ln())
        return loss / len(times), baseline

    step = D("1e-25")
    gradients = []
    for layer in weights:
        gradient = []
        for row in layer:
            derivatives = []
            for j, value in enumerate(row):
                row[j] = value + step
                above = loss_baseline()[0]
                row[j] = value - step
                below = loss_baseline()[0]
                row[j] = value
                derivatives.append(str((above - below) / (2 * step)))
            gradient.append(derivatives)
        gradients.append(gradient)
    loss, baseline = loss_baseline()
    profiles = [D("-0.7"), D("0.4"), D("1.1")]
    prediction_times = [D(s) for s in ("0", "0.5", "1", "1.5", "2", "4", "5", "10")]
    predictions = [
        [
            str(
                (
                    -sum(
                        (
                            (b + forward(value, t)).exp()
                            for t, b in zip(event_times, baseline, strict=True)
                            if t <= when
                        ),
                        D(0),
                    )
                ).exp()
            )
            for when in prediction_times
        ]
        for value in profiles
    ]
    return dict(
        family=family,
        time=[str(t) for t in times],
        event=events,
        x=[[str(value)] for value in x],
        weights=[[[str(w) for w in row] for row in layer] for layer in weights],
        loss=str(loss),
        gradient=gradients,
        event_times=[str(t) for t in event_times],
        baseline_log_hazard=[str(value) for value in baseline],
        prediction_x=[[str(value)] for value in profiles],
        prediction_times=[str(t) for t in prediction_times],
        prediction=predictions,
    )


if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 85
        cases = [reference(family) for family in ("loghaz", "deephit", "pchazard")]
        cox_cases = [cox_reference(family) for family in ("deepsurv", "coxtime")]
    Path("tests/fixtures/survival-neural-discrete.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
    Path("tests/fixtures/survival-neural-cox.json").write_text(
        json.dumps(cox_cases, indent=2) + "\n"
    )
