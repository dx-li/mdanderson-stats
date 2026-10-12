"""Independent high-precision ToxFinder Hessian and archived-report checks.

Requires mpmath and NumPy. Reads the numeric executable fixtures; never imports
mdanderson_stats. Reports mathematical errors against original native results.
"""

import argparse
import csv
import io
import json
from pathlib import Path

import mpmath as mp
import numpy as np


def check_information(rows):
    mp.mp.dps = 40
    result = []
    for row in rows:
        parameters = row["fitted_modes"][0]
        determinants = []
        errors = []
        for outcome in (0, 1):

            def objective(*t):
                a1, a2, a3, b1, b2, b3 = t
                value = mp.mpf(0)
                for (x, y), z in zip(
                    row["doses"] + [row["candidate"]], row["outcomes"] + [outcome]
                ):
                    xp, yp = mp.mpf(x) ** b1, mp.mpf(y) ** b2
                    q = a1 * xp + a2 * yp + a3 * (xp * yp) ** b3
                    value += z * mp.log(q) - mp.log1p(q)
                return -value - sum((shape - 1) * mp.log(v) for shape, v in zip(row["shapes"], t))

            h = np.empty((6, 6))
            for i in range(6):
                for j in range(6):
                    order = [0] * 6
                    order[i] += 1
                    order[j] += 1
                    h[i, j] = float(mp.diff(objective, parameters, order))
            errors.append(float(abs(h - np.asarray(row["raw_hessians"][outcome])).max()))
            determinants.append(float(np.maximum(np.linalg.eigvalsh(h), 0.1).prod()))
        x, y = (mp.mpf(v) for v in row["candidate"])
        probabilities = []
        for a1, a2, a3, b1, b2, b3 in row["samples"]:
            q = a1 * x**b1 + a2 * y**b2 + a3 * (x**b1 * y**b2) ** b3
            probabilities.append(q / (1 + q))
        probability = float(mp.fsum(probabilities) / len(probabilities))
        information = (1 - probability) * determinants[0] + probability * determinants[1]
        result.append(
            {
                "candidate": row["candidate"],
                "maximum_hessian_error": max(errors),
                "relative_information_error": abs(information / row["native_information"] - 1),
            }
        )
    return result


def check_archive(row):
    mp.mp.dps = 60
    fields = next(csv.reader(io.StringIO(row["csv_row"])))
    while not fields[-1]:
        fields.pop()
    values = [mp.mpf(v) for v in fields]
    draws = [values[i : i + 6] for i in range(12, len(values), 6)]
    result = []
    for i, (x, y) in enumerate((values[j] / 1200, values[j + 1] / 600) for j in (1, 3, 5)):
        probabilities = []
        for a1, a2, a3, b1, b2, b3 in draws:
            q = a1 * x**b1 + a2 * y**b2 + a3 * (x**b1 * y**b2) ** b3
            probabilities.append(q / (1 + q))
        mean = mp.fsum(probabilities) / len(probabilities)
        sd = mp.sqrt(mp.fsum((p - mean) ** 2 for p in probabilities) / (len(probabilities) - 1))
        expected = row["recommendations"][i]
        result.append(
            {
                "posterior_mean": str(mean),
                "posterior_sd_sample": str(sd),
                "mean_reference_error": str(abs(mean - mp.mpf(expected["posterior_mean"]))),
                "sd_reference_error": str(abs(sd - mp.mpf(expected["posterior_sd_sample"]))),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    information = json.loads(
        (args.fixture_directory / "toxfinder-native-information.json").read_text()
    )
    archive = json.loads(
        (args.fixture_directory / "toxfinder-native-archived-summary.json").read_text()
    )
    result = {"information": check_information(information), "archive": check_archive(archive)}
    if any(
        r["maximum_hessian_error"] > 1e-12 or r["relative_information_error"] > 1e-12
        for r in result["information"]
    ):
        raise ArithmeticError("independent information check exceeded tolerance")
    if any(
        mp.mpf(r["mean_reference_error"]) > mp.mpf("1e-55")
        or mp.mpf(r["sd_reference_error"]) > mp.mpf("1e-55")
        for r in result["archive"]
    ):
        raise ArithmeticError("independent archived report check exceeded tolerance")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
