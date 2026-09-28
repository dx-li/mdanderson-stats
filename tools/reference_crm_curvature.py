"""Independent decimal arithmetic for minus d² log L / d beta² at beta=0.

For a DLT, log L=exp(beta)*log(d), so information is -log(d).
For a non-DLT, differentiating log(1-d**exp(beta)) gives the expression below.
Decimal.from_float preserves each actual binary64 input, including the largest
representable skeleton below one. No package calculation is imported.
"""

import csv
from decimal import Decimal, localcontext
from pathlib import Path

probabilities = [
    1e-300,
    1e-100,
    1e-12,
    0.01,
    0.05,
    0.15,
    0.3,
    0.5,
    0.9,
    1 - 1e-6,
    1 - 1e-10,
    1 - 1e-14,
    float.fromhex("0x1.fffffffffffffp-1"),
]
output = Path("tests/fixtures/crm-prior-ess-curvature.csv")
with output.open("w", newline="") as file, localcontext() as context:
    context.prec = 90
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(["skeleton", "dlt", "information"])
    for probability in probabilities:
        d = Decimal.from_float(probability)
        t = -d.ln()
        no_dlt = d * t * (t - (1 - d)) / (1 - d) ** 2
        assert no_dlt > 0
        for outcome, information in ((0, no_dlt), (1, t)):
            writer.writerow([repr(probability), outcome, str(information)])
print("Saved 26 independent 90-digit curvature references.")
