"""Generate independent 60-digit Decimal references from the recovered source tables.

No mdanderson_stats import; expressions and numerical constants come directly
from the parent-authored, text-and-visually audited published-table extraction.
Run from the repository root. The original article/supplement is not redistributed.
"""

import ast
import hashlib
import itertools
import json
from decimal import Decimal, localcontext
from pathlib import Path


def expression(node, inputs):
    if isinstance(node, ast.Name):
        return inputs[node.id]
    if isinstance(node, ast.Constant):
        return Decimal(str(node.value))
    if isinstance(node, ast.Set):
        return {expression(x, inputs) for x in node.elts}
    if isinstance(node, ast.BinOp):
        a, b = expression(node.left, inputs), expression(node.right, inputs)
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Div):
            return a / b
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        a, b = expression(node.left, inputs), expression(node.comparators[0], inputs)
        if isinstance(node.ops[0], ast.Eq):
            return Decimal(a == b)
        if isinstance(node.ops[0], ast.In):
            return Decimal(a in b)
    raise ValueError(ast.dump(node))


def main():
    source = Path("research/bless-published-parameters-audit.json").read_bytes()
    audit = json.loads(source)
    cases = []
    with localcontext() as ctx:
        ctx.prec = 60
        for model, spec in audit["models"].items():
            reference = {
                k: (v[0] if isinstance(v, list) else v) for k, v in spec["reference"].items()
            }
            inputs = [reference, spec["example"]["input"]]
            # Full categorical factorial, with reference and two nonreference
            # continuous vectors, covers every main effect and interaction.
            keys, choices = [], []
            for key in reference:
                keys.append(key)
                if key == "ecog":
                    choices.append(range(5))
                elif key == "cancer":
                    choices.append(("lung", "breast", "hematologic"))
                elif key in ("ldh_U_L", "protein_g_dL", "nlr"):
                    choices.append((reference[key],))
                else:
                    choices.append((0, 1))
            for values in itertools.product(*choices):
                for continuous in ((0, 0, 0), (100, 1, 2), (1700, 6.3, 31)):
                    x = dict(zip(keys, values))
                    for key, value in zip(("ldh_U_L", "protein_g_dL", "nlr"), continuous):
                        if key in x:
                            x[key] = value
                    inputs.append(x)
            for raw in inputs:
                x = {k: Decimal(str(v)) for k, v in raw.items() if k != "cancer"}
                if model == "gcmm":
                    x.update(
                        {c: Decimal(raw["cancer"] == c) for c in ("lung", "breast", "hematologic")}
                    )
                    x["unilateral"] = 1 - x["bilateral"]
                eta = sum(
                    Decimal(term["coefficient"])
                    * expression(ast.parse(term["term"], mode="eval").body, x)
                    for term in spec["terms"]
                )
                survival = [Decimal(s) ** eta.exp() for s in spec["baseline"]]
                normalized = {
                    k.replace("ldh_U_L", "ldh_u_l").replace("protein_g_dL", "protein_g_dl"): v
                    for k, v in raw.items()
                }
                cases.append(
                    {
                        "model": model,
                        "predictors": normalized,
                        "eta": str(eta),
                        "survival": [str(v) for v in survival],
                    }
                )
    result = {
        "source_audit_sha256": hashlib.sha256(source).hexdigest(),
        "method": "60-digit Decimal; AST evaluation of transcribed e-Tables 1–8 terms",
        "cases": cases,
    }
    Path("tests/fixtures/bless-published-decimal.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"Generated {len(cases)} independent Decimal references")


if __name__ == "__main__":
    main()
