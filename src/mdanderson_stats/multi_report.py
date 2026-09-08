"""Readable tables for immutable MULTI session snapshots."""

from html import escape
from typing import Any, cast


def _format_report(report: dict[str, object], digits: int) -> str:
    """Render the internally generated version-1 report; not a report-file parser."""
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
        raise ValueError("digits must be an integer from 1 to 17")
    # The producer owns this schema. JSON-compatible nested records are narrowed
    # here, separately from numerical models and the public session interface.
    data = cast(dict[str, Any], report)
    lines = ["# MULTI analysis report", "", f"NumPy version: {data['numpy_version']}", ""]

    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, dict) and set(value) == {"nonfinite"}:
            value = value["nonfinite"]
        if isinstance(value, float):
            value = format(value, f".{digits}g")
        if isinstance(value, bool):
            value = "true" if value else "false"
        return (
            escape(str(value))
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", "")
            .replace("\n", "<br>")
        )

    def table(headers: list[str], rows: list[list[Any]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
        lines.append("")

    def details(label: str, value: Any) -> None:
        lines.extend([f"**{cell(label)}**", ""])
        if isinstance(value, dict) and {"null_weight", "weights", "a", "b"} <= value.keys():
            lines.extend([f"Beta components: {len(value['weights'])}", ""])
            rows = [[0, value["null_weight"], 1, 1]]
            rows += [
                [i + 1, w, a, b]
                for i, (w, a, b) in enumerate(
                    zip(value["weights"], value["a"], value["b"], strict=True)
                )
            ]
            table(["Component", "Proportion", "Beta a", "Beta b"], rows)
        elif isinstance(value, dict) and "nonfinite" not in value:
            simple = [
                [k, v]
                for k, v in value.items()
                if not isinstance(v, (dict, list)) or isinstance(v, dict) and "nonfinite" in v
            ]
            if simple:
                table(["Field", "Value"], simple)
            for key, child in value.items():
                if isinstance(child, (dict, list)) and not (
                    isinstance(child, dict) and "nonfinite" in child
                ):
                    details(f"{label}.{key}", child)
        elif isinstance(value, list):
            if all(
                not isinstance(v, (list, dict)) or isinstance(v, dict) and "nonfinite" in v
                for v in value
            ):
                table(["Index (1-based)", "Value"], [[i + 1, v] for i, v in enumerate(value)])
            else:
                for i, child in enumerate(value):
                    details(f"{label}[{i + 1}]", child)
        else:
            lines.extend([cell(value), ""])

    datasets = {d["id"]: d for d in data["datasets"]}
    for event in data["events"]:
        if event["event"] == "change_data":
            dataset = datasets[event["dataset"]]
            lines.extend([f"## Dataset {dataset['id']}: {cell(dataset['source'])}", ""])
            table(
                ["Rank", "Observation", "P-value"],
                [
                    [rank + 1, index + 1, dataset["entered_values"][index]]
                    for rank, index in enumerate(dataset["order"])
                ],
            )
            if dataset["warnings"]:
                table(
                    ["Line", "Column", "Token", "Diagnostic"],
                    [
                        [w["line"], w["column"], w["token"], w["reason"]]
                        for w in dataset["warnings"]
                    ],
                )
            continue
        if event["event"] == "set_seed":
            details("Random seed reset", event["state"])
            continue
        dataset = datasets[event["dataset"]]
        procedure = event["procedure"]
        lines.extend([f"## {procedure} — dataset {dataset['id']}", ""])
        details("Settings", event["settings"])
        if "null_estimate_source" in event:
            details("Null estimate source", event["null_estimate_source"])
        if event["status"] == "failed":
            details("FAILED", event["error"])
        else:
            result = event["result"]
            decision = procedure in (
                "multiple_testing",
                "sharpened_testing",
                "nonparametric_testing",
                "beta_mixture_testing",
            )
            if decision:
                raw = result if isinstance(result, dict) else {"reject": result}
                order = raw.get("order", dataset["order"])
                ranks = {index: rank + 1 for rank, index in enumerate(dataset["order"])}
                columns = [
                    (key, title)
                    for key, title in (
                        ("adjusted_pvalues", "Adjusted P-value"),
                        ("critical_values", "Critical alpha"),
                        ("scores", "Reciprocal-density score"),
                        ("log_density", "Log density"),
                    )
                    if raw.get(key) is not None
                ]
                rows = [
                    [
                        step + 1,
                        ranks[index],
                        index + 1,
                        dataset["entered_values"][index],
                        *[raw[key][index] for key, _ in columns],
                        "*" if raw["reject"][index] else "",
                    ]
                    for step, index in enumerate(order)
                ]
                table(
                    [
                        "Step",
                        "Rank",
                        "Observation",
                        "P-value",
                        *[title for _, title in columns],
                        "Reject",
                    ],
                    rows,
                )
                lines.extend(
                    [
                        f"Rejected: {sum(raw['reject'])} of {len(order)}. "
                        "An asterisk marks rejection.",
                        "",
                    ]
                )
                remaining = {
                    k: v
                    for k, v in raw.items()
                    if k not in {"order", "reject", *[key for key, _ in columns]}
                }
                if remaining:
                    details("Result diagnostics", remaining)
            else:
                details("Result", result)
        if "rng_after" in event:
            details("Random state after run", event["rng_after"])
    return "\n".join(lines)
