"""Captured-input statistical documents for standalone and complete desktop workflows."""

from html import escape

from .boin_desktop import read_boin_desktop_project
from .boin_desktop_pending import boin_desktop_pending_table
from .boin_desktop_protocol import boin_desktop_combination_protocol
from .boin_desktop_report import (
    _number,
    _table,
    boin_desktop_pending_table_html,
    boin_desktop_workflow_html,
)
from .boin_desktop_single import boin_desktop_single_counts, select_boin_desktop_single
from .boin_desktop_tite import boin_desktop_stft
from .boin_desktop_univariate_protocol import boin_desktop_univariate_protocol


def _document(study, title, body):
    if read_boin_desktop_project(study.xml) != study:
        raise ValueError("document inputs do not match saved XML")
    return study.to_html(
        "<!doctype html><html><head><meta charset='utf-8'>" + f"<title>{escape(title)}</title>"
        "<style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #aaa;padding:.4em}pre{white-space:pre-wrap}</style>"
        f"</head><body><h1>{escape(title)}</h1>"
        + body
        + "<details><summary>Complete saved inputs</summary><pre>"
        + escape(study.xml)
        + "</pre></details></body></html>"
    )


def boin_desktop_single_selection_html(study, *, outcomes_complete):
    """Recompute and report final single/TITE selection from the captured counts."""
    result = select_boin_desktop_single(study, outcomes_complete=outcomes_complete)
    n, y = boin_desktop_single_counts(study)
    rows = []
    for j in range(len(n)):
        rows.append(
            (
                j + 1,
                n[j],
                y[j],
                bool(result.eliminated[j]),
                _number(result.isotonic_mean[j]),
                _number(result.isotonic_interval[j, 0]),
                _number(result.isotonic_interval[j, 1]),
                _number(result.report_overdose_probability[j]),
                _number(result.safety_overdose_probability[j]),
            )
        )
    body = "<p>Final selected dose: " + escape(str(result.dose or "No selection")) + "</p>"
    body += _table(
        [
            "Dose",
            "Patients",
            "DLT",
            "Eliminated",
            "Isotonic mean",
            "Display lower",
            "Display upper",
            "Report overdose probability",
            "Safety overdose probability",
        ],
        rows,
    )
    body += (
        "<p>Complete outcomes explicitly confirmed. Marginal beta limits are transformed "
        "by weighted isotonic regression; they are not simultaneous credible bands. "
        "Report probabilities use the estimation prior; safety uses Beta(1,1). "
        "Untried doses have no estimates. Native near-tie floating-point identity is "
        "not claimed; final selection uses mathematical tie ordering.</p>"
    )
    return _document(study, "BOIN final single-agent selection", body)


def boin_desktop_stft_html(study):
    """Report per-patient saved days, analysis weights and their exact total."""
    result = boin_desktop_stft(study)
    body = _table(
        ["Pending patient", "Follow-up (days)", "Standardized weight"],
        [
            (i + 1, repr(float(t)), repr(float(w)))
            for i, (t, w) in enumerate(zip(result.followup_days, result.weights, strict=True))
        ],
    )
    body += (
        f"<p>Assessment window (days): {result.window_days!r}. STFT/WSTFT: {result.total!r}.</p>"
    )
    body += (
        "<p>Only pending patients contribute. Follow-up values are saved in days even "
        "when the assessment window is displayed in months. Use the current-dose "
        "patient/DLT/pending row of the applicable decision table; this total alone "
        "does not determine the next dose.</p>"
    )
    return _document(study, "BOIN standardized follow-up", body)


def _methods_html(markdown):
    """Render this package's authored headings, paragraphs and simple pipe tables."""
    parts = []
    for block in markdown.strip().split("\n\n"):
        lines = block.splitlines()
        if (
            len(lines) > 1
            and "|" in lines[0]
            and set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}
        ):
            parts.append(
                _table(
                    [v.strip() for v in lines[0].strip("|").split("|")],
                    [[v.strip() for v in line.strip("|").split("|")] for line in lines[2:]],
                )
            )
        elif block.startswith("#"):
            level = min(6, len(block) - len(block.lstrip("#")))
            parts.append(f"<h{level}>" + escape(block[level:].strip()) + f"</h{level}>")
        else:
            parts.append("<p>" + escape(block).replace("\n", "<br>") + "</p>")
    return "".join(parts)


def boin_desktop_protocol_html(workflow, *, language="en"):
    """Bind authored methods, decisions and operating characteristics to one study.

    Chinese methods and pending tables are supported; detailed simulation
    diagnostic labels remain English. No native proprietary template is copied.
    """
    if any(r.profile != "safe" for r in workflow.primary.results):
        raise ValueError(
            "protocol methods require safe simulation; use the legacy diagnostic report"
        )
    report = boin_desktop_workflow_html(workflow)
    study = workflow.primary.results[0].study
    if study.mode in ("single", "tite"):
        methods = boin_desktop_univariate_protocol(study, language=language)
    else:
        methods = boin_desktop_combination_protocol(study, language=language)
    body = _methods_html(methods)
    if study.mode == "tite":
        table = boin_desktop_pending_table(study)
        pending = boin_desktop_pending_table_html(study, table, language=language)
        body += pending.split("<body>", 1)[1].split("</body>", 1)[0]
    if study.mode in ("combination", "contour"):
        score = boin_desktop_score_table_html(study, language=language)
        body += score.split("<body>", 1)[1].split("</body>", 1)[0]
    title = (
        "Captured-input statistical methods and simulation"
        if language == "en"
        else "固定输入统计方法与模拟"
    )
    combined = report.split("<body>", 1)[1].split("</body>", 1)[0]
    # Retain one native marker envelope; embedded bodies contain no marker comments.
    return _document(study, title, body + "<hr>" + combined)


def boin_desktop_combination_selection_html(study, *, outcomes_complete, profile="safe"):
    """Report saved standalone combination/contour counts independently of patient history."""
    from .boin_desktop_final import (
        boin_desktop_combination_counts,
        select_boin_desktop_combination_counts,
    )

    result = select_boin_desktop_combination_counts(
        study, outcomes_complete=outcomes_complete, profile=profile
    )
    n, y = boin_desktop_combination_counts(study)
    selected = result.selection.contour if study.mode == "contour" else result.selection.dose
    body = f"<p>Profile: {escape(profile)}. Final selection: {escape(str(selected))}.</p>"
    if result.selected_eliminated:
        body += "<p><strong>Explicit legacy result selects an eliminated dose.</strong></p>"
    rows = []
    for a in range(study.shape[0]):
        for b in range(study.shape[1]):
            rows.append(
                (
                    (a + 1, b + 1),
                    n[a, b],
                    y[a, b],
                    bool(result.selection.eliminated[a, b]),
                    _number(result.reported_mean[a, b]),
                    _number(result.lower[a, b]),
                    _number(result.upper[a, b]),
                    _number(result.selection.safety_overdose_probability[a, b]),
                )
            )
    body += _table(
        [
            "Dose",
            "Patients",
            "DLT",
            "Eliminated",
            "Isotonic mean",
            "Display lower",
            "Display upper",
            "Safety overdose probability",
        ],
        rows,
    )
    body += (
        "<p>Complete outcomes explicitly confirmed. Display limits are rounded isotonic "
        "transforms of marginal beta quantiles, not simultaneous credible bands. Native "
        "Iso floating-point and near-tie identity are not claimed. These standalone counts "
        "are distinct from the saved patient conduct history.</p>"
    )
    return _document(study, "BOIN final combination selection", body)


def boin_desktop_score_table_html(study, *, profile="safe", language="en"):
    """Generate a captured-input score appendix, preserving ranking scope."""
    from .boin_desktop_score import boin_desktop_score_table

    if language not in ("en", "zh"):
        raise ValueError("language must be en or zh")
    table = boin_desktop_score_table(study, profile=profile)
    title = "Combination desirability scores" if language == "en" else "联合用药期望得分"
    body = f"<p>Profile: {escape(profile)}. Ranking maximum: {table.ranking_maximum}.</p>"
    body += _table(
        ["Patients", "DLT", "Interval mass", "Allocation score", "Rank"]
        if language == "en"
        else ["患者数", "毒性数", "区间概率", "分配得分", "排序"],
        [
            (
                r.patients,
                (">= " if r.eliminated_from else "") + str(r.toxicities),
                r.interval_probability,
                r.allocation_score,
                "E" if r.eliminated_from else r.rank,
            )
            for r in table.rows
        ],
    )
    note = (
        "Ranks use all cohort multiples through the planned maximum before rows above "
        "EarlyStop are hidden. E compresses all higher DLT counts at that patient count. "
        "Safe ranks use interval mass + .0005*n and strict safety cutoff; the explicit "
        "native table ranks raw mass and eliminates at equality. Native floating-point "
        "near-tie identity is not claimed. The table does not replace admissible-dose, "
        "sticky exclusion, run-in or lowest-dose stopping rules."
        if language == "en"
        else "排序先使用计划最大样本量内的全部队列人数，再隐藏超过 EarlyStop 的行。"
        "E 包含该人数下所有更多毒性事件的情况。"
        "安全模式按区间概率加 .0005*n 排序并使用严格安全阈值；"
        "原始表兼容模式按区间概率排序，在等于阈值时也剔除。"
        "不保证原始程序浮点近似相等时的排序一致。"
        "此表不能替代可选剂量、持续剔除、初始递增或最低剂量停止规则。"
    )
    return _document(study, title, body + "<p>" + escape(note) + "</p>")
