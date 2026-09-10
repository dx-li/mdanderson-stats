"""Independent English and Chinese statistical protocol text for BOIN designs."""

import numpy as np

from ._validation import count
from .boin import BOINDesign


def boin_protocol(
    design: BOINDesign,
    *,
    doses: int = 5,
    cohorts: int = 10,
    cohort_size: int = 3,
    start_dose: int = 1,
    titration: bool = False,
    titration_cap: int | None = None,
    language: str = "en",
) -> str:
    """Return Markdown methods text and a complete integer decision table.

    This generates statistical methods, not a complete clinical protocol or
    fabricated operating characteristics. Simulation results can be appended
    separately. Supports the site's maximum planned enrollment of 200 patients.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    if language not in ("en", "zh"):
        raise ValueError("language must be 'en' or 'zh'")
    sizes = count([doses, cohorts, cohort_size, start_dose], "protocol settings")
    if sizes.shape != (4,) or np.any(sizes < 1):
        raise ValueError("protocol settings must be positive integers")
    levels, nc, size, start = map(int, sizes)
    maximum = nc * size
    if not 2 <= levels <= 100 or start > levels or maximum > 200:
        raise ValueError("require 2..100 doses, a valid start dose and at most 200 patients")
    if not isinstance(titration, (bool, np.bool_)):
        raise ValueError("titration must be boolean")
    cap_value = count(levels if titration_cap is None else titration_cap, "titration_cap")
    if cap_value.ndim != 0 or not start <= cap_value <= levels:
        raise ValueError("titration_cap must be between starting and highest dose")
    if not titration and titration_cap is not None:
        raise ValueError("titration_cap requires titration=True")
    cap = int(cap_value)
    en = language == "en"

    def text(english: str, chinese: str) -> str:
        return english if en else chinese

    phi = repr(design.target)
    le, ld = (repr(v) for v in (design.escalation_boundary, design.deescalation_boundary))
    cutoff = repr(design.elimination_probability)
    parts = [text("# BOIN statistical methods", "# BOIN 统计方法")]
    parts.append(
        text(
            f"The Bayesian optimal interval (BOIN) design targets a DLT probability of {phi} "
            f"across {levels} ordered doses, starting at dose {start}. Maximum enrollment is "
            f"{maximum} patients ({nc} cohorts of {size}). Outcomes must be fully evaluable "
            "before each dose decision; this is not a time-to-event design.",
            f"本研究采用贝叶斯最优区间（BOIN）设计，目标剂量限制性毒性（DLT）概率为 {phi}，"
            f"共设 {levels} 个有序剂量水平，从第 {start} 剂量开始。最多入组 {maximum} 例"
            f"（{nc} 个队列，每队列 {size} 例）。"
            "每次剂量决策前须获得完整可评价的结局；本设计不处理待观察的毒性结局。",
        )
    )
    parts.append(
        text(
            f"At the current dose, escalate when cumulative DLTs/patients <= {le}; "
            f"de-escalate when DLTs/patients >= {ld}; otherwise stay. An unavailable move "
            "at a dose boundary becomes stay. Never escalate into an eliminated dose. "
            f"The corresponding indifference probabilities are {design.safe_probability!r} "
            f"and {design.toxic_probability!r}. Safety rules below take precedence.",
            f"当前剂量的累计 DLT 数/患者数不大于 {le} 时升量，不小于 {ld} 时降量，否则维持。"
            "如剂量边界使升降量不可行，则维持当前剂量。禁止升至已排除的剂量。"
            f"对应的备择毒性概率为 {design.safe_probability!r} 和 "
            f"{design.toxic_probability!r}。"
            "以下安全规则优先执行。",
        )
    )
    if design.stay_at_one_of_three:
        parts.append(
            text(
                "Modification: stay with exactly 1 DLT among 3 patients.",
                "修正规则：恰有 3 例患者且其中 1 例发生 DLT 时维持剂量。",
            )
        )
    if design.deescalate_at_two_of_six:
        parts.append(
            text(
                "Modification: de-escalate with exactly 2 DLTs among 6 patients.",
                "修正规则：恰有 6 例患者且其中 2 例发生 DLT 时降量。",
            )
        )
    parts.append(
        text(
            f"Use a Beta(1,1) safety prior. Once at least 3 patients have been treated at a "
            f"dose, eliminate it and all higher doses if Pr(p > {phi} | data) > {cutoff}. "
            "Exclusions persist. If the current dose is eliminated, move to the highest "
            "remaining dose; if the lowest dose is eliminated, stop without an MTD.",
            f"安全性分析采用 Beta(1,1) 先验。当某剂量至少已治疗 3 例且 "
            f"Pr(p > {phi} | 数据) > {cutoff} 时，排除该剂量及所有更高剂量，且不再恢复。"
            "当前剂量被排除时，降至尚未排除的最高剂量；最低剂量被排除时终止试验，不选择 MTD。",
        )
    )
    if design.extra_safe:
        extra = repr(design.elimination_probability - design.safety_offset)
        parts.append(
            text(
                f"Extra safety: after at least 3 patients at the lowest dose, also stop without "
                f"an MTD when its posterior overdose probability exceeds {extra}.",
                f"额外安全规则：最低剂量至少治疗 3 例后，若其后验超标概率大于 {extra}，"
                "则终止试验且不选择 MTD。",
            )
        )
    if design.early_stop_patients is not None:
        parts.append(
            text(
                f"Stop for precision once at least {design.early_stop_patients} patients are "
                "treated at the current dose and the resulting next assignment stays there, "
                "including when a dose boundary or exclusion prevents escalation. "
                "Then select the MTD.",
                f"当前剂量至少已治疗 {design.early_stop_patients} 例且下一次实际分配仍为该剂量时，"
                "按精度规则提前结束并选择 MTD；包括因剂量边界或排除规则无法升量而维持的情况。",
            )
        )
    if titration and size > 1 and start < levels:
        parts.append(
            text(
                f"Accelerated titration: escalate one patient per dose up to dose {cap}. "
                "At the first DLT, second grade-2 toxicity across titration patients, or the "
                f"highest dose, add {size - 1} patients at that dose before a BOIN decision. "
                "If a lower titration cap is reached without either toxicity trigger, begin "
                "a full cohort at the next dose. Thereafter use full cohorts, shortening the "
                "last cohort if necessary to respect the total enrollment cap.",
                f"加速滴定：每个剂量治疗 1 例并逐级升量，滴定上限为第 {cap} 剂量。出现首次 DLT、"
                "滴定患者中第二次 2 级毒性或到达最高剂量时，"
                f"在当前剂量补充 {size - 1} 例后按 BOIN 决策。"
                "如到达低于最高剂量的滴定上限且未触发上述毒性规则，则在下一剂量开始完整队列。此后采用完整队列，"
                "必要时缩短最后队列，以保证不超过总入组上限。",
            )
        )
    parts.append(
        text(
            "At trial completion, fit Beta(.05,.05) posterior means using increasing "
            "isotonic regression with inverse posterior-variance weights. Refit only "
            "treated, noneliminated doses for selection. Select the fitted mean closest "
            "to target; among tied closest doses choose the highest if all tied means "
            "are below target, otherwise the lowest. No eligible dose means no MTD.",
            "试验结束时，采用 Beta(.05,.05) 先验的后验均值，以"
            "后验方差的倒数为权重进行递增保序回归。"
            "选择时仅对已治疗且未排除的剂量重新拟合，选择拟合均值最接近目标的剂量。若距离并列，且所有并列均值"
            "均低于目标，则选择其中最高剂量；否则选择最低剂量。无符合条件的剂量时不选择 MTD。",
        )
    )
    if design.bound_mtd:
        parts.append(
            text(
                f"Further restrict MTD selection to fitted toxicity <= {ld}.",
                f"MTD 选择还须满足拟合毒性概率不大于 {ld}。",
            )
        )
    parts.append(
        text(
            "## Decision table\n\nSafety takes precedence. Cutoffs are inclusive DLT counts; "
            "a dash means the safety trigger is impossible at that sample size. The last "
            "column applies only at the lowest dose.\n\n"
            "| Patients at dose | Escalate at most | De-escalate at least | "
            "Eliminate at least | Lowest-dose stop at least |",
            "## 决策表\n\n安全规则优先。界值均为包含端点的 DLT 数；"
            "横线表示该样本量下不可能触发该安全规则。"
            "最后一列仅用于最低剂量。\n\n"
            "| 当前剂量例数 | 升量至多 | 降量至少 | 排除至少 | 最低剂量终止至少 |",
        )
    )
    rows = ["| ---: | ---: | ---: | ---: | ---: |"]
    table = design.boundary_table(maximum)
    for n, e, d, eliminate, stop in zip(
        table.patients,
        table.escalate_max,
        table.deescalate_min,
        table.eliminate_min,
        table.lowest_stop_min,
        strict=True,
    ):
        rows.append(
            f"| {n} | {e} | {d} | {eliminate if eliminate <= n else '—'} | "
            f"{stop if stop <= n else '—'} |"
        )
    parts[-1] += "\n" + "\n".join(rows)
    parts.append(
        text(
            "Reference: Yan et al. (2020), BOIN: An R Package for Designing Single-Agent "
            "and Drug-Combination Dose-Finding Trials Using Bayesian Optimal Interval Designs. "
            "https://doi.org/10.18637/jss.v094.i13",
            "参考文献：Yan 等（2020），BOIN 软件包及贝叶斯最优区间剂量探索设计。https://doi.org/10.18637/jss.v094.i13",
        )
    )
    return "\n\n".join(parts) + "\n"
