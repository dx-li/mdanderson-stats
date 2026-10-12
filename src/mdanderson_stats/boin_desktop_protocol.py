"""Independent bilingual methods text for saved BOIN combination studies."""

import numpy as np

from .boin_desktop import BOINDesktopStudy
from .boin_desktop_controller import desktop_combination_design


def boin_desktop_combination_protocol(study: BOINDesktopStudy, *, language: str = "en") -> str:
    """Describe safe conduct and simulation separately, with integer boundaries.

    This is statistical methods text, not a complete clinical protocol, source
    template reproduction, or an endorsement of clinical validity.
    """
    if language not in ("en", "zh"):
        raise ValueError("language must be en or zh")
    design = desktop_combination_design(study)
    en = language == "en"

    def text(english, chinese):
        return english if en else chinese

    maximum = sum(study.subtrial_maximums) if study.mode == "contour" else study.maximum_patients
    table = design.boundary_table(maximum)
    parts = [text("# BOIN combination statistical methods", "# BOIN 联合用药统计方法")]
    parts.append(
        text(
            f"The {study.shape[0]} by {study.shape[1]} ordered dose grid "
            f"targets DLT probability "
            f"{study.target}. Cohorts contain {study.cohort_size} patients; "
            f"starting dose is "
            f"{study.start_dose}. Planned enrollment is at most {maximum}. "
            f"Mode: {study.mode}.",
            f"采用 {study.shape[0]} × {study.shape[1]} 有序剂量组合，目标 DLT 概率为 "
            f"{study.target}。"
            f"每队列 {study.cohort_size} 例，起始剂量为 {study.start_dose}，最多入组 "
            f"{maximum} 例。"
            f"模式：{study.mode}。",
        )
    )
    parts.append(
        text(
            f"Indifference probabilities are {design.safe_probability} and "
            f"{design.toxic_probability}. "
            f"The escalation/de-escalation probability boundaries are "
            f"{design._base.escalation_boundary:.12g} and "
            f"{design._base.deescalation_boundary:.12g}. "
            "The integer table below takes precedence when run-in "
            "modifications apply.",
            f"无差异区间参数为 {design.safe_probability} 和 {design.toxic_probability}，"
            f"升降剂量边界为 {design._base.escalation_boundary:.12g} 和 "
            f"{design._base.deescalation_boundary:.12g}。启用导入规则时以下整数表优先。",
        )
    )
    parts.append(
        text(
            f"Safety uses Pr(p > {study.target} | y,n) under Beta(1,1), with "
            f"elimination cutoff "
            f"{design.elimination_probability}. At least three observations "
            f"are needed. "
            "Elimination also excludes combinations at least as high in both "
            "drugs. "
            f"Extra-safe stopping: {design.extra_safe}; offset: "
            f"{design.safety_offset}. "
            "Safe conduct never assigns an eliminated combination.",
            f"安全性采用 Beta(1,1) 先验下 Pr(p > {study.target} | y,n)，排除阈值为 "
            f"{design.elimination_probability}，至少需三例观察。两种药物剂量均不低于被排除组合"
            f"的组合也予以排除。额外安全规则：{design.extra_safe}；偏移量：{design.safety_offset}。"
            "安全模式不分配已排除的组合。",
        )
    )
    parts.append(text("## Conduct flow", "## 实施流程"))
    for english, chinese in (
        (
            "1. Check actual enrollment against capacity before adding a patient.",
            "1. 添加患者前检查实际入组人数上限。",
        ),
        (
            "2. Complete the current cohort at its assigned dose. At a cohort "
            "boundary, wait until all pending outcomes resolve.",
            "2. 在已分配剂量完成当前队列；队列边界处等待全部待观察结局确定。",
        ),
        (
            "3. Apply safety and dose boundaries to cumulative completed "
            "outcomes. Stop for convergence only when the precision sample "
            "size is reached and no move is indicated or available.",
            "3. 根据累计完整结局应用安全与剂量边界；"
            "达到精度样本量且无需或无法升降量时才因收敛停止。",
        ),
        (
            "4. At completion, select only treated, noneliminated "
            "combinations. No admissible combination means no MTD "
            "recommendation.",
            "4. 完成后只在已治疗且未排除的组合中选择；无符合条件的组合则不推荐 MTD。",
        ),
    ):
        parts.append(text(english, chinese))
    parts.append(
        text(
            f"Precision threshold at a dose: {design.early_stop_patients}. "
            f"Run-in enabled: {study.run_in}. "
            "Run-in is active only for target 0.25 and cohort size three, "
            "retaining the dose after one DLT among three.",
            f"单剂量精度阈值：{design.early_stop_patients}。导入规则：{study.run_in}；"
            "仅目标为 0.25 且队列大小为三时启用，三例中一例 DLT 时维持剂量。",
        )
    )
    if study.mode == "contour":
        parts.append(
            text(
                f"Waterfall subtrial maxima in execution order are "
                f"{study.subtrial_maximums}. "
                "The initial staircase is followed by lower-row subtrials. A "
                "subtrial stops at its "
                "capacity, safety stop, or convergence; weighted isotonic "
                "estimates determine the "
                "transition. Final selection can include one combination per row. "
                "During simulation "
                "the native run-in additionally de-escalates at two DLTs among "
                "six; the conduct table "
                "does not introduce that extra override.",
                f"瀑布子试验按执行顺序的上限为 {study.subtrial_maximums}。初始阶梯后依次开展较低行"
                "子试验；达到人数上限、安全停止或收敛时结束子试验，以加权保序估计决定转移。"
                "最终每行可选一个组合。模拟中的原生导入规则还在六例中两例 DLT 时降量；"
                "实施表不额外加入此覆盖规则。",
            )
        )
    else:
        parts.append(
            text(
                "Escalation and de-escalation consider adjacent combinations "
                "changing one drug at "
                "a time. Posterior target proximity ranks candidates; random ties "
                "use the recorded seed.",
                "升降量考虑每次仅改变一种药物的相邻组合，以后验概率接近目标的程度排序；"
                "随机并列选择使用所记录的种子。",
            )
        )
    parts.append(
        text(
            f"Simulation accelerated titration: {study.titration}; escalation "
            f"cap: {study.titration_cap}. "
            "Safe simulation retains actual observations, tops up the current "
            "cohort without phantom "
            "patients and enforces total planned enrollment. Saved patient "
            "conduct uses its cohort "
            "controller and does not reproduce the simulation titration "
            "prelude. Legacy simulation "
            "must be explicitly requested and is not the default protocol.",
            f"模拟加速滴定：{study.titration}；升量上限参数：{study.titration_cap}。安全模拟保留"
            "实际观察、补足当前队列而不虚增患者，并遵守总入组上限。保存患者的实施流程使用队列"
            "控制器，不执行模拟的滴定前奏。旧版兼容模拟必须显式请求，不作为默认方案。",
        )
    )
    parts.append(
        text(
            "Final selection uses a Beta(0.05,0.05) working prior and "
            "weighted bivariate isotonic "
            "estimation. Reported isotonic transforms of marginal beta limits "
            "are not simultaneous "
            "credible bands. Simulation operating characteristics require "
            "explicit truth scenarios; "
            "none are invented by this methods document.",
            "最终选择采用 Beta(0.05,0.05) 工作先验与加权二维保序估计。对边际 beta 分位数的"
            "保序变换不构成同时可信带。模拟运行特征需要明确的真实毒性场景，本文不虚构结果。",
        )
    )
    parts.append(text("## Conduct integer boundaries", "## 实施整数边界"))
    parts.append(
        text(
            "Patients | Escalate if DLT ≤ | De-escalate if DLT ≥ | Eliminate if DLT ≥",
            "患者数 | DLT 不大于此值则升量 | DLT 不小于此值则降量 | DLT 不小于此值则排除",
        )
    )
    rows = [parts.pop(), "--- | --- | --- | ---"]
    for n, up, down, eliminate in zip(
        table.patients, table.escalate_max, table.deescalate_min, table.eliminate_min, strict=True
    ):
        cutoff = str(int(eliminate)) if np.isfinite(eliminate) else "—"
        rows.append(f"{n} | {int(up)} | {int(down)} | {cutoff}")
    parts.append("\n".join(rows))
    return "\n\n".join(parts) + "\n"
