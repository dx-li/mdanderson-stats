"""Independent bilingual statistical methods for saved single-agent/TITE studies."""

from .boin_desktop_single import desktop_single_design
from .boin_desktop_tite import desktop_tite_settings


def boin_desktop_univariate_protocol(study, *, language="en") -> str:
    """Describe the safe desktop analysis, without inventing operating characteristics.

    Complete-outcome tables do not replace the pending-outcome TITE decision
    algorithm. Native templates, page layouts and clinical approval are not claimed.
    """
    if language not in ("en", "zh"):
        raise ValueError("language must be en or zh")
    design = desktop_single_design(study)
    if study.maximum_patients % study.cohort_size:
        raise ValueError("planned maximum must contain whole cohorts")

    def text(english, chinese):
        return english if language == "en" else chinese

    parts = [text("# Desktop BOIN statistical methods", "# 桌面 BOIN 统计方法")]
    parts.append(
        text(
            f"Mode: {study.mode}. Target DLT probability: {study.target}; "
            f"{study.shape[0]} ordered doses, starting at dose {study.start_dose[0]}. "
            f"Cohort size: {study.cohort_size}; maximum enrollment: {study.maximum_patients}.",
            f"模式：{study.mode}。目标 DLT 概率为 {study.target}；"
            f"共有 {study.shape[0]} 个有序剂量，"
            f"从第 {study.start_dose[0]} 剂量开始。每队列 {study.cohort_size} 例，"
            f"最多入组 {study.maximum_patients} 例。",
        )
    )
    parts.append(
        text(
            f"Indifference probabilities: {design.safe_probability}, {design.toxic_probability}. "
            f"Escalation boundary: {design.escalation_boundary:.12g}; "
            f"de-escalation boundary: {design.deescalation_boundary:.12g}. "
            f"Run-in modification active: {study.run_in}; when active, stay with one DLT "
            "among three patients. Safety takes precedence over the dose boundaries.",
            f"无差异区间参数为 {design.safe_probability}、{design.toxic_probability}。"
            f"升量边界为 {design.escalation_boundary:.12g}，降量边界为 "
            f"{design.deescalation_boundary:.12g}。导入修正规则：{study.run_in}；"
            "启用时三例中一例 DLT 维持剂量。安全规则优先于剂量边界。",
        )
    )
    parts.append(
        text(
            f"Use a Beta(1,1) safety prior. With at least three patients, eliminate a dose "
            f"and higher doses if Pr(p > {study.target} | data) > "
            f"{design.elimination_probability}. "
            f"Extra-safe lowest-dose stopping: {design.extra_safe}; probability offset: "
            f"{design.safety_offset}. Never assign an eliminated dose. "
            f"The precision threshold is {design.early_stop_patients} patients at a dose, "
            "with stopping only when the resulting assignment stays at that dose.",
            f"安全性采用 Beta(1,1) 先验，至少三例时，若 Pr(p > {study.target} | 数据) > "
            f"{design.elimination_probability} 则排除该剂量及更高剂量。最低剂量额外安全停止："
            f"{design.extra_safe}；概率偏移量：{design.safety_offset}。不分配已排除剂量。"
            f"精度阈值为同剂量 {design.early_stop_patients} 例，且下一分配仍为该剂量时才停止。",
        )
    )
    if study.mode == "single":
        parts.append(
            text(
                "Dose decisions use complete cumulative outcomes. Accelerated titration "
                f"enabled: {study.titration}; saved cap: {study.titration_cap}. Native desktop "
                "titration treats one patient per dose until a DLT or the endpoint "
                "min(cap+1, number of doses), then tops up that same dose to a cohort. "
                "Safe simulation respects the total enrollment cap.",
                f"剂量决策采用完整累计结局。加速滴定：{study.titration}；保存上限参数："
                f"{study.titration_cap}。桌面滴定每剂量一例，出现 DLT 或到达 "
                "min(上限参数+1, 剂量数) 后在同剂量补足队列。安全模拟遵守总入组上限。",
            )
        )
    else:
        timing = desktop_tite_settings(study)
        weights = timing.trimester_weights or (1 / 3, 1 / 3, 1 / 3)
        parts.append(
            text(
                f"Assessment window: {timing.window_months:.12g} months "
                f"({timing.window_days:.12g} days); accrual: {timing.accrual_per_month} per month. "
                f"Maximum pending fraction: {timing.max_pending_fraction}. Analysis trimester "
                f"weights: {weights}. Simulated event distribution: "
                f"{timing.distribution}; late-half probability: {timing.late_probability}.",
                f"观察窗为 {timing.window_months:.12g} 月（{timing.window_days:.12g} 天），"
                f"月入组率为 {timing.accrual_per_month}。最大待观察比例为 "
                f"{timing.max_pending_fraction}。分析用三段权重为 {weights}；"
                f"模拟事件分布为 {timing.distribution}，后半窗概率为 {timing.late_probability}。",
            )
        )
        parts.append(
            text(
                "Enrollment starts at time zero. Exponential arrival gaps advance enrollment "
                "and pending checks. Observed safety and raw-DLT de-escalation precede the "
                "pending-fraction suspension gate. Otherwise pending DLTs are imputed using "
                "a Beta(target/2, 1-target/2) working prior and weighted follow-up. The final "
                "cohort is followed through its assessment window. Safe analysis uses only "
                "observed information and never moves its clock backward. The saved STFT "
                "calculator takes follow-up in days even when the window is displayed in months. "
                "One month is 30.436666666666667 days. No separate web minimum-follow-up gate "
                "is imposed by this desktop calendar.",
                "首例在时间零入组，以指数到达间隔推进入组与待观察检查。已观察安全性规则及原始 "
                "DLT 降量判断优先于待观察比例暂停规则；否则以 Beta(目标/2, 1-目标/2) "
                "工作先验及加权随访插补待观察 DLT。末队列随访至观察窗结束。安全分析只使用"
                "已观察信息，时间不倒退。STFT 计算器的随访值始终以天计，即使观察窗显示为月。"
                "一个月为 30.436666666666667 天。桌面日历不额外施加网页最低随访门槛。",
            )
        )
    parts.append(
        text(
            "Final selection requires complete outcomes. Fit Beta(.05,.05) posterior means "
            "with inverse posterior-variance weighted isotonic regression among treated, "
            "noneliminated doses. Select the mean nearest target; exact ties below target "
            "favor the higher dose, otherwise the lower dose. No eligible dose means no MTD. "
            "Native near-tie floating-point and random-stream identity are not claimed.",
            "最终选择须确认结局完整。对已治疗且未排除剂量的 Beta(.05,.05) 后验均值，"
            "以其后验方差倒数加权作保序回归，选择最接近目标的剂量。精确并列且均低于目标时选"
            "较高剂量，否则选较低剂量。无符合条件剂量则不选 MTD。"
            "不声称原生近并列浮点或随机流一致。",
        )
    )
    parts.append(
        text(
            "## Complete-outcome integer boundaries\n\nThese are complete-outcome cutoffs; "
            "pending TITE outcomes require the calendar rules above. Safety cutoffs are "
            "inclusive; a dash means unreachable at this sample size.",
            "## 完整结局整数边界\n\n下表仅适用于完整结局；TITE 待观察结局需采用上述日历规则。"
            "安全阈值包含端点，横线表示该样本量下不可达。",
        )
    )
    rows = [
        text(
            "Patients | Escalate ≤ | De-escalate ≥ | Eliminate ≥ | Lowest stop ≥",
            "例数 | 升量 ≤ | 降量 ≥ | 排除 ≥ | 最低剂量停止 ≥",
        ),
        "--- | --- | --- | --- | ---",
    ]
    table = design.boundary_table(study.maximum_patients)
    for n, up, down, eliminate, stop in zip(
        table.patients,
        table.escalate_max,
        table.deescalate_min,
        table.eliminate_min,
        table.lowest_stop_min,
        strict=True,
    ):
        rows.append(
            f"{n} | {up} | {down} | {eliminate if eliminate <= n else '—'} | "
            f"{stop if stop <= n else '—'}"
        )
    parts.append("\n".join(rows))
    parts.append(
        text(
            "This is independently authored statistical methods text, not a complete clinical "
            "protocol. Operating characteristics must be generated from explicit truth scenarios. "
            "Legacy defects require an explicit compatibility profile "
            "and are not the safe default.",
            "本文为独立编写的统计方法，不是完整临床方案。运行特征须基于明确真实场景模拟。"
            "旧版缺陷须显式指定兼容模式，不作为安全默认值。",
        )
    )
    return "\n\n".join(parts) + "\n"
