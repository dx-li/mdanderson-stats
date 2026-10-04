# Independent base-R references for automatic success-cutoff calibration.
# This is a mathematical reference, not a reproduction of the application UI.
options(warn = 2, digits = 17)

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1L) normalizePath(args[[1]], mustWork = TRUE) else normalizePath(".", mustWork = TRUE)
out_dir <- if (length(args) >= 2L) args[[2]] else file.path(repo_root, "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Single-arm binary: compute all n+1 strict-posterior decision states directly.
binary_oc <- function(n, cutoff, margin, da, db, aa, ab, direction) {
  x <- 0:n
  effective_analysis <- if (direction == "greater") {
    pbeta(margin, aa + x, ab + n - x, lower.tail = FALSE)
  } else {
    pbeta(margin, aa + x, ab + n - x)
  }
  effective_design <- if (direction == "greater") {
    pbeta(margin, da + x, db + n - x, lower.tail = FALSE)
  } else {
    pbeta(margin, da + x, db + n - x)
  }
  ineffective_design <- if (direction == "greater") {
    pbeta(margin, da + x, db + n - x)
  } else {
    pbeta(margin, da + x, db + n - x, lower.tail = FALSE)
  }
  predictive <- choose(n, x) * exp(lbeta(da + x, db + n - x) - lbeta(da, db))
  success <- effective_analysis > cutoff
  tp <- sum(predictive[success] * effective_design[success])
  fp <- sum(predictive[success] * ineffective_design[success])
  c(success_count = sum(success), tp = tp, fp = fp,
    power = sum(predictive[success]), pid = if (sum(success) > 0) fp / (tp + fp) else NA_real_)
}

binary_calibrate <- function(spec) {
  x <- 0:spec$n
  tail <- if (spec$direction == "greater") "upper" else "lower"
  breaks <- if (tail == "upper") {
    pbeta(spec$margin, spec$aa + x, spec$ab + spec$n - x, lower.tail = FALSE)
  } else {
    pbeta(spec$margin, spec$aa + x, spec$ab + spec$n - x)
  }
  # At an exact posterior breakpoint the strict rule excludes tied counts, so
  # evaluating every breakpoint plus the interval lower bound covers all states.
  candidates <- sort(unique(c(spec$lower,
    breaks[breaks > spec$lower & breaks <= spec$upper])))
  rows <- lapply(candidates, function(cutoff) {
    z <- binary_oc(spec$n, cutoff, spec$margin, spec$da, spec$db,
                   spec$aa, spec$ab, spec$direction)
    data.frame(case = spec$name, cutoff = cutoff, success_count = z[["success_count"]],
      true_positive = z[["tp"]], false_positive = z[["fp"]],
      success_probability = z[["power"]], pid = z[["pid"]])
  })
  curve <- do.call(rbind, rows)
  feasible <- which(is.finite(curve$pid) & curve$pid <= spec$target)
  selected <- if (length(feasible)) curve[feasible[[1]], , drop = FALSE] else NULL
  list(curve = curve, selected = selected,
       settings = data.frame(case = spec$name, n = spec$n, direction = spec$direction,
         margin = spec$margin, design_alpha = spec$da, design_beta = spec$db,
         analysis_alpha = spec$aa, analysis_beta = spec$ab,
         cutoff_min = spec$lower, cutoff_max = spec$upper, target_pid = spec$target,
         feasible = !is.null(selected)))
}

binary_cases <- list(
  greater_unequal_priors = list(name = "greater_unequal_priors", n = 11L,
    direction = "greater", margin = .30, da = 2.2, db = 5.1, aa = 1.4, ab = 2.7,
    lower = .55, upper = .98, target = .18),
  less_unequal_priors = list(name = "less_unequal_priors", n = 9L,
    direction = "less", margin = .62, da = 4.1, db = 2.3, aa = 2.2, ab = 1.6,
    lower = .50, upper = .99, target = .16),
  discrete_jump = list(name = "discrete_jump", n = 5L,
    direction = "greater", margin = .40, da = 1.3, db = 3.7, aa = 1, ab = 1,
    lower = .60, upper = .999, target = .12),
  exact_n1 = list(name = "exact_n1", n = 1L,
    direction = "greater", margin = .50, da = 1, db = 1, aa = 1, ab = 1,
    lower = .50, upper = .75, target = .30),
  infeasible_range = list(name = "infeasible_range", n = 4L,
    direction = "greater", margin = .30, da = 2, db = 5, aa = 1, ab = 1,
    lower = .60, upper = .70, target = .001)
)
binary_results <- lapply(binary_cases, binary_calibrate)
binary_curves <- do.call(rbind, lapply(binary_results, `[[`, "curve"))
binary_settings <- do.call(rbind, lapply(binary_results, `[[`, "settings"))
binary_probes <- do.call(rbind, lapply(binary_cases, function(spec) {
  x <- 0:spec$n
  breaks <- if (spec$direction == "greater") {
    pbeta(spec$margin, spec$aa + x, spec$ab + spec$n - x, lower.tail = FALSE)
  } else {
    pbeta(spec$margin, spec$aa + x, spec$ab + spec$n - x)
  }
  boundaries <- sort(unique(c(spec$lower,
    breaks[breaks > spec$lower & breaks <= spec$upper], spec$upper)))
  probes <- if (length(boundaries) > 1L) {
    boundaries[-length(boundaries)] + diff(boundaries) / 2
  } else numeric()
  if (!length(probes)) return(NULL)
  do.call(rbind, lapply(probes, function(cutoff) {
    z <- binary_oc(spec$n, cutoff, spec$margin, spec$da, spec$db,
                   spec$aa, spec$ab, spec$direction)
    data.frame(case = spec$name, cutoff = cutoff, success_count = z[["success_count"]],
      true_positive = z[["tp"]], false_positive = z[["fp"]],
      success_probability = z[["power"]], pid = z[["pid"]])
  }))
}))
binary_selected <- do.call(rbind, lapply(binary_results, function(z) {
  if (is.null(z$selected)) return(data.frame(case = z$settings$case, feasible = FALSE,
    cutoff = NA_real_, success_count = NA_integer_, true_positive = NA_real_,
    false_positive = NA_real_, success_probability = NA_real_, pid = NA_real_))
  data.frame(case = z$selected$case, feasible = TRUE,
    cutoff = z$selected$cutoff, success_count = z$selected$success_count,
    true_positive = z$selected$true_positive, false_positive = z$selected$false_positive,
    success_probability = z$selected$success_probability, pid = z$selected$pid)
}))
write.csv(binary_settings, file.path(out_dir, "success-automatic-binary-settings.csv"), row.names = FALSE)
write.csv(binary_curves, file.path(out_dir, "success-automatic-binary-states.csv"), row.names = FALSE)
write.csv(binary_probes, file.path(out_dir, "success-automatic-binary-interior-probes.csv"), row.names = FALSE)
write.csv(binary_selected, file.path(out_dir, "success-automatic-binary-selected.csv"), row.names = FALSE)

# Normal model: integrate the bivariate true-effect / posterior-score law by
# conditioning on the standardized true effect. This does not call Python code.
normal_oc_ref <- function(cutoff, se, dm, ds, am, ass, margin, direction) {
  stopifnot(length(se) %in% c(1L, 2L), length(dm) == length(se),
            length(ds) == length(se), length(am) == length(se), length(ass) == length(se))
  sign <- if (direction == "greater") 1 else -1
  contrast <- if (length(se) == 1L) 1 else c(1, -1)
  w <- ass^2 / (ass^2 + se^2)
  post_var <- ass^2 * se^2 / (ass^2 + se^2)
  true_mean <- sum(contrast * dm)
  true_sd <- sqrt(sum(ds^2))
  score_mean <- sum(contrast * (w * dm + (1 - w) * am))
  score_var <- sum(w^2 * (ds^2 + se^2))
  post_sd <- sqrt(sum(post_var))
  covariance <- sum(w * ds^2)
  rho <- covariance / (true_sd * sqrt(score_var))
  stopifnot(rho > 0, rho < 1)
  boundary <- sign * margin + qnorm(cutoff) * post_sd
  b <- (boundary - sign * score_mean) / sqrt(score_var)
  a <- (sign * margin - sign * true_mean) / true_sd
  conditional_success <- function(z) pnorm((rho * z - b) / sqrt(1 - rho^2))
  integrate_part <- function(lo, hi) {
    if (lo >= hi) return(0)
    integrate(function(z) dnorm(z) * conditional_success(z), lo, hi,
      rel.tol = 2e-11, abs.tol = 2e-13, subdivisions = 300L, stop.on.error = TRUE)$value
  }
  low <- max(-12, min(12, a))
  fp <- if (a <= -12) 0 else integrate_part(-12, low)
  tp <- if (a >= 12) 0 else integrate_part(max(-12, a), 12)
  power <- pnorm(-b)
  c(tp = tp, fp = fp, power = power, pid = fp / power, rho = rho,
    score_mean = sign * score_mean, score_sd = sqrt(score_var), posterior_sd = post_sd)
}

normal_bisect <- function(spec, tolerance = 1e-8, max_evaluations = 80L) {
  evaluate <- function(cutoff) normal_oc_ref(cutoff, spec$se, spec$dm, spec$ds,
    spec$am, spec$ass, spec$margin, spec$direction)
  lo <- spec$lower; hi <- spec$upper
  lo_result <- evaluate(lo)
  flo <- lo_result[["pid"]] - spec$target
  if (flo <= 0) {
    return(data.frame(case = spec$name, target_pid = spec$target, cutoff_min = spec$lower,
      cutoff_max = spec$upper, cutoff_tolerance = tolerance, lower_infeasible = lo,
      upper_feasible = lo, lower_pid = lo_result[["pid"]], upper_pid = lo_result[["pid"]],
      success_probability = lo_result[["power"]], true_positive = lo_result[["tp"]],
      false_positive = lo_result[["fp"]], correlation = lo_result[["rho"]], evaluations = 1L))
  }
  hi_result <- evaluate(hi)
  fhi <- hi_result[["pid"]] - spec$target
  if (!(flo >= 0 && fhi <= 0)) {
    stop(sprintf("reference target not bracketed for %s: PID endpoints %.17g, %.17g; target %.17g",
      spec$name, flo + spec$target, fhi + spec$target, spec$target))
  }
  evals <- 2L
  best <- hi_result
  while (hi - lo > tolerance && evals < max_evaluations) {
    mid <- lo + (hi - lo) / 2
    middle_result <- evaluate(mid)
    fm <- middle_result[["pid"]] - spec$target
    evals <- evals + 1L
    if (fm <= 0) {
      hi <- mid
      best <- middle_result
    } else {
      lo <- mid
      lo_result <- middle_result
    }
  }
  stopifnot(hi - lo <= tolerance)
  data.frame(case = spec$name, target_pid = spec$target, cutoff_min = spec$lower,
    cutoff_max = spec$upper, cutoff_tolerance = tolerance, lower_infeasible = lo,
    upper_feasible = hi, lower_pid = lo_result[["pid"]], upper_pid = best[["pid"]],
    success_probability = best[["power"]], true_positive = best[["tp"]],
    false_positive = best[["fp"]], correlation = best[["rho"]],
    evaluations = evals)
}

normal_cases <- list(
  normal_greater = list(name = "normal_greater", se = .45, dm = .10, ds = .38,
    am = -.05, ass = .72, margin = .15, direction = "greater", lower = .60, upper = .999,
    null = .15, target = .12),
  normal_unequal_two_arm = list(name = "normal_unequal_two_arm", se = c(.4, .6),
    dm = c(.45, .10), ds = c(.38, .65), am = c(.08, .18), ass = c(.31, 1.1),
    margin = .20, direction = "greater", null = c(.20, 0),
    lower = .60, upper = .999, target = .10),
  normal_reflected_less = list(name = "normal_reflected_less", se = c(.4, .6),
    dm = c(-.45, -.10), ds = c(.38, .65), am = c(-.08, -.18), ass = c(.31, 1.1),
    margin = -.20, direction = "less", null = c(-.20, 0),
    lower = .60, upper = .999, target = .10)
)
normal_selected <- do.call(rbind, lapply(normal_cases, normal_bisect))
normal_settings <- do.call(rbind, lapply(normal_cases, function(s) data.frame(
  case = s$name, direction = s$direction, n_arms = length(s$se),
  se_1 = s$se[1], se_2 = if (length(s$se) == 2L) s$se[2] else NA_real_,
  design_mean_1 = s$dm[1], design_mean_2 = if (length(s$dm) == 2L) s$dm[2] else NA_real_,
  design_sd_1 = s$ds[1], design_sd_2 = if (length(s$ds) == 2L) s$ds[2] else NA_real_,
  analysis_mean_1 = s$am[1], analysis_mean_2 = if (length(s$am) == 2L) s$am[2] else NA_real_,
  analysis_sd_1 = s$ass[1], analysis_sd_2 = if (length(s$ass) == 2L) s$ass[2] else NA_real_,
  null_mean_1 = if (length(s$null) == 1L) s$null else s$null[1],
  null_mean_2 = if (length(s$null) == 2L) s$null[2] else NA_real_,
  margin = s$margin, cutoff_min = s$lower, cutoff_max = s$upper, target_pid = s$target
)))
curve_cutoffs <- c(.60, .75, .90, .975, .999)
normal_curves <- do.call(rbind, lapply(normal_cases, function(spec) {
  do.call(rbind, lapply(curve_cutoffs, function(cutoff) {
    z <- normal_oc_ref(cutoff, spec$se, spec$dm, spec$ds, spec$am, spec$ass,
                       spec$margin, spec$direction)
    data.frame(case = spec$name, cutoff = cutoff, true_positive = z[["tp"]],
      false_positive = z[["fp"]], success_probability = z[["power"]], pid = z[["pid"]],
      correlation = z[["rho"]])
  }))
}))
write.csv(normal_selected, file.path(out_dir, "success-automatic-normal-selected.csv"), row.names = FALSE)
write.csv(normal_curves, file.path(out_dir, "success-automatic-normal-curve.csv"), row.names = FALSE)
write.csv(normal_settings, file.path(out_dir, "success-automatic-normal-settings.csv"), row.names = FALSE)

# Survival conversion and scale-rescaling identity: SE=1/sqrt(D*r*(1-r));
# divide all log-HR scale parameters and the margin by k while multiplying D by k^2.
survival_cases <- list(
  survival_base = list(name = "survival_base", events = 120, allocation = .4,
    dm = -.18, ds = .32, am = -.04, ass = .75, margin = -.05,
    lower = .60, upper = .999, target = .10),
  survival_rescaled = list(name = "survival_rescaled", events = 480, allocation = .4,
    dm = -.09, ds = .16, am = -.02, ass = .375, margin = -.025,
    lower = .60, upper = .999, target = .10)
)
survival_selected <- do.call(rbind, lapply(survival_cases, function(spec) {
  normal_bisect(list(name = spec$name, se = 1 / sqrt(spec$events * spec$allocation * (1 - spec$allocation)),
    dm = spec$dm, ds = spec$ds, am = spec$am, ass = spec$ass,
    margin = spec$margin, direction = "less", lower = spec$lower,
    upper = spec$upper, target = spec$target))
}))
write.csv(survival_selected, file.path(out_dir, "success-automatic-survival-selected.csv"), row.names = FALSE)
write.csv(do.call(rbind, lapply(survival_cases, function(s) data.frame(
  case = s$name, events = s$events, treatment_allocation = s$allocation,
  design_mean = s$dm, design_sd = s$ds, analysis_mean = s$am,
  analysis_sd = s$ass, margin = s$margin, direction = "less",
  cutoff_min = s$lower, cutoff_max = s$upper, target_pid = s$target
))), file.path(out_dir, "success-automatic-survival-settings.csv"), row.names = FALSE)

cat("binary selected cutoffs:\n"); print(binary_selected)
cat("normal selected brackets:\n"); print(normal_selected)
cat("survival selected brackets:\n"); print(survival_selected)
