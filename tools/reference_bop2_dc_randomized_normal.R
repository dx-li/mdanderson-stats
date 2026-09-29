# Independent base-R reference for randomized Normal BOP2-DC monitor/replay.
# Rscript tools/reference_bop2_dc_randomized_normal.R [path-prefix]
# Tail integration is over the standardized CONTROL Student-t density on R,
# independent of the Python implementation's uniform-quantile integration.
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[[1]] else "tests/fixtures/bop2-dc-randomized-normal-"
cases <- read.csv(paste0(prefix, "cases.csv"), stringsAsFactors = FALSE,
                  colClasses = c("character", rep("numeric", 15), "character", "character", "numeric", "character", "numeric"))
parse_num <- function(x) as.numeric(strsplit(x, ";", fixed = TRUE)[[1]])
parse_int <- function(x) as.integer(strsplit(x, ";", fixed = TRUE)[[1]])

# Direct convolution integral: C = muC+sC*X, X~t(dfC); then integrate
# P(T > margin+C | C) = sf_t((margin-(muT-muC)+sC*X)/sT).
difference_tail <- function(mu_c, scale_c, df_c, mu_t, scale_t, df_t, margin) {
  delta <- mu_t - mu_c
  if (!is.finite(delta)) stop("reference posterior location difference overflow")
  if (margin == delta) return(list(probability = 0.5, abs_error = 0))
  z <- integrate(function(x) {
    dt(x, df = df_c) * pt((margin - delta + scale_c * x) / scale_t,
                           df = df_t, lower.tail = FALSE)
  }, lower = -Inf, upper = Inf, rel.tol = 2e-12, abs.tol = 2e-13,
  subdivisions = 2000L, stop.on.error = TRUE)
  if (!is.finite(z$value) || !is.finite(z$abs.error) || z$value < 0 || z$value > 1)
    stop("invalid posterior difference integral")
  list(probability = z$value, abs_error = z$abs.error)
}

# Independent NIG sufficient-statistic update for prior (mu0,kappa0,a0,b0).
update_nig <- function(mu0, kappa0, a0, b0, values, offset) {
  mu0c <- mu0 - offset
  yc <- values - offset
  n <- length(yc)
  if (n == 0) return(list(location_centered = mu0c, offset = offset,
                          kappa = kappa0, shape = a0, scale_ig = b0,
                          df = 2 * a0, t_scale = sqrt(b0 / (a0 * kappa0)),
                          mean_centered = mu0c, sum_squares = 0))
  ybar <- mean(yc)
  ss <- sum((yc - ybar)^2)
  kn <- kappa0 + n
  an <- a0 + n / 2
  mun <- (kappa0 * mu0c + n * ybar) / kn
  bn <- b0 + ss / 2 + kappa0 * n * (ybar - mu0c)^2 / (2 * kn)
  if (!is.finite(bn) || bn <= 0) stop("unrepresentable reference NIG scale")
  list(location_centered = mun, offset = offset, kappa = kn, shape = an,
       scale_ig = bn, df = 2 * an, t_scale = sqrt(bn / (an * kn)),
       mean_centered = ybar, sum_squares = ss)
}

obf_cutoff <- function(lambda, look, n) {
  2 * pnorm(qnorm((1 + lambda) / 2) / sqrt(look / n)) - 1
}
decision_at <- function(pl, pc, look, row) {
  n <- row$max_subjects
  if (look < n) {
    no_go <- pl < row$lambda_lrv * (look / n)^row$gamma_lrv &&
      pc < row$lambda_cmv * (look / n)^row$gamma_cmv
    graduate <- FALSE
    gl <- gc <- NA_real_
    if (row$graduate_at_interim == 1) {
      gl <- obf_cutoff(row$lambda_lrv, look, n)
      gc <- obf_cutoff(row$lambda_cmv, look, n)
      graduate <- pl > gl && pc > gc
    }
    if (no_go && graduate) stop("overlapping reference decision regions")
    action <- if (no_go) "stop_no_go" else if (graduate) "graduate" else "continue"
    return(list(action = action, cutoff_lrv = row$lambda_lrv * (look / n)^row$gamma_lrv,
                cutoff_cmv = row$lambda_cmv * (look / n)^row$gamma_cmv,
                graduate_lrv = gl, graduate_cmv = gc))
  }
  go <- pl > row$lambda_lrv && pc > row$lambda_cmv
  no_go <- pl < row$lambda_lrv && pc < row$lambda_cmv
  action <- if (go) "final_go" else if (no_go) "final_no_go" else "final_consider"
  list(action = action, cutoff_lrv = row$lambda_lrv, cutoff_cmv = row$lambda_cmv,
       graduate_lrv = NA_real_, graduate_cmv = NA_real_)
}

out <- list(); idx <- 0L
for (i in seq_len(nrow(cases))) {
  r <- cases[i, , drop = FALSE]
  assignments <- strsplit(r$assignments, "", fixed = TRUE)[[1]]
  values <- parse_num(r$outcomes)
  input_shift <- r$common_shift
  looks <- parse_int(r$looks)
  if (length(assignments) != r$max_subjects || length(values) != r$max_subjects)
    stop(paste("case tape length mismatch:", r$case_id))
  if (!all(assignments %in% c("0", "1")) || !all(is.finite(values)) ||
      !all(looks >= 1 & looks <= r$max_subjects) || tail(looks, 1) != r$max_subjects ||
      any(diff(looks) <= 0)) stop(paste("invalid case input:", r$case_id))
  terminal <- "continue"
  for (look in looks) {
    prefix_arm <- assignments[seq_len(look)]
    prefix_y <- values[seq_len(look)]
    offset <- if (any(prefix_arm == "0")) prefix_y[which(prefix_arm == "0")[[1]]] else prefix_y[[1]]
    cdat <- prefix_y[prefix_arm == "0"]
    tdat <- prefix_y[prefix_arm == "1"]
    cp <- update_nig(r$c_mu, r$c_k, r$c_a, r$c_b, cdat, offset)
    tp <- update_nig(r$t_mu, r$t_k, r$t_a, r$t_b, tdat, offset)
    diff_loc <- tp$location_centered - cp$location_centered
    lrv <- difference_tail(cp$location_centered, cp$t_scale, cp$df,
                           tp$location_centered, tp$t_scale, tp$df, r$theta_lrv)
    cmv <- difference_tail(cp$location_centered, cp$t_scale, cp$df,
                           tp$location_centered, tp$t_scale, tp$df, r$theta_cmv)
    decision <- decision_at(lrv$probability, cmv$probability, look, r)
    idx <- idx + 1L
    out[[idx]] <- data.frame(
      case_id = r$case_id, assignments = r$assignments, outcomes_centered = r$outcomes,
      common_shift = input_shift, look = look, control_n = length(cdat), treatment_n = length(tdat),
      location_offset = input_shift + offset,
      control_location_centered = cp$location_centered,
      treatment_location_centered = tp$location_centered,
      control_location = cp$location_centered + offset + input_shift,
      treatment_location = tp$location_centered + offset + input_shift,
      control_kappa = cp$kappa, control_shape = cp$shape, control_df = cp$df,
      control_scale_ig = cp$scale_ig, control_t_scale = cp$t_scale,
      control_sum_squares = cp$sum_squares,
      treatment_kappa = tp$kappa, treatment_shape = tp$shape, treatment_df = tp$df,
      treatment_scale_ig = tp$scale_ig, treatment_t_scale = tp$t_scale,
      treatment_sum_squares = tp$sum_squares,
      difference_location = diff_loc,
      posterior_lrv = lrv$probability, error_lrv = lrv$abs_error,
      posterior_cmv = cmv$probability, error_cmv = cmv$abs_error,
      no_go_cutoff_lrv = decision$cutoff_lrv, no_go_cutoff_cmv = decision$cutoff_cmv,
      graduate_cutoff_lrv = decision$graduate_lrv,
      graduate_cutoff_cmv = decision$graduate_cmv,
      decision = decision$action
    )
    terminal <- decision$action
    if (terminal != "continue") break
  }
}

# Prior-only df=1 arm means are independent Cauchy distributions.  With
# control prior (-0.5, kappa=2, a=0.5, b=1) and treatment prior
# (1, kappa=2, a=0.5, b=4), the locations are -0.5 and 1 and scales 1 and 2.
# Their difference is Cauchy(location=1.5, scale=3), giving a nonzero-location
# analytic check of the direct convolution integral.
cauchy_margins <- c(-3, -0.5, 0, 1.5, 3, 5)
prior_only <- do.call(rbind, lapply(cauchy_margins, function(margin) {
  numerical <- difference_tail(-0.5, 1, 1, 1, 2, 1, margin)
  analytic <- 0.5 - atan((margin - 1.5) / 3) / pi
  data.frame(control_df = 1, treatment_df = 1, control_location = -0.5,
             treatment_location = 1, control_t_scale = 1, treatment_t_scale = 2,
             margin = margin, numerical_probability = numerical$probability,
             numerical_error = numerical$abs_error, analytic_probability = analytic,
             absolute_discrepancy = abs(numerical$probability - analytic))
}))
write.csv(do.call(rbind, out), paste0(prefix, "monitor-replay.csv"), row.names = FALSE)
write.csv(prior_only, paste0(prefix, "prior-only-cauchy.csv"), row.names = FALSE)
