# Independent base-R oracle for BOP2-DC randomized binary monitoring and exact OCs.
# Rscript tools/reference_bop2_dc_randomized_binary.R [fixture-prefix]
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[[1]] else "tests/fixtures/bop2-dc-randomized-binary-"
settings <- read.csv(paste0(prefix, "settings.csv"), stringsAsFactors = FALSE,
                     colClasses = c("character", rep("numeric", 11), "character", "character", "numeric"))
truths <- read.csv(paste0(prefix, "truths.csv"), stringsAsFactors = FALSE)
parse_ints <- function(x) as.integer(strsplit(x, "[; ]+")[[1]])

# For integer beta shapes, F_Beta(a,b)(x) is a finite binomial upper tail.
beta_cdf_integer <- function(x, a, b) {
  stopifnot(a >= 1, b >= 1, a == as.integer(a), b == as.integer(b))
  if (x <= 0) return(0)
  if (x >= 1) return(1)
  total <- a + b - 1
  sum(vapply(a:total, function(k) choose(total, k) * x^k * (1 - x)^(total - k), numeric(1)))
}

# P(T-C > delta), integrating the treatment density against a polynomial
# control CDF. The reverse tail is 1 minus this value (continuous posteriors).
beta_difference_above <- function(ac, bc, at, bt, delta) {
  if (delta == 0 && ac == at && bc == bt) return(0.5)
  if (delta >= 1) return(0)
  if (delta <= -1) return(1)
  lo <- max(0, delta)
  integrate(function(t) {
    vapply(t, function(tt) exp(dbeta(tt, at, bt, log = TRUE)) *
             beta_cdf_integer(tt - delta, ac, bc), numeric(1))
  }, lower = lo, upper = 1, rel.tol = 2e-13, abs.tol = 2e-14,
  subdivisions = 1000L, stop.on.error = TRUE)$value
}
obf_cutoff <- function(lambda, look, n) {
  2 * pnorm(qnorm((1 + lambda) / 2) / sqrt(look / n)) - 1
}

decision_at <- function(p_lrv, p_cmv, look, s) {
  n <- s$max_subjects
  if (look < n) {
    no_lrv <- s$lambda_lrv * (look / n)^s$gamma_lrv
    no_cmv <- s$lambda_cmv * (look / n)^s$gamma_cmv
    no_go <- p_lrv < no_lrv && p_cmv < no_cmv
    graduate <- FALSE
    if (s$graduate_at_interim == 1) {
      graduate <- p_lrv > obf_cutoff(s$lambda_lrv, look, n) &&
        p_cmv > obf_cutoff(s$lambda_cmv, look, n)
    }
    if (no_go && graduate) stop("overlapping rules in reference settings")
    if (no_go) return("stop_no_go")
    if (graduate) return("graduate")
    return("continue")
  }
  go <- p_lrv > s$lambda_lrv && p_cmv > s$lambda_cmv
  no_go <- p_lrv < s$lambda_lrv && p_cmv < s$lambda_cmv
  if (go) return("final_go")
  if (no_go) return("final_no_go")
  "final_consider"
}

posterior_at <- function(path, look, s) {
  arms <- strsplit(s$arm_assignments, "", fixed = TRUE)[[1]][seq_len(look)]
  y <- as.integer(strsplit(path, "", fixed = TRUE)[[1]][seq_len(look)])
  cidx <- arms == "0"
  tidx <- arms == "1"
  cn <- sum(cidx); tn <- sum(tidx); cy <- sum(y[cidx]); ty <- sum(y[tidx])
  ac <- s$control_alpha + cy; bc <- s$control_beta + cn - cy
  at <- s$treatment_alpha + ty; bt <- s$treatment_beta + tn - ty
  list(cn = cn, tn = tn, cy = cy, ty = ty,
       pl = beta_difference_above(ac, bc, at, bt, s$theta_lrv),
       pc = beta_difference_above(ac, bc, at, bt, s$theta_cmv))
}

paths <- expand.grid(rep(list(0:1), 4), KEEP.OUT.ATTRS = FALSE)
paths$path <- apply(paths, 1, paste0, collapse = "")
paths$path_id <- seq_len(nrow(paths))
state_rows <- list(); path_rows <- list(); oc_rows <- list()
for (si in seq_len(nrow(settings))) {
  s <- settings[si, , drop = FALSE]
  looks <- parse_ints(s$looks)
  arms <- strsplit(s$arm_assignments, "", fixed = TRUE)[[1]]
  current_paths <- vector("list", nrow(paths))
  for (pi in seq_len(nrow(paths))) {
    path <- paths$path[pi]
    terminal <- "continue"; terminal_look <- s$max_subjects
    for (look in looks) {
      post <- posterior_at(path, look, s)
      action <- decision_at(post$pl, post$pc, look, s)
      final <- look == s$max_subjects
      state_rows[[length(state_rows) + 1L]] <- data.frame(
        config_id = s$config_id, path_id = paths$path_id[pi], path = path,
        look = look, control_n = post$cn, treatment_n = post$tn,
        control_y = post$cy, treatment_y = post$ty,
        posterior_lrv = post$pl, posterior_cmv = post$pc,
        no_go_cutoff_lrv = if (final) s$lambda_lrv else s$lambda_lrv * (look / s$max_subjects)^s$gamma_lrv,
        no_go_cutoff_cmv = if (final) s$lambda_cmv else s$lambda_cmv * (look / s$max_subjects)^s$gamma_cmv,
        graduate_cutoff_lrv = if (!final && s$graduate_at_interim == 1) obf_cutoff(s$lambda_lrv, look, s$max_subjects) else NA_real_,
        graduate_cutoff_cmv = if (!final && s$graduate_at_interim == 1) obf_cutoff(s$lambda_cmv, look, s$max_subjects) else NA_real_,
        decision = action
      )
      if (action != "continue") { terminal <- action; terminal_look <- look; break }
    }
    current_paths[[pi]] <- data.frame(config_id = s$config_id, path_id = paths$path_id[pi],
                                      path = path, terminal_decision = terminal,
                                      terminal_look = terminal_look)
    path_rows[[length(path_rows) + 1L]] <- current_paths[[pi]]
  }
  for (ti in seq_len(nrow(truths))) {
    truth <- truths[ti, , drop = FALSE]
    stop_mass <- setNames(rep(0, length(looks)), looks)
    graduate_mass <- stop_mass; sample_mass <- stop_mass
    fg <- fc <- fn <- expected <- 0
    for (pi in seq_len(nrow(paths))) {
      y <- as.integer(strsplit(paths$path[pi], "", fixed = TRUE)[[1]])
      prob_i <- ifelse(arms == "0", truth$control_probability, truth$treatment_probability)
      prob <- prod(ifelse(y == 1, prob_i, 1 - prob_i))
      out <- current_paths[[pi]]
      action <- out$terminal_decision[[1]]; look <- out$terminal_look[[1]]
      expected <- expected + prob * look
      sample_mass[as.character(look)] <- sample_mass[as.character(look)] + prob
      if (action == "stop_no_go") stop_mass[as.character(look)] <- stop_mass[as.character(look)] + prob
      else if (action == "graduate") graduate_mass[as.character(look)] <- graduate_mass[as.character(look)] + prob
      else if (action == "final_go") fg <- fg + prob
      else if (action == "final_consider") fc <- fc + prob
      else if (action == "final_no_go") fn <- fn + prob
    }
    oc_rows[[length(oc_rows) + 1L]] <- data.frame(
      config_id = s$config_id, truth_id = truth$truth_id,
      control_probability = truth$control_probability, treatment_probability = truth$treatment_probability,
      stop_look_2 = if ("2" %in% names(stop_mass)) stop_mass[["2"]] else 0,
      graduate_look_2 = if ("2" %in% names(graduate_mass)) graduate_mass[["2"]] else 0,
      final_go = fg, final_consider = fc, final_no_go = fn,
      no_go_probability = sum(stop_mass) + fn,
      graduation_probability = sum(graduate_mass),
      terminal_probability = sum(stop_mass) + sum(graduate_mass) + fg + fc + fn,
      sample_size_2 = if ("2" %in% names(sample_mass)) sample_mass[["2"]] else 0,
      sample_size_4 = if ("4" %in% names(sample_mass)) sample_mass[["4"]] else 0,
      expected_sample_size = expected
    )
  }
}

tail_rows <- do.call(rbind, lapply(c(-0.5, 0, 0.25, 0.5), function(delta) {
  data.frame(control_alpha = 1, control_beta = 2, treatment_alpha = 2,
             treatment_beta = 1, margin = delta,
             probability_above = beta_difference_above(1, 2, 2, 1, delta))
}))
write.csv(do.call(rbind, state_rows), paste0(prefix, "monitor-paths.csv"), row.names = FALSE)
write.csv(do.call(rbind, path_rows), paste0(prefix, "path-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, oc_rows), paste0(prefix, "exact-oc.csv"), row.names = FALSE)
write.csv(tail_rows, paste0(prefix, "beta-tail-examples.csv"), row.names = FALSE)
