# Independent base-R exact finite-grid calibration oracle for randomized
# binary BOP2-DC. This script is standalone and does not call Python.
# Usage: Rscript reference_bop2_dc_randomized_binary_calibration.R [prefix]
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[[1]] else "tests/fixtures/bop2-dc-randomized-binary-calibration-"
settings <- read.csv(paste0(prefix, "settings.csv"), stringsAsFactors = FALSE,
                     colClasses = "character", check.names = FALSE)
truths <- read.csv(paste0(prefix, "truths.csv"), stringsAsFactors = FALSE,
                   colClasses = "character", check.names = FALSE)
parse_values <- function(x) as.numeric(strsplit(x, ";", fixed = TRUE)[[1]])
parse_ints <- function(x) as.integer(strsplit(x, ";", fixed = TRUE)[[1]])
parse_flag <- function(x) as.integer(x) == 1L

# Integer-shape Beta CDF from a finite binomial sum.
beta_cdf_integer <- function(x, a, b) {
  stopifnot(a >= 1, b >= 1, a == as.integer(a), b == as.integer(b))
  if (x <= 0) return(0)
  if (x >= 1) return(1)
  total <- a + b - 1
  sum(vapply(a:total, function(k) choose(total, k) * x^k * (1 - x)^(total - k), numeric(1)))
}

# P(T-C > delta), integrating treatment density against the control CDF.
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
decision_at <- function(pl, pc, look, s, parameters) {
  n <- as.integer(s$max_subjects)
  lambda_lrv <- parameters[[1]]; lambda_cmv <- parameters[[2]]
  gamma_lrv <- parameters[[3]]; gamma_cmv <- parameters[[4]]
  if (look < n) {
    no_lrv <- lambda_lrv * (look / n)^gamma_lrv
    no_cmv <- lambda_cmv * (look / n)^gamma_cmv
    no_go <- pl < no_lrv && pc < no_cmv
    graduate <- FALSE
    if (parse_flag(s$graduate_at_interim)) {
      graduate <- pl > obf_cutoff(lambda_lrv, look, n) &&
        pc > obf_cutoff(lambda_cmv, look, n)
    }
    if (no_go && graduate) stop("overlapping interim rules in oracle input")
    if (no_go) return("stop_no_go")
    if (graduate) return("graduate")
    return("continue")
  }
  go <- pl > lambda_lrv && pc > lambda_cmv
  no_go <- pl < lambda_lrv && pc < lambda_cmv
  if (go) return("final_go")
  if (no_go) return("final_no_go")
  "final_consider"
}
posterior_at <- function(path, look, s, arms) {
  y <- as.integer(strsplit(path, "", fixed = TRUE)[[1]][seq_len(look)])
  current_arms <- arms[seq_len(look)]
  cidx <- current_arms == 0L; tidx <- current_arms == 1L
  cn <- sum(cidx); tn <- sum(tidx); cy <- sum(y[cidx]); ty <- sum(y[tidx])
  ac <- as.integer(s$control_alpha) + cy
  bc <- as.integer(s$control_beta) + cn - cy
  at <- as.integer(s$treatment_alpha) + ty
  bt <- as.integer(s$treatment_beta) + tn - ty
  list(p_lrv = beta_difference_above(ac, bc, at, bt, as.numeric(s$theta_lrv)),
       p_cmv = beta_difference_above(ac, bc, at, bt, as.numeric(s$theta_cmv)))
}

metric_rows <- list(); decision_rows <- list(); selected_rows <- list()
for (si in seq_len(nrow(settings))) {
  s <- settings[si, , drop = FALSE]
  config_id <- s$config_id[[1]]
  n <- as.integer(s$max_subjects)
  looks <- parse_ints(s$looks)
  arms <- as.integer(strsplit(s$arm_assignments, "", fixed = TRUE)[[1]])
  if (length(arms) != n || tail(looks, 1) != n || any(!looks %in% seq_len(n))) {
    stop(paste("invalid schedule for", config_id))
  }
  # Match itertools.product(grid_1,...,grid_4): rightmost axis varies fastest.
  grids <- list(parse_values(s$lambda_lrv_grid), parse_values(s$lambda_cmv_grid),
                parse_values(s$gamma_lrv_grid), parse_values(s$gamma_cmv_grid))
  candidates <- expand.grid(rev(grids), KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  candidates <- candidates[, rev(seq_along(grids)), drop = FALSE]
  names(candidates) <- c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
  candidates$candidate_index <- seq_len(nrow(candidates)) - 1L
  truths_for_config <- truths[truths$config_id == config_id, , drop = FALSE]
  if (!setequal(truths_for_config$scenario, c("futile", "effective"))) {
    stop(paste("need exactly futile and effective truths for", config_id))
  }
  path_grid <- expand.grid(rep(list(0:1), n), KEEP.OUT.ATTRS = FALSE)
  path_grid$path <- apply(path_grid, 1, paste0, collapse = "")
  config_metric_rows <- list()

  for (ci in seq_len(nrow(candidates))) {
    p <- unlist(candidates[ci, c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")],
                use.names = FALSE)
    path_decisions <- vector("list", nrow(path_grid))
    for (pi in seq_len(nrow(path_grid))) {
      path <- path_grid$path[[pi]]
      terminal_action <- "continue"; terminal_look <- n
      for (look in looks) {
        post <- posterior_at(path, look, s, arms)
        action <- decision_at(post$p_lrv, post$p_cmv, look, s, p)
        if (action != "continue") {
          terminal_action <- action; terminal_look <- look
          break
        }
      }
      path_decisions[[pi]] <- c(action = terminal_action, look = terminal_look)
    }
    scenario_data <- list()
    for (scenario in c("futile", "effective")) {
      tr <- truths_for_config[truths_for_config$scenario == scenario, , drop = FALSE]
      if (nrow(tr) != 1L) stop(paste("duplicate truth for", config_id, scenario))
      pc <- as.numeric(tr$control_probability); pt <- as.numeric(tr$treatment_probability)
      action_names <- c("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
      decision_mass <- matrix(0, nrow = length(looks), ncol = length(action_names),
                              dimnames = list(NULL, action_names))
      sample_mass <- numeric(length(looks)); expected_n <- 0
      for (pi in seq_len(nrow(path_grid))) {
        y <- as.integer(strsplit(path_grid$path[[pi]], "", fixed = TRUE)[[1]])
        response_p <- ifelse(arms == 0L, pc, pt)
        path_probability <- prod(ifelse(y == 1L, response_p, 1 - response_p))
        terminal_action <- unname(path_decisions[[pi]][["action"]])
        terminal_look <- as.integer(path_decisions[[pi]][["look"]])
        li <- match(terminal_look, looks)
        if (is.na(li)) stop("terminal look not in scheduled looks")
        sample_mass[[li]] <- sample_mass[[li]] + path_probability
        expected_n <- expected_n + path_probability * terminal_look
        if (terminal_action == "continue") stop("path did not reach terminal decision")
        decision_mass[li, terminal_action] <- decision_mass[li, terminal_action] + path_probability
      }
      for (li in seq_along(looks)) for (action in action_names) {
        decision_rows[[length(decision_rows) + 1L]] <- data.frame(
          config_id = config_id, candidate_index = candidates$candidate_index[[ci]],
          scenario = scenario, look = looks[[li]], decision = action,
          probability = decision_mass[li, action],
          sample_size_probability = sample_mass[[li]], stringsAsFactors = FALSE)
      }
      scenario_data[[scenario]] <- list(decision = decision_mass, sample = sample_mass,
                                        expected = expected_n)
    }
    futile <- scenario_data$futile; effective <- scenario_data$effective
    fg <- sum(futile$decision[, "graduate"]) + futile$decision[length(looks), "final_go"]
    fn <- sum(effective$decision[, "stop_no_go"]) +
      effective$decision[length(looks), "final_no_go"]
    cgr <- sum(effective$decision[, "graduate"]) +
      effective$decision[length(looks), "final_go"]
    fcr <- max(futile$decision[length(looks), "final_consider"],
               effective$decision[length(looks), "final_consider"])
    fc_input <- s$false_consider_limit[[1]]
    fc_limit <- if (fc_input == "NA") NA_real_ else as.numeric(fc_input)
    feasible <- fg <= as.numeric(s$false_go_limit) &&
      fn <= as.numeric(s$false_no_go_limit) &&
      (is.na(fc_limit) || fcr <= fc_limit)
    row <- data.frame(
      config_id = config_id, candidate_index = candidates$candidate_index[[ci]],
      lambda_lrv = p[[1]], lambda_cmv = p[[2]], gamma_lrv = p[[3]], gamma_cmv = p[[4]],
      false_go_rate = fg, false_no_go_rate = fn, correct_go_rate = cgr,
      false_consider_rate = fcr, expected_n_futile = futile$expected,
      expected_n_effective = effective$expected,
      false_go_limit = as.numeric(s$false_go_limit),
      false_no_go_limit = as.numeric(s$false_no_go_limit),
      false_consider_limit = fc_limit, feasible = feasible,
      objective = s$objective[[1]], stringsAsFactors = FALSE)
    metric_rows[[length(metric_rows) + 1L]] <- row
    config_metric_rows[[length(config_metric_rows) + 1L]] <- row
  }
  config_metrics <- do.call(rbind, config_metric_rows)
  feasible_ids <- which(config_metrics$feasible)
  chosen <- NA_integer_
  if (length(feasible_ids)) {
    objective <- s$objective[[1]]
    if (objective == "cgr") {
      rank <- order(-config_metrics$correct_go_rate[feasible_ids],
                    config_metrics$expected_n_futile[feasible_ids], feasible_ids)
    } else if (objective == "ess_futile") {
      rank <- order(config_metrics$expected_n_futile[feasible_ids],
                    -config_metrics$correct_go_rate[feasible_ids], feasible_ids)
    } else stop(paste("unsupported objective", objective))
    chosen <- config_metrics$candidate_index[feasible_ids[rank[[1]]]]
  }
  selected_rows[[length(selected_rows) + 1L]] <- data.frame(
    config_id = config_id, objective = s$objective[[1]], selected_index = chosen,
    selected_lambda_lrv = if (is.na(chosen)) NA_real_ else config_metrics$lambda_lrv[config_metrics$candidate_index == chosen][[1]],
    selected_lambda_cmv = if (is.na(chosen)) NA_real_ else config_metrics$lambda_cmv[config_metrics$candidate_index == chosen][[1]],
    selected_gamma_lrv = if (is.na(chosen)) NA_real_ else config_metrics$gamma_lrv[config_metrics$candidate_index == chosen][[1]],
    selected_gamma_cmv = if (is.na(chosen)) NA_real_ else config_metrics$gamma_cmv[config_metrics$candidate_index == chosen][[1]],
    feasible_count = length(feasible_ids), no_feasible_candidate = !length(feasible_ids),
    stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, metric_rows), paste0(prefix, "candidate-metrics.csv"), row.names = FALSE)
write.csv(do.call(rbind, decision_rows), paste0(prefix, "candidate-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, selected_rows), paste0(prefix, "selected.csv"), row.names = FALSE)
