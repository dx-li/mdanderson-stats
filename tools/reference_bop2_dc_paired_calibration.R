#!/usr/bin/env Rscript

# Independent exhaustive path reference for the paired BOP2-DC finite-grid
# calibrator. Base R only. This script deliberately does not call package or
# Python code; it enumerates all 4^4 patient outcome paths for each joint truth.

args <- commandArgs(trailingOnly = TRUE)
repo <- if (length(args)) args[[1]] else "."
output <- file.path(repo, "tests", "fixtures", "bop2-dc-paired-calibration.csv")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

N <- 4L
looks <- c(2L, 4L)
prior <- c(0.3, 0.2, 0.4, 0.1)
limits <- c(false_go = 1, false_no_go = 1, false_consider = 1)

# Candidate grids are rows of two endpoint-specific controls. The nested loop
# order matches Python itertools.product: the first grid is outermost.
lambda_lrv_grid <- rbind(c(0.4, 0.6), c(0.8, 0.9))
lambda_cmv_grid <- rbind(c(0.15, 0.25), c(0.45, 0.55))
gamma_lrv_grid <- rbind(c(0.0, 0.5), c(0.5, 0.0))
gamma_cmv_grid <- rbind(c(0.5, 0.5))

path_categories <- as.matrix(expand.grid(rep(list(1:4), N)))
path_weights <- function(probability) {
  apply(path_categories, 1, function(path) prod(probability[path]))
}

truths <- list(
  multiple_efficacy = list(
    lrv = c(0.2, 0.15), cmv = c(0.5, 0.45),
    futile_margin = c(0.2, 0.2), effective_margin = c(0.6, 0.5),
    futile_p11 = c(0, 0.04, 0.2), effective_p11 = c(0.1, 0.3, 0.5)
  ),
  efficacy_toxicity = list(
    lrv = c(0.2, 0.5), cmv = c(0.5, 0.2),
    futile_margin = c(0.2, 0.5), effective_margin = c(0.6, 0.15),
    futile_p11 = c(0, 0.1, 0.2), effective_p11 = c(0, 0.09, 0.15)
  )
)

make_joint <- function(margins, p11) {
  c(p11, margins[[1]] - p11, margins[[2]] - p11,
    1 - margins[[1]] - margins[[2]] + p11)
}

posterior_tails <- function(endpoint, counts, n, prior, lrv, cmv) {
  if (endpoint == "multiple_efficacy") {
    success <- c(counts[[1]] + counts[[2]], counts[[1]] + counts[[3]])
    alpha <- c(prior[[1]] + prior[[2]], prior[[1]] + prior[[3]]) + success
    beta <- c(prior[[3]] + prior[[4]], prior[[2]] + prior[[4]]) + n - success
    return(cbind(
      pbeta(lrv, alpha, beta, lower.tail = FALSE),
      pbeta(cmv, alpha, beta, lower.tail = FALSE)
    ))
  }

  efficacy <- counts[[1]] + counts[[2]]
  toxicity <- counts[[1]] + counts[[3]]
  efficacy_alpha <- prior[[1]] + prior[[2]] + efficacy
  efficacy_beta <- prior[[3]] + prior[[4]] + n - efficacy
  toxicity_alpha <- prior[[1]] + prior[[3]] + toxicity
  toxicity_beta <- prior[[2]] + prior[[4]] + n - toxicity
  rbind(
    c(
      pbeta(lrv[[1]], efficacy_alpha, efficacy_beta, lower.tail = FALSE),
      pbeta(cmv[[1]], efficacy_alpha, efficacy_beta, lower.tail = FALSE)
    ),
    c(pbeta(lrv[[2]], toxicity_alpha, toxicity_beta),
      pbeta(cmv[[2]], toxicity_alpha, toxicity_beta))
  )
}

scenario_oc <- function(endpoint, joint, params, lrv, cmv) {
  weights <- path_weights(joint)
  stop_n2 <- final_go <- final_consider <- final_no_go <- numeric(nrow(path_categories))
  sample_size <- rep(N, nrow(path_categories))
  for (i in seq_len(nrow(path_categories))) {
    categories <- path_categories[i, ]
    counts2 <- tabulate(categories[1:2], nbins = 4L)
    tails2 <- posterior_tails(endpoint, counts2, 2L, prior, lrv, cmv)
    bad <- vapply(1:2, function(j) {
      tails2[j, 1] < params$lambda_lrv[[j]] * (2 / N)^params$gamma_lrv[[j]] &&
        tails2[j, 2] < params$lambda_cmv[[j]] * (2 / N)^params$gamma_cmv[[j]]
    }, logical(1))
    stop <- if (endpoint == "multiple_efficacy") all(bad) else any(bad)
    if (stop) {
      stop_n2[[i]] <- 1
      sample_size[[i]] <- 2
      next
    }

    counts4 <- tabulate(categories, nbins = 4L)
    tails4 <- posterior_tails(endpoint, counts4, N, prior, lrv, cmv)
    endpoint_go <- vapply(1:2, function(j) {
      tails4[j, 1] > params$lambda_lrv[[j]] && tails4[j, 2] > params$lambda_cmv[[j]]
    }, logical(1))
    endpoint_no <- vapply(1:2, function(j) {
      tails4[j, 1] < params$lambda_lrv[[j]] && tails4[j, 2] < params$lambda_cmv[[j]]
    }, logical(1))
    if (endpoint == "multiple_efficacy") {
      final_go[[i]] <- as.numeric(any(endpoint_go))
      final_no_go[[i]] <- as.numeric(all(endpoint_no))
    } else {
      final_go[[i]] <- as.numeric(all(endpoint_go))
      final_no_go[[i]] <- as.numeric(any(endpoint_no))
    }
    final_consider[[i]] <- 1 - final_go[[i]] - final_no_go[[i]]
  }
  stop_probability <- sum(weights * stop_n2)
  go_probability <- sum(weights * final_go)
  consider_probability <- sum(weights * final_consider)
  no_go_probability <- sum(weights * final_no_go)
  list(
    stop = c(stop_probability, 0),
    go = go_probability,
    consider = consider_probability,
    no_go = no_go_probability,
    sample_size_probability = c(stop_probability, 1 - stop_probability),
    expected_sample_size = sum(weights * sample_size)
  )
}

rows <- list()
row_index <- 0L
for (endpoint in names(truths)) {
  truth <- truths[[endpoint]]
  for (association_id in 1:3) {
    futile_joint <- make_joint(truth$futile_margin, truth$futile_p11[[association_id]])
    effective_joint <- make_joint(truth$effective_margin, truth$effective_p11[[association_id]])
    candidates <- list()
    candidate_index <- 0L
    for (i1 in seq_len(nrow(lambda_lrv_grid))) {
      for (i2 in seq_len(nrow(lambda_cmv_grid))) {
        for (i3 in seq_len(nrow(gamma_lrv_grid))) {
          for (i4 in seq_len(nrow(gamma_cmv_grid))) {
            candidate_index <- candidate_index + 1L
            params <- list(
              lambda_lrv = lambda_lrv_grid[i1, ],
              lambda_cmv = lambda_cmv_grid[i2, ],
              gamma_lrv = gamma_lrv_grid[i3, ],
              gamma_cmv = gamma_cmv_grid[i4, ]
            )
            futile <- scenario_oc(endpoint, futile_joint, params, truth$lrv, truth$cmv)
            effective <- scenario_oc(endpoint, effective_joint, params, truth$lrv, truth$cmv)
            fgr <- futile$go
            fngr <- effective$stop[[1]] + effective$no_go
            cgr <- effective$go
            fcr <- max(futile$consider, effective$consider)
            feasible <- fgr <= limits[["false_go"]] && fngr <= limits[["false_no_go"]] &&
              fcr <= limits[["false_consider"]]
            candidates[[candidate_index]] <- list(
              params = params, futile = futile, effective = effective,
              fgr = fgr, fngr = fngr, cgr = cgr, fcr = fcr,
              feasible = feasible
            )
          }
        }
      }
    }

    feasible_indices <- which(vapply(candidates, `[[`, logical(1), "feasible"))
    cgr_order <- feasible_indices[order(
      -vapply(candidates[feasible_indices], `[[`, numeric(1), "cgr"),
      vapply(candidates[feasible_indices], function(x) x$futile$expected_sample_size, numeric(1)),
      feasible_indices
    )]
    ess_order <- feasible_indices[order(
      vapply(candidates[feasible_indices], function(x) x$futile$expected_sample_size, numeric(1)),
      -vapply(candidates[feasible_indices], `[[`, numeric(1), "cgr"),
      feasible_indices
    )]
    selected_cgr <- cgr_order[[1]] - 1L
    selected_ess <- ess_order[[1]] - 1L

    for (candidate_index in seq_along(candidates)) {
      candidate <- candidates[[candidate_index]]
      params <- candidate$params
      row_index <- row_index + 1L
      rows[[row_index]] <- data.frame(
        endpoint = endpoint,
        association_id = association_id,
        futile_p11 = truth$futile_p11[[association_id]],
        effective_p11 = truth$effective_p11[[association_id]],
        candidate_index = candidate_index - 1L,
        lambda_lrv_1 = params$lambda_lrv[[1]],
        lambda_lrv_2 = params$lambda_lrv[[2]],
        lambda_cmv_1 = params$lambda_cmv[[1]],
        lambda_cmv_2 = params$lambda_cmv[[2]],
        gamma_lrv_1 = params$gamma_lrv[[1]],
        gamma_lrv_2 = params$gamma_lrv[[2]],
        gamma_cmv_1 = params$gamma_cmv[[1]],
        gamma_cmv_2 = params$gamma_cmv[[2]],
        futile_stop_n2 = candidate$futile$stop[[1]],
        futile_stop_n4 = candidate$futile$stop[[2]],
        futile_final_go = candidate$futile$go,
        futile_final_consider = candidate$futile$consider,
        futile_final_no_go = candidate$futile$no_go,
        futile_sample_size_p_n2 = candidate$futile$sample_size_probability[[1]],
        futile_sample_size_p_n4 = candidate$futile$sample_size_probability[[2]],
        futile_ess = candidate$futile$expected_sample_size,
        effective_stop_n2 = candidate$effective$stop[[1]],
        effective_stop_n4 = candidate$effective$stop[[2]],
        effective_final_go = candidate$effective$go,
        effective_final_consider = candidate$effective$consider,
        effective_final_no_go = candidate$effective$no_go,
        effective_sample_size_p_n2 = candidate$effective$sample_size_probability[[1]],
        effective_sample_size_p_n4 = candidate$effective$sample_size_probability[[2]],
        effective_ess = candidate$effective$expected_sample_size,
        fgr = candidate$fgr,
        fngr = candidate$fngr,
        cgr = candidate$cgr,
        fcr = candidate$fcr,
        feasible = candidate$feasible,
        false_go_limit = limits[["false_go"]],
        false_no_go_limit = limits[["false_no_go"]],
        false_consider_limit = limits[["false_consider"]],
        selected_cgr_index = selected_cgr,
        selected_ess_index = selected_ess,
        stringsAsFactors = FALSE
      )
    }
  }
}

reference <- do.call(rbind, rows)
write.csv(reference, output, row.names = FALSE, na = "")
cat("Wrote", nrow(reference), "paired calibration rows to", output, "\n")
