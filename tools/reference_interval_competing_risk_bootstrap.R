# Source-pinned intccr coefficient-bootstrap references using fixed row tapes.
# Run from the repository root, optionally as:
#   Rscript tools/reference_interval_competing_risk_bootstrap.R INPUT.csv OUTPUT_DIR
# The ignored research/raw/intccr source cache must be present; no packages are installed.
options(warn = 2, digits = 17)
args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else
  "tests/fixtures/interval-competing-risk-inputs.csv"
output_dir <- if (length(args) >= 2L) args[[2L]] else "tests/fixtures"
if (length(args) > 2L) stop("expected at most INPUT.csv and OUTPUT_DIR")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Hash-check the native wrapper and the focused model/optimizer sources needed
# by its refits. Do not source reference_interval_competing_risk.R here: that
# script also runs unrelated fits and rewrites their fixtures.
expected <- c(
  "intccr/R/ciregic.R" = "e1f2f6e29d84e6f520e0482adfdcc0a48c8f9b90",
  "intccr/R/bssmle.R" = "d122c3ac22a446bfb8071b720c9df962de188ebe",
  "intccr/R/Surv2.R" = "c32f52d7c8c275cda51dd23e9868d9d7e2615fd9",
  "intccr/R/bsderivs.R" = "e468f3fd071e490b6e57f94ffc490776ab56752a",
  "intccr/R/naive_b.R" = "bb46420a92a19bd634f2db6d69e5edb0d6677248",
  "intccr/R/bssmle_lse.R" = "416b818d2e826169e20f917990193736011fa012",
  "alabama/R/constrOptim.nl.R" = "05c93764ceb3eacafd1d2cc06ad8d6b4f7015cf1",
  "numDeriv/R/numDeriv.R" = "394d5f1db1d644fd1b44a2e70f0294ce04fcf50d",
  "numDeriv/R/num2Deriv.R" = "9adbafd4d39dc95d2d0b69283a5fc571bf1e497c",
  "intccr/R/bssmle_se.R" = "c923ec3cd71842c4e4895d34905c635381ade5d6"
)
for (name in names(expected)) {
  actual <- system2(
    "git", c("hash-object", shQuote(file.path("research/raw", name))), stdout = TRUE
  )
  if (!identical(unname(actual), unname(expected[name]))) {
    stop(paste("pinned source hash mismatch or source unavailable:", name))
  }
}
library(splines)
for (name in c(
  "numDeriv/R/numDeriv.R", "numDeriv/R/num2Deriv.R",
  "alabama/R/constrOptim.nl.R", "intccr/R/Surv2.R",
  "intccr/R/bsderivs.R", "intccr/R/naive_b.R",
  "intccr/R/bssmle.R", "intccr/R/bssmle_lse.R", "intccr/R/ciregic.R"
)) {
  source(file.path("research/raw", name))
}

# Keep the source optimizer unchanged and route bssmle's namespace call to it.
native_optimizer <- constrOptim.nl
reference_optimizer <- function(...) do.call(native_optimizer, list(...))
redirect <- function(expr) {
  if (identical(expr, quote(alabama::constrOptim.nl))) return(quote(reference_optimizer))
  if (is.call(expr)) return(as.call(lapply(as.list(expr), redirect)))
  expr
}
body(bssmle) <- redirect(body(bssmle))

input <- read.csv(input_path)
required <- c("v", "u", "event", "x1", "x2")
if (!all(required %in% names(input))) stop("input is missing required columns")
tape_path <- "tests/fixtures/interval-competing-risk-bootstrap-tapes.csv"
tape_frame <- read.csv(tape_path)
case_ids <- sort(unique(tape_frame$case))
if (!identical(case_ids, c(1L, 2L, 3L))) stop("unexpected fixed tape cases")

support_rows <- list()
diagnostic_rows <- list()
slope_rows <- list()
successful_slopes <- list()
for (case in 1:2) {
  tape_case <- tape_frame[tape_frame$case == case, ]
  ids <- as.integer(tape_case$row_id)
  if (length(ids) != nrow(input) || any(ids < 1L | ids > nrow(input))) {
    stop("fixed tape has invalid source-row indices")
  }
  boot <- input[ids, , drop = FALSE]
  fit <- bssmle(
    Surv2(v, u, event = event) ~ x1 + x2,
    data = boot,
    alpha = c(0, 1),
    k = 0.5
  )
  q <- length(fit$varnames)
  m <- (length(fit$beta) - 2L * q) / 2L
  if (q != 2L || m != as.integer(m)) stop("unexpected native coefficient layout")
  slope_index <- (2L * m + 1L):(2L * m + 2L * q)
  slopes <- fit$beta[slope_index]
  tt <- c(boot$v, boot$u[boot$event > 0])
  nk <- floor(0.5 * length(tt)^(1 / 3))
  knots <- if (nk == 0L) numeric() else unique(quantile(
    tt[tt < max(tt) & tt > min(tt)],
    seq(0, 1, by = 1 / (nk + 1L))
  )[(2L):(nk + 1L)])

  support_rows[[case]] <- data.frame(
    case = case,
    minimum = min(tt),
    maximum = max(tt),
    nk = nk,
    knot_index = seq_along(knots),
    knot = as.numeric(knots)
  )
  slope_rows[[case]] <- data.frame(
    case = case,
    parameter = seq_along(slopes),
    value = slopes
  )
  successful_slopes[[case]] <- slopes
  diagnostic_rows[[case]] <- data.frame(
    case = case,
    alpha1 = 0,
    alpha2 = 1,
    k = 0.5,
    n = nrow(boot),
    cause1 = sum(boot$event == 1),
    cause2 = sum(boot$event == 2),
    censored = sum(boot$event == 0),
    convergence = fit$convergence,
    loglikelihood = fit$loglikelihood,
    q = q,
    n_basis = m
  )
}

slopes_matrix <- do.call(rbind, successful_slopes)
native_covariance <- stats::cov(slopes_matrix)
covariance_rows <- data.frame(
  row = rep(seq_len(nrow(native_covariance)), times = ncol(native_covariance)),
  col = rep(seq_len(ncol(native_covariance)), each = nrow(native_covariance)),
  value = as.vector(native_covariance)
)
write.csv(do.call(rbind, support_rows), file.path(output_dir,
  "interval-competing-risk-bootstrap-support.csv"), row.names = FALSE)
write.csv(do.call(rbind, slope_rows), file.path(output_dir,
  "interval-competing-risk-bootstrap-native-slopes.csv"), row.names = FALSE)
write.csv(covariance_rows, file.path(output_dir,
  "interval-competing-risk-bootstrap-native-covariance.csv"), row.names = FALSE)
write.csv(do.call(rbind, diagnostic_rows), file.path(output_dir,
  "interval-competing-risk-bootstrap-diagnostics.csv"), row.names = FALSE)

absent_case <- tape_frame[tape_frame$case == 3L, ]
absent_ids <- as.integer(absent_case$row_id)
absent <- input[absent_ids, , drop = FALSE]
absent_error <- tryCatch({
  bssmle(
    Surv2(v, u, event = event) ~ x1 + x2,
    data = absent,
    alpha = c(0, 1),
    k = 0.5
  )
  "NO_ERROR"
}, error = function(error) conditionMessage(error))
writeLines(absent_error, file.path(output_dir,
  "interval-competing-risk-bootstrap-absent-cause.txt"))
cat("successful replicates:", nrow(slopes_matrix),
    "; covariance denominator:", nrow(slopes_matrix) - 1L,
    "; absent-cause error:", absent_error, "\n")
