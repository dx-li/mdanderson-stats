# Independent base-R oracle for BOP2-DC's complete Normal endpoint.
# Uses the paper's NIG conjugate formulas directly; no repository code/packages.
options(warn=2, digits=17)
args <- commandArgs(trailingOnly=TRUE)
root <- if (length(args)) args[[1]] else "."
fixture_dir <- file.path(root, "tests", "fixtures")
case_file <- file.path(fixture_dir, "bop2-dc-normal-cases.csv")
path_file <- file.path(fixture_dir, "bop2-dc-normal-paths.csv")
cases <- read.csv(case_file, stringsAsFactors=FALSE, check.names=FALSE)
paths <- read.csv(path_file, stringsAsFactors=FALSE, check.names=FALSE)

parse_num <- function(text) as.numeric(strsplit(text, ";", fixed=TRUE)[[1]])
posterior <- function(y, prior_mean, prior_precision, prior_shape, prior_scale) {
  n <- length(y)
  # Center both data and prior at the first observation before summing.
  origin <- y[[1]]
  yc <- y - origin
  prior_centered <- prior_mean - origin
  ybar_centered <- mean(yc)
  kn <- prior_precision + n
  an <- prior_shape + n/2
  mn_centered <- (prior_precision*prior_centered + n*ybar_centered)/kn
  sse <- sum((yc-ybar_centered)^2)
  bn <- prior_scale + sse/2 +
    prior_precision*n*(ybar_centered-prior_centered)^2/(2*kn)
  if (!is.finite(kn) || !is.finite(an) || !is.finite(bn) ||
      kn <= 0 || an <= 0 || bn <= 0)
    stop("unrepresentable NIG posterior")
  list(origin=origin, centered_location=mn_centered,
       location=origin+mn_centered, precision=kn, shape=an,
       variance_scale=bn, df=2*an, t_scale=sqrt(bn/(an*kn)))
}
tail_prob <- function(theta, post) {
  # Subtract the common origin first: avoids loss at large additive offsets.
  standardized <- (theta-post$origin-post$centered_location)/post$t_scale
  pt(standardized, df=post$df, lower.tail=FALSE)
}
decision <- function(n, N, looks, p_lrv, p_cmv, lambda_lrv, lambda_cmv,
                     gamma_lrv, gamma_cmv) {
  if (n == N) {
    go <- p_lrv > lambda_lrv && p_cmv > lambda_cmv
    no_go <- p_lrv < lambda_lrv && p_cmv < lambda_cmv
    return(if (go) "final_go" else if (no_go) "final_no_go" else "final_consider")
  }
  if (n %in% looks[-length(looks)]) {
    stop_no_go <- p_lrv < lambda_lrv*(n/N)^gamma_lrv &&
      p_cmv < lambda_cmv*(n/N)^gamma_cmv
    if (stop_no_go) return("stop_no_go")
  }
  "continue"
}

case_rows <- list()
for (i in seq_len(nrow(cases))) {
  cfg <- cases[i,]
  # Construct exactly representable offsets arithmetically. R decimal parsing
  # can lose fractional bits around 1e15 before the statistical calculation.
  y <- parse_num(cfg$observations) + cfg$observation_offset
  cfg$prior_mean <- cfg$prior_mean + cfg$observation_offset
  cfg$theta_lrv <- cfg$theta_lrv + cfg$observation_offset
  cfg$theta_cmv <- cfg$theta_cmv + cfg$observation_offset
  looks <- parse_num(cfg$looks)
  evaluate_n <- sort(unique(c(min(2L, length(y)), length(y))))
  for (n in evaluate_n) {
    post <- posterior(y[seq_len(n)], cfg$prior_mean, cfg$prior_precision,
                      cfg$prior_shape, cfg$prior_scale)
    pl <- tail_prob(cfg$theta_lrv, post)
    pc <- tail_prob(cfg$theta_cmv, post)
    action <- decision(n, cfg$max_subjects, looks, pl, pc, cfg$lambda_lrv,
                       cfg$lambda_cmv, cfg$gamma_lrv, cfg$gamma_cmv)
    case_rows[[length(case_rows)+1L]] <- data.frame(
      case_id=cfg$case_id, n=n, location_offset=post$origin,
      posterior_centered_location=post$centered_location,
      posterior_location=post$location,
      posterior_precision=post$precision, posterior_shape=post$shape,
      posterior_variance_scale=post$variance_scale, posterior_df=post$df,
      posterior_t_scale=post$t_scale, probability_lrv=pl,
      probability_cmv=pc, decision=action)
  }
}
case_reference <- do.call(rbind, case_rows)
write.csv(case_reference, file.path(fixture_dir, "bop2-dc-normal-reference.csv"),
          row.names=FALSE, na="")

# Fixed path tape gives an independent replay/absorption and empirical MCSE oracle.
# All six outcomes per path are supplied, but only observed prefixes are analyzed.
replay <- list()
N <- 6L
looks <- c(2L, 4L, 6L)
cfg <- list(prior_mean=0.15, prior_precision=0.7, prior_shape=2,
            prior_scale=1, theta_lrv=0, theta_cmv=1,
            lambda_lrv=0.75, lambda_cmv=0.4, gamma_lrv=0.5, gamma_cmv=0.5)
for (id in sort(unique(paths$trial))) {
  y <- paths$value[paths$trial == id][order(paths$index[paths$trial == id])]
  if (length(y) != N) stop("path tape must have exactly six observations per trial")
  ended <- FALSE
  for (n in looks) {
    post <- posterior(y[seq_len(n)], cfg$prior_mean, cfg$prior_precision,
                      cfg$prior_shape, cfg$prior_scale)
    pl <- tail_prob(cfg$theta_lrv, post)
    pc <- tail_prob(cfg$theta_cmv, post)
    action <- decision(n, N, looks, pl, pc, cfg$lambda_lrv,
                       cfg$lambda_cmv, cfg$gamma_lrv, cfg$gamma_cmv)
    replay[[length(replay)+1L]] <- data.frame(
      trial=id, sample_size=n, posterior_location=post$location,
      posterior_df=post$df, posterior_scale=post$t_scale,
      probability_lrv=pl, probability_cmv=pc, decision=action)
    if (action != "continue") {
      ended <- TRUE
      break
    }
  }
  if (!ended) stop("path did not reach a terminal final decision")
}
replay_reference <- do.call(rbind, replay)
write.csv(replay_reference, file.path(fixture_dir, "bop2-dc-normal-replay-reference.csv"),
          row.names=FALSE, na="")

labels <- c("stop_no_go", "final_go", "final_consider", "final_no_go")
terminal <- replay_reference[!duplicated(replay_reference$trial, fromLast=TRUE),]
counts <- as.integer(table(factor(terminal$decision, levels=labels)))
probability <- counts/nrow(terminal)
summary <- data.frame(
  decision=labels, count=counts, probability=probability,
  binomial_mcse=sqrt(probability*(1-probability)/nrow(terminal)))
enrollment <- terminal$sample_size
summary$mean_enrollment <- mean(enrollment)
summary$enrollment_mcse <- sd(enrollment)/sqrt(length(enrollment))
write.csv(summary, file.path(fixture_dir, "bop2-dc-normal-replay-summary.csv"),
          row.names=FALSE, na="")
cat("Wrote independent NIG posterior, strict-decision and fixed-path references.\n")
