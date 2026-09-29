# Independent base-R full-grid calibration oracle for BOP2-DC Normal.
# Input paths are truth-centered outcomes; every candidate reuses each supplied path.
# No package beyond base R (stats/utils, attached by default) and no repository code.
options(warn=2, digits=17)
args <- commandArgs(trailingOnly=TRUE)
prefix <- if (length(args)) args[[1]] else
  "research/raw/bop2-dc-normal-calibration-"
settings <- read.csv(paste0(prefix, "settings.csv"), stringsAsFactors=FALSE,
                     check.names=FALSE)
grid <- read.csv(paste0(prefix, "grid.csv"), stringsAsFactors=FALSE,
                 check.names=FALSE)
paths <- read.csv(paste0(prefix, "paths.csv"), stringsAsFactors=FALSE,
                  check.names=FALSE)
if (nrow(settings) != 1L || nrow(grid) < 1L || nrow(paths) < 1L)
  stop("settings must have one row and grid/paths must be nonempty")
s <- settings[1,]
looks <- as.integer(strsplit(s$looks, ";", fixed=TRUE)[[1]])
truths <- c(futile=as.numeric(s$theta_futile), effective=as.numeric(s$theta_effective))
labels <- c("stop_no_go", "final_go", "final_consider", "final_no_go")
required_grid <- c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
if (!all(required_grid %in% names(grid))) stop("grid requires lambda_lrv/lambda_cmv/gamma_lrv/gamma_cmv")
if (!all(c("phase","truth","trial","patient","value") %in% names(paths)))
  stop("paths require phase, truth, trial, patient and truth-centered value")
if (!all(c("prior_mean","theta_lrv","theta_cmv","prior_precision","prior_shape",
           "prior_scale","max_subjects","looks","theta_futile","theta_effective",
           "false_go_limit","false_no_go_limit","false_consider_limit") %in% names(s)))
  stop("settings lacks a required model/trial/constraint field")
N <- as.integer(s$max_subjects)
if (N < 1L || N > 1000L || any(!is.finite(looks)) ||
    any(looks < 1L | looks > N) || tail(looks,1L) != N ||
    any(diff(looks) <= 0L)) stop("invalid max_subjects/looks")
if (!(s$prior_precision > 0 && s$prior_shape > 0 && s$prior_scale > 0) ||
    !(s$theta_lrv < s$theta_cmv) ||
    !(truths[["futile"]] < truths[["effective"]]) ||
    truths[["effective"]] < s$theta_cmv)
  stop("invalid prior, thresholds or truth ordering")
if (any(!is.finite(as.matrix(grid))) ||
    any(grid$lambda_lrv <= 0 | grid$lambda_lrv >= 1) ||
    any(grid$lambda_cmv <= 0 | grid$lambda_cmv >= 1) ||
    any(grid$gamma_lrv < 0 | grid$gamma_lrv > 1) ||
    any(grid$gamma_cmv < 0 | grid$gamma_cmv > 1))
  stop("candidate grid has invalid values")
if (s$false_go_limit < 0 || s$false_go_limit > 1 ||
    s$false_no_go_limit < 0 || s$false_no_go_limit > 1 ||
    (!is.na(s$false_consider_limit) &&
       (s$false_consider_limit < 0 || s$false_consider_limit > 1)))
  stop("error limits must lie in [0,1]")
if (any(!is.finite(paths$value)) ||
    any(!paths$phase %in% c("calibration","validation")) ||
    any(!paths$truth %in% names(truths)) ||
    any(paths$patient < 1L | paths$patient > N | paths$patient != as.integer(paths$patient)))
  stop("invalid centered path tape")

# Paths for the two means must be identical after centering; phase streams may differ.
for (phase in c("calibration","validation")) {
  parts <- lapply(names(truths), function(truth) {
    x <- paths[paths$phase==phase & paths$truth==truth,
               c("trial","patient","value")]
    x[order(x$trial,x$patient),]
  })
  if (nrow(parts[[1]]) != nrow(parts[[2]]) ||
      !all(parts[[1]]$trial == parts[[2]]$trial) ||
      !all(parts[[1]]$patient == parts[[2]]$patient) ||
      !identical(parts[[1]]$value,parts[[2]]$value))
    stop("truth-centered paths must be exactly common across truths within phase")
  if (nrow(parts[[1]]) == 0L) stop("both calibration and validation paths are required")
}

post <- function(y, truth) {
  # y and all location inputs are relative to the truth mean, avoiding large offsets.
  n <- length(y)
  origin <- y[[1]]
  yc <- y-origin
  prior_centered <- s$prior_mean-truth-origin
  ybar <- mean(yc)
  kn <- s$prior_precision+n
  an <- s$prior_shape+n/2
  mn_centered <- (s$prior_precision*prior_centered+n*ybar)/kn
  bn <- s$prior_scale+sum((yc-ybar)^2)/2+
    s$prior_precision*n*(ybar-prior_centered)^2/(2*kn)
  if (!is.finite(kn) || !is.finite(an) || !is.finite(bn) ||
      kn <= 0 || an <= 0 || bn <= 0) stop("unrepresentable NIG posterior")
  list(origin=origin, location=mn_centered, df=2*an, t_scale=sqrt(bn/(an*kn)))
}
tail <- function(theta, truth, p) {
  centered_threshold <- s[[theta]]-truth-p$origin
  pt((centered_threshold-p$location)/p$t_scale, df=p$df, lower.tail=FALSE)
}
classify <- function(n, pl, pc, candidate) {
  if (n == N) {
    go <- pl > candidate$lambda_lrv && pc > candidate$lambda_cmv
    no_go <- pl < candidate$lambda_lrv && pc < candidate$lambda_cmv
    return(if (go) "final_go" else if (no_go) "final_no_go" else "final_consider")
  }
  if (n %in% looks[-length(looks)]) {
    l <- candidate$lambda_lrv*(n/N)^candidate$gamma_lrv
    c <- candidate$lambda_cmv*(n/N)^candidate$gamma_cmv
    if (pl < l && pc < c) return("stop_no_go")
  }
  "continue"
}
replay <- function(data, truth, candidate) {
  data <- data[order(data$patient),]
  if (nrow(data) != N || !identical(as.integer(data$patient),seq_len(N)))
    stop("each path must contain patients 1..N exactly once")
  for (n in looks) {
    y <- data$value[seq_len(n)]
    p <- post(y, truth)
    pl <- tail("theta_lrv",truth,p)
    pc <- tail("theta_cmv",truth,p)
    action <- classify(n,pl,pc,candidate)
    if (action != "continue") return(list(decision=action,n=n))
  }
  stop("path ended without a terminal decision")
}
evaluate <- function(phase, truth_name, candidate) {
  truth <- truths[[truth_name]]
  data <- paths[paths$phase==phase & paths$truth==truth_name,]
  trial_list <- split(data,data$trial)
  if (length(trial_list)==0L) stop("missing phase/truth paths")
  result <- lapply(trial_list,replay,truth=truth,candidate=candidate)
  outcomes <- vapply(result,function(x)x$decision,character(1))
  enrollment <- vapply(result,function(x)x$n,integer(1))
  probabilities <- vapply(labels,function(x)mean(outcomes==x),numeric(1))
  mcse <- sqrt(probabilities*(1-probabilities)/length(outcomes))
  c(setNames(probabilities,paste0("p_",labels)),
    setNames(mcse,paste0("mcse_",labels)),
    mean_enrollment=mean(enrollment),
    enrollment_mcse=if(length(enrollment)>1L) sd(enrollment)/sqrt(length(enrollment)) else NA_real_,
    trials=length(enrollment))
}
records <- list()
candidate_metrics <- vector("list",nrow(grid))
for (i in seq_len(nrow(grid))) {
  candidate <- grid[i,,drop=FALSE]
  values <- list(
    futile=evaluate("calibration","futile",candidate),
    effective=evaluate("calibration","effective",candidate))
  for (truth_name in names(values)) {
    v <- values[[truth_name]]
    records[[length(records)+1L]] <- data.frame(
      candidate=i-1L,phase="calibration",truth=truth_name,
      metric=names(v),value=unname(v))
  }
  futile <- values$futile
  effective <- values$effective
  fg <- unname(futile["p_final_go"])
  fn <- unname(effective["p_stop_no_go"]+effective["p_final_no_go"])
  cg <- unname(effective["p_final_go"])
  fc_futile <- unname(futile["p_final_consider"])
  fc_effective <- unname(effective["p_final_consider"])
  fc <- max(fc_futile,fc_effective)
  feasible <- fg <= s$false_go_limit && fn <= s$false_no_go_limit &&
    (is.na(s$false_consider_limit) || fc <= s$false_consider_limit)
  candidate_metrics[[i]] <- data.frame(
    candidate=i-1L,lambda_lrv=candidate$lambda_lrv,lambda_cmv=candidate$lambda_cmv,
    gamma_lrv=candidate$gamma_lrv,gamma_cmv=candidate$gamma_cmv,
    false_go_rate=fg,false_go_mcse=sqrt(fg*(1-fg)/futile["trials"]),
    false_no_go_rate=fn,false_no_go_mcse=sqrt(fn*(1-fn)/effective["trials"]),
    correct_go_rate=cg,correct_go_mcse=sqrt(cg*(1-cg)/effective["trials"]),
    false_consider_futile=fc_futile,false_consider_futile_mcse=
      sqrt(fc_futile*(1-fc_futile)/futile["trials"]),
    false_consider_effective=fc_effective,false_consider_effective_mcse=
      sqrt(fc_effective*(1-fc_effective)/effective["trials"]),
    futile_expected_sample_size=unname(futile["mean_enrollment"]),
    futile_enrollment_mcse=unname(futile["enrollment_mcse"]),
    effective_expected_sample_size=unname(effective["mean_enrollment"]),
    effective_enrollment_mcse=unname(effective["enrollment_mcse"]),
    feasible=feasible)
}
metrics <- do.call(rbind,candidate_metrics)
selection <- list()
for (objective in c("cgr","ess_futile")) {
  feasible <- metrics[metrics$feasible,]
  if (nrow(feasible)==0L) stop(paste("no feasible candidate for",objective))
  if (objective=="cgr") {
    order_index <- order(-feasible$correct_go_rate,
                         feasible$futile_expected_sample_size,feasible$candidate)
  } else {
    order_index <- order(feasible$futile_expected_sample_size,
                         -feasible$correct_go_rate,feasible$candidate)
  }
  selected <- feasible$candidate[order_index[1]]
  selection[[objective]] <- data.frame(objective=objective,selected_index=selected)
  candidate <- grid[selected+1L,,drop=FALSE]
  validation <- list(
    futile=evaluate("validation","futile",candidate),
    effective=evaluate("validation","effective",candidate))
  for (truth_name in names(validation)) {
    v <- validation[[truth_name]]
    records[[length(records)+1L]] <- data.frame(
      candidate=selected,phase=paste0("validation_",objective),truth=truth_name,
      metric=names(v),value=unname(v))
  }
  fg <- unname(validation$futile["p_final_go"])
  fn <- unname(validation$effective["p_stop_no_go"]+validation$effective["p_final_no_go"])
  fc_futile <- unname(validation$futile["p_final_consider"])
  fc_effective <- unname(validation$effective["p_final_consider"])
  fc <- max(fc_futile,fc_effective)
  validation_feasible <- fg <= s$false_go_limit && fn <= s$false_no_go_limit &&
    (is.na(s$false_consider_limit) || fc <= s$false_consider_limit)
  selection[[objective]]$validation_false_go_rate <- fg
  selection[[objective]]$validation_false_no_go_rate <- fn
  selection[[objective]]$validation_correct_go_rate <- unname(validation$effective["p_final_go"])
  selection[[objective]]$validation_false_consider_rate <- fc
  selection[[objective]]$validation_feasible <- validation_feasible
}
write_exact <- function(data,path) {
  for (name in names(data)) if (is.numeric(data[[name]])) data[[name]] <- sprintf("%.17g",data[[name]])
  write.csv(data,path,row.names=FALSE,na="")
}
write_exact(do.call(rbind,records),paste0(prefix,"reference.csv"))
write_exact(metrics,paste0(prefix,"metrics.csv"))
write_exact(do.call(rbind,selection),paste0(prefix,"selection.csv"))
cat("Evaluated the supplied full grid on common centered paths and both held-out selections.\n")
