# Numerical references from unchanged cmprsk 2.2-12 source at
# f81411e1e3f57822796bae2a6870657e455362e9. Compile src/crr.f to cmprsk.so
# under research/raw/cmprsk before running; no package installation is needed.
# Example on macOS: gfortran -O2 -fPIC -dynamiclib src/crr.f -o cmprsk.so
options(warn=2, digits=17)
library(survival)
source_root <- "research/raw/cmprsk"
stopifnot(unname(tools::md5sum(file.path(source_root,
  c("R/cmprsk.R", "src/crr.f")))) ==
  c("33e69a329c50dd9549bb0931f99b1ac4", "abcc0f1aca5c56602513cfb1451bd19a"))
dyn.load(file.path(source_root, "cmprsk.so"))
source(file.path(source_root, "R/cmprsk.R"))

id <- 1:40
data <- data.frame(time=c(0, rep(1:12, each=3), 13, 14, 15),
  status=c(1, rep(c(1,2,0, 2,1,1, 0,1,2, 1,0,1),3), 2,0,1),
  x1=round(sin(id*1.7) + id/80, 6),
  x2=round(cos(id*.9) - id/100, 6), group=rep(c("A","B","B","A"),10))
write.csv(data, "tests/fixtures/fine-gray-input.csv", row.names=FALSE)
metric_rows <- event_rows <- prediction_rows <- list()
cases <- c("fixed_one", "fixed_groups", "mixed_groups", "time_only", "target_two",
  "zero_censor")
for (case in cases) {
  args <- list(ftime=data$time, fstatus=data$status, gtol=1e-10, maxiter=100)
  if (!(case %in% c("fixed_one", "zero_censor"))) args$cengroup <- data$group
  if (case == "zero_censor") {
    # Put target, competing and censor events at zero. A common positive time
    # translation preserves this fixed-effect model and avoids the native
    # approximate-left-limit wrapper's exact-zero defect.
    args$ftime[2:4] <- 0
    args$ftime <- args$ftime + 1
  }
  if (case != "time_only") args$cov1 <- as.matrix(data[,c("x1","x2")])
  if (case == "mixed_groups") {
    args$cov2 <- as.matrix(data["x1"])
    args$tf <- function(t) matrix(t/10, ncol=1)
  }
  if (case == "time_only") {
    args$cov2 <- as.matrix(data[,c("x1","x2")])
    args$tf <- function(t) cbind(1+t/10,sqrt(1+t/10))
  }
  if (case == "target_two") args$failcode <- 2
  fit <- do.call(crr, args)
  stopifnot(fit$converged, all(is.finite(fit$coef)))
  if (case == "zero_censor") fit$uftime <- fit$uftime - 1
  for (metric in c("coef", "score", "inf", "var", "invinf", "loglik", "loglik.null")) {
    a <- as.matrix(fit[[metric]])
    for (i in seq_len(nrow(a))) for (j in seq_len(ncol(a))) {
      metric_rows[[length(metric_rows)+1L]] <- data.frame(case=case,
        metric=metric, row=i, column=j, value=a[i,j])
    }
  }
  residuals <- matrix(NA_real_, nrow=length(fit$uftime), ncol=3)
  residuals[,seq_len(length(fit$coef))] <- fit$res
  event_rows[[length(event_rows)+1L]] <- data.frame(case=case, time=fit$uftime,
    baseline_increment=fit$bfitj, residual_1=residuals[,1],
    residual_2=residuals[,2], residual_3=residuals[,3])
  for (profile in c("mean", "explicit")) {
    grid <- seq(quantile(data$x1,.025),quantile(data$x1,.975),length.out=5)
    x2 <- if (profile=="mean") mean(data$x2) else .25
    for (i in seq_along(grid)) {
      # Single-profile calls also exercise the native time-covariates-only path.
      pargs <- list(object=fit)
      if (case != "time_only") pargs$cov1 <- c(grid[i],x2)
      if (case == "mixed_groups") pargs$cov2 <- grid[i]
      if (case == "time_only") pargs$cov2 <- c(grid[i],x2)
      native <- do.call(predict.crr, pargs)
      for (time_kind in c("native", "custom")) {
        times <- if (time_kind=="native") sort(unique(c(0,fit$uftime))) else
          c(0,.5,1,2.5,7,12,16)
        index <- findInterval(times, native[,1])
        incidence <- c(0,native[,2])[index+1L]
        prediction_rows[[length(prediction_rows)+1L]] <- data.frame(case=case,
          profile=profile, row=i, x1=grid[i], x2=x2, time_kind=time_kind,
          time=times, incidence=incidence)
      }
    }
  }
}
for (name in c("metric", "event", "prediction")) {
  write.csv(do.call(rbind,get(paste0(name,"_rows"))),
    paste0("tests/fixtures/fine-gray-",name,".csv"),row.names=FALSE)
}
cat("Verified six native Fine-Gray fits, sandwich covariance, residuals and CIFs.\n")
