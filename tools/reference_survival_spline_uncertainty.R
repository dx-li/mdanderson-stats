# Independent base-R spline-survival summaries using original-coordinate draws.
# Reuses the previously verified native-kernel fit/covariance reference inputs.
# The raw truncated-power basis is independent of Python's stable tail branches.
options(warn=2, digits=17)
metric <- read.csv("tests/fixtures/survival-spline-metric.csv")
knots_table <- read.csv("tests/fixtures/survival-spline-knots.csv")
profiles <- rbind(c(-.7,.2), c(.4,-.3), c(1.2,.7))
times <- c(0,.03,.2,.7,1,2,4,10,100,Inf)
set.seed(16628)
draw_rows <- summaries <- list()
for (case in unique(metric$case)) {
  dat <- metric[metric$case==case,]
  mu <- dat$value[dat$metric=="parameters"]
  nc <- length(mu)
  covariance <- matrix(0,nc,nc)
  cr <- dat[dat$metric=="covariance",]
  stopifnot(nrow(cr)==nc*nc)
  covariance[cbind(cr$row,cr$column)] <- cr$value
  knots <- knots_table$value[knots_table$case==case]
  ng <- length(knots)
  scale <- strsplit(case,"_",fixed=TRUE)[[1]][1]
  theta <- sweep(matrix(rnorm(64*nc),64,nc)%*%chol(covariance),2,mu,"+")
  theta[1,] <- mu
  # Deliberately rising finite-time curve: the native MC summary does not
  # filter it. Its forced endpoints still equal S(0)=1 and S(Inf)=0.
  theta[2,2] <- -abs(mu[2])
  if (ng>2) theta[2,3:ng] <- 0
  for (i in seq_len(nrow(theta))) for (j in seq_len(nc)) {
    draw_rows[[length(draw_rows)+1L]] <- data.frame(
      case=case,draw=i,parameter=j,value=theta[i,j])
  }
  for (profile in seq_len(nrow(profiles))) for (time in times) {
    if(time==0) values <- rep(1,nrow(theta))
    else if(is.infinite(time)) values <- rep(0,nrow(theta))
    else {
      z <- log(time)
      basis <- c(1,z)
      if(ng>2) for(j in 2:(ng-1)) {
        weight <- (knots[ng]-knots[j])/(knots[ng]-knots[1])
        basis <- c(basis, max(z-knots[j],0)^3-
                   weight*max(z-knots[1],0)^3-
                   (1-weight)*max(z-knots[ng],0)^3)
      }
      eta <- drop(theta[,seq_len(ng),drop=FALSE]%*%basis +
                  theta[,(ng+1):nc,drop=FALSE]%*%profiles[profile,])
      values <- switch(scale,hazard=exp(-exp(eta)),odds=plogis(-eta),normal=pnorm(-eta))
    }
    stopifnot(all(is.finite(values)),all(values>=0 & values<=1))
    limits <- quantile(values,c(.025,.975),names=FALSE,type=7,na.rm=TRUE)
    summaries[[length(summaries)+1L]] <- data.frame(
      case=case,profile=profile,time=time,draws=length(values),
      lower=limits[1],upper=limits[2],sd=sd(values))
  }
}
write.csv(do.call(rbind,draw_rows),"tests/fixtures/survival-spline-mc-draws.csv",row.names=FALSE)
write.csv(do.call(rbind,summaries),"tests/fixtures/survival-spline-mc-summary.csv",row.names=FALSE)
cat("Wrote",length(draw_rows),"parameter cells and",length(summaries),"summary rows.\n")
