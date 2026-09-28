# Independent native survival 3.6-4 references for three exact AFT models.
# Python parameters: intercept, numeric slopes, log(survreg scale).
# Pointwise bounds below use an explicit delta method, not flexsurv's
# Monte-Carlo parameter-draw intervals. No extra R packages are installed.
options(warn=2, digits=17)
library(survival)
stopifnot(as.character(packageVersion("survival")) == "3.6.4")
id <- 1:48
x1 <- round(cos(id * .83) + id / 100, 6)
x2 <- round(sin(id * 1.37) - id / 150, 6)
latent <- exp(1.1 + .4*x1 - .3*x2 + .65*sin(id*2.17) + .2*cos(id*.21))
censor <- exp(1.5 + .45*cos(id*1.61))
data <- data.frame(time=round(pmin(latent,censor)*20)/20,
                   event=as.integer(latent <= censor), x1=x1, x2=x2)
write.csv(data, "tests/fixtures/parametric-survival-input.csv", row.names=FALSE)
metrics <- predictions <- list()
log_survival <- function(t, mu, sigma, distribution) {
  if(distribution=="weibull") pweibull(t,shape=1/sigma,scale=exp(mu),lower.tail=FALSE,log.p=TRUE)
  else if(distribution=="lognormal") plnorm(t,meanlog=mu,sdlog=sigma,lower.tail=FALSE,log.p=TRUE)
  else plogis(log(t),location=mu,scale=sigma,lower.tail=FALSE,log.p=TRUE)
}
for(distribution in c("weibull","lognormal","loglogistic")) {
  for(design in c("covariates","intercept")) {
    form <- if(design=="covariates") Surv(time,event)~x1+x2 else Surv(time,event)~1
    fit <- survreg(form,data=data,dist=distribution,
                   control=survreg.control(maxiter=100,rel.tolerance=1e-12))
    theta <- c(coef(fit), log(fit$scale))
    stopifnot(all(is.finite(theta)), all(eigen(fit$var,symmetric=TRUE)$values > 0))
    case <- paste(distribution,design,sep="_")
    values <- list(parameters=theta,covariance=fit$var,
                   information=solve(fit$var),log_likelihood=fit$loglik[2])
    for(metric in names(values)) {
      a <- as.matrix(values[[metric]])
      for(i in seq_len(nrow(a))) for(j in seq_len(ncol(a))) {
        metrics[[length(metrics)+1L]] <- data.frame(case=case,metric=metric,
          row=i,column=j,value=a[i,j])
      }
    }
    grid <- seq(quantile(data$x1,.025),quantile(data$x1,.975),length.out=5)
    for(profile in c("mean","explicit")) {
      xx2 <- if(profile=="mean") mean(data$x2) else .25
      for(row in seq_along(grid)) {
        v <- if(design=="covariates") c(1,grid[row],xx2) else 1
        eta <- sum(v*coef(fit))
        for(time_kind in c("native","custom")) {
          times <- if(time_kind=="native") sort(unique(c(0,data$time[data$event==1]))) else
            c(0,.01,.25,.75,1,2.5,5,10,50,1000)
          for(time in times) {
            ls <- log_survival(time,eta,fit$scale,distribution)
            lower <- upper <- exp(ls)
            se <- 0
            if(time>0) {
              # Numerical derivatives of R log cumulative hazard, using the
              # native full joint coefficient/log-scale covariance.
              fn <- function(th) log(-log_survival(time,sum(v*head(th,-1)),exp(tail(th,1)),distribution))
              step <- 1e-4
              grad <- sapply(seq_along(theta),function(k) {
                unit <- rep(0,length(theta)); unit[k] <- step
                (fn(theta-2*unit)-8*fn(theta-unit)+8*fn(theta+unit)-fn(theta+2*unit))/(12*step)
              })
              se <- sqrt(as.numeric(t(grad) %*% fit$var %*% grad))
              lower <- exp(-exp(log(-ls)+qnorm(.975)*se))
              upper <- exp(-exp(log(-ls)-qnorm(.975)*se))
            }
            predictions[[length(predictions)+1L]] <- data.frame(
              case=case,profile=profile,row=row,x1=grid[row],x2=xx2,time_kind=time_kind,
              time=time,log_survival=ls,survival=exp(ls),cumulative_hazard=-ls,
              log_cumulative_hazard_se=se,lower=lower,upper=upper)
          }
        }
      }
    }
  }
}
write.csv(do.call(rbind,metrics), "tests/fixtures/parametric-survival-metric.csv", row.names=FALSE)
write.csv(do.call(rbind,predictions), "tests/fixtures/parametric-survival-prediction.csv", row.names=FALSE)
cat("Verified six native survreg fits; wrote joint covariance and prediction references.\n")
