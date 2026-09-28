# Independent generalized-gamma references from pinned flexsurv 2.3.2 kernels.
# Uses installed Rcpp only; does not install or claim to run flexsurvreg.
options(warn=2, digits=17)
stopifnot(as.character(packageVersion("Rcpp")) == "1.1.1")
native <- "research/raw/flexsurv/src"
expected <- c(
  "gengamma.cpp"="34c799553ec1f74aa587854148bcf95834f607df",
  "gengamma.h"="e363f768ccf60786998b12972b5f80d44b036e69",
  "distribution.h"="af33d99708d25ecf19985e8be120249116658fd5",
  "mapply.h"="00c3272fa7c4e506a17ef97cee9215b313f2c7c5",
  "mapply_4.h"="0ee12ad8d1f7457c581655afc24aaae61effda9b",
  "mapply_5.h"="d0887e22dda5e32a359756d64d9dcf616b6213a8",
  "rep_len.h"="a41f896927d8d906e5c7e2d89c0c35f267f83382")
for (name in names(expected)) {
  actual <- system2("git", c("hash-object", shQuote(file.path(native, name))), stdout=TRUE)
  stopifnot(identical(unname(actual), unname(expected[name])))
}
dir.create("research/raw/flexsurv-rcpp-cache", showWarnings=FALSE)
Rcpp::sourceCpp(file.path(native, "gengamma.cpp"),
               cacheDir="research/raw/flexsurv-rcpp-cache", showOutput=FALSE)

distribution_rows <- list()
for (Q in c(-2,-1,-.5,-.1,-.01,-.001,0,.001,.01,.1,.5,1,2)) {
  z <- c(-20,-8,-3,-1,0,.2,1,3,8,20)
  mu <- .4; sigma <- .7
  times <- exp(mu + sigma*z)
  distribution_rows[[length(distribution_rows)+1L]] <- data.frame(
    Q=Q, mu=mu, sigma=sigma, time=times, z=z,
    log_density=dgengamma_work(times,mu,sigma,Q,TRUE),
    log_cdf=pgengamma_work(times,mu,sigma,Q,TRUE,TRUE),
    log_survival=pgengamma_work(times,mu,sigma,Q,FALSE,TRUE))
}
write.csv(do.call(rbind, distribution_rows),
          "tests/fixtures/generalized-gamma-distribution.csv", row.names=FALSE)

input_rows <- metric_rows <- prediction_rows <- list()
append_metrics <- function(case, values) {
  for (metric in names(values)) {
    a <- as.matrix(values[[metric]])
    for (i in seq_len(nrow(a))) for (j in seq_len(ncol(a))) {
      metric_rows[[length(metric_rows)+1L]] <<- data.frame(
        case=case,metric=metric,row=i,column=j,value=a[i,j])
    }
  }
}
for (sign in c(1,-1)) {
  id <- 1:96
  x1 <- cos(id*.71) + id/200
  x2 <- sin(id*1.31) - id/250
  truthQ <- sign*.8
  probability <- ((id*37) %% 97 + .5)/97
  gamma <- qgamma(if(sign>0) probability else 1-probability, shape=1/truthQ^2)
  z <- log(truthQ^2*gamma)/truthQ
  latent <- exp(.8 + .35*x1 - .25*x2 + .65*z)
  censor <- exp(1.35 + .4*cos(id*1.73))
  data <- data.frame(time=pmin(latent,censor),event=as.integer(latent<=censor),x1=x1,x2=x2)
  for (design in c("covariates","intercept")) {
    case <- paste(if(sign>0) "positive" else "negative",design,sep="_")
    input_rows[[length(input_rows)+1L]] <- cbind(case=case,data)
    X <- if(design=="covariates") cbind(1,x1,x2) else matrix(1,nrow=96,ncol=1)
    p <- ncol(X)
    fn <- function(theta) {
      mu <- drop(X %*% theta[seq_len(p)])
      sigma <- exp(theta[p+1]); Q <- theta[p+2]
      if (!all(is.finite(c(mu,sigma,Q))) || sigma<=0) return(1e100)
      ld <- dgengamma_work(data$time,mu,sigma,Q,TRUE)
      ls <- pgengamma_work(data$time,mu,sigma,Q,FALSE,TRUE)
      selected <- ifelse(data$event==1,ld,ls)
      if(!all(is.finite(selected))) return(1e100)
      -sum(selected)
    }
    initial <- c(.8,if(p>1)c(.35,-.25),log(.65),truthQ)
    first <- optim(initial,fn,method="BFGS",control=list(reltol=1e-12,maxit=1000,ndeps=rep(1e-4,p+2)))
    fit <- optim(first$par,fn,method="BFGS",control=list(reltol=1e-14,maxit=500,ndeps=rep(2e-5,p+2)))
    stopifnot(fit$convergence==0,sign*fit$par[p+2]>.1)
    h <- 1e-4
    score <- sapply(seq_along(fit$par),function(j) {
      step <- rep(0,length(fit$par)); step[j] <- h
      (fn(fit$par-2*step)-8*fn(fit$par-step)+8*fn(fit$par+step)-fn(fit$par+2*step))/(12*h)
    })
    stopifnot(max(abs(score))<1e-4)
    information <- optimHess(fit$par,fn,control=list(ndeps=rep(1e-4,p+2)))
    stopifnot(all(eigen(information,symmetric=TRUE)$values>0))
    covariance <- solve(information)
    append_metrics(case,list(parameters=fit$par,covariance=covariance,
                             information=information,log_likelihood=-fit$value))
    # Original Stacy coordinates: [log(scale) intercept, slopes, log(shape), log(k)].
    # This smooth change of parameters preserves the same positive-Q optimum.
    if(sign>0) {
      sigma <- exp(fit$par[p+1]); Q <- fit$par[p+2]
      original <- fit$par
      original[1] <- original[1]+2*sigma*log(Q)/Q
      original[p+1] <- log(Q)-log(sigma)
      original[p+2] <- -2*log(Q)
      J <- diag(p+2)
      J[1,p+1] <- 2*sigma*log(Q)/Q
      J[1,p+2] <- 2*sigma*(1-log(Q))/Q^2
      J[p+1,p+1] <- -1; J[p+1,p+2] <- 1/Q
      J[p+2,p+2] <- -2/Q
      vc <- J %*% covariance %*% t(J)
      append_metrics(paste0("original_",design),list(parameters=original,covariance=vc,
                                                    information=solve(vc),log_likelihood=-fit$value))
    }
    profiles <- if(p>1) rbind(c(1,-.7,.2),c(1,.4,-.3),c(1,1.2,.7)) else matrix(1,nrow=3,ncol=1)
    times <- c(0,.01,.2,.7,1,2,4,10,100,1e8)
    for(row in 1:3) {
      mu <- sum(profiles[row,]*fit$par[seq_len(p)])
      sigma <- exp(fit$par[p+1]); Q <- fit$par[p+2]
      prediction_rows[[length(prediction_rows)+1L]] <- data.frame(
        case=case,row=row,x1=if(p>1)profiles[row,2] else 0,x2=if(p>1)profiles[row,3] else 0,
        time=times,log_survival=pgengamma_work(times,mu,sigma,Q,FALSE,TRUE))
    }
    cat(case,"Q =",format(fit$par[p+2],digits=8),"logLik =",format(-fit$value,digits=10),
        "max score =",format(max(abs(score)),digits=3),"\n")
  }
}
write.csv(do.call(rbind,input_rows),"tests/fixtures/generalized-gamma-input.csv",row.names=FALSE)
write.csv(do.call(rbind,metric_rows),"tests/fixtures/generalized-gamma-metric.csv",row.names=FALSE)
write.csv(do.call(rbind,prediction_rows),"tests/fixtures/generalized-gamma-prediction.csv",row.names=FALSE)
cat("Native kernels: 130 distribution rows, four fits plus two original-coordinate transforms.\n")
