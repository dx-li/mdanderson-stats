# Native flexsurv 2.3.2 spline kernels with an independent base-R fit harness.
# Does not execute or claim parity with flexsurvspline's full formula/fit stack.
options(warn=2, digits=17)
stopifnot(as.character(packageVersion("Rcpp"))=="1.1.1")
source_root <- "research/raw/flexsurv"
hashes <- c(
  "src/splines.cpp"="8e0b8dd1dfebeb6f780e555a0befcc964aa81e87",
  "R/spline.R"="b46b2060dbc40da6d1390e725cbc3c753a9120c9",
  "R/deriv.R"="b5475ad42cd2793c02d10ec6c2f80831a465fea7",
  "R/deriv2.R"="0fdcc37c0213f8f4c66969b4de5a683694d4c8ad")
for(name in names(hashes)) {
  actual <- system2("git",c("hash-object",shQuote(file.path(source_root,name))),stdout=TRUE)
  stopifnot(identical(unname(actual),unname(hashes[name])))
}
dir.create("research/raw/spline-rcpp-cache",showWarnings=FALSE)
Rcpp::sourceCpp(file.path(source_root,"src/splines.cpp"),
               cacheDir="research/raw/spline-rcpp-cache",showOutput=FALSE)
source(file.path(source_root,"R/spline.R"))
source(file.path(source_root,"R/deriv.R"))
source(file.path(source_root,"R/deriv2.R"))

# Retain direct log-domain formulas when the native wrapper underflows through
# exponentiation before taking a log. Its ordinary probability is also retained.
logsf_direct <- function(eta,scale) {
  if(scale=="hazard") -exp(eta)
  else if(scale=="odds") -pmax(eta,0)-log1p(exp(-abs(eta)))
  else pnorm(-eta,log.p=TRUE)
}
# The derivative is quadratic on each knot interval and constant in either
# tail. Endpoints plus any interior quadratic vertex cover its global minimum.
minimum_slope <- function(gamma, knots) {
  minimum <- min(drop(dbasis(knots,knots)%*%gamma))
  for(j in seq_len(length(knots)-1L)) {
    points <- c(knots[j],mean(knots[j:(j+1L)]),knots[j+1L])
    values <- drop(dbasis(knots,points)%*%gamma)
    a <- 2*(values[3]-2*values[2]+values[1])
    b <- values[3]-values[1]-a
    if(a>0) {
      vertex <- -b/(2*a)
      if(vertex>0 && vertex<1) minimum <- min(minimum,a*vertex^2+b*vertex+values[1])
    }
  }
  minimum
}
distribution <- bases <- list()
z <- c(-8,-2,-1.5,-.1,0,1.5,2,8)
for(k in c(0,1,4)) {
  knots <- seq(-2,2,length.out=k+2)
  gamma <- c(-.4,1.3,if(k>0).02*(-1)^seq_len(k))
  B <- basis(knots,z); D <- dbasis(knots,z)
  for(i in seq_along(z)) for(j in seq_along(knots)) {
    bases[[length(bases)+1L]] <- data.frame(k=k,z=z[i],column=j,
      knot=knots[j],gamma=gamma[j],basis=B[i,j],derivative=D[i,j])
  }
  eta <- drop(B%*%gamma); slope <- drop(D%*%gamma)
  stopifnot(all(slope>0))
  for(scale in c("hazard","odds","normal")) {
    distribution[[length(distribution)+1L]] <- data.frame(
      k=k,scale=scale,time=exp(z),eta=eta,slope=slope,
      log_density=log(slope)-z+ldlink(scale)(eta),
      log_survival=logsf_direct(eta,scale),
      native_density=dsurvspline(exp(z),gamma=gamma,knots=knots,scale=scale),
      native_survival=psurvspline(exp(z),gamma=gamma,knots=knots,scale=scale,lower.tail=FALSE))
  }
}
write.csv(do.call(rbind,bases),"tests/fixtures/survival-spline-basis.csv",row.names=FALSE)
write.csv(do.call(rbind,distribution),"tests/fixtures/survival-spline-distribution.csv",row.names=FALSE)

data <- read.csv("tests/fixtures/generalized-gamma-input.csv")
data <- data[data$case=="positive_covariates",c("time","event","x1","x2")]
write.csv(data,"tests/fixtures/survival-spline-input.csv",row.names=FALSE)
X <- as.matrix(data[,c("x1","x2")])
metric_rows <- predictions <- knots_rows <- list()
for(scale in c("hazard","odds","normal")) for(k in c(0,4)) {
  case <- paste(scale,k,sep="_")
  knots <- quantile(log(data$time[data$event==1]),seq(0,1,length.out=k+2))
  m <- length(knots)
  B <- basis(knots,log(data$time)); D <- dbasis(knots,log(data$time))
  fn <- function(theta) {
    gamma <- matrix(theta[seq_len(m)],nrow=nrow(data),ncol=m,byrow=TRUE)
    gamma[,1] <- gamma[,1]+drop(X%*%tail(theta,2))
    ld <- dsurvspline(data$time,gamma=gamma,knots=knots,scale=scale,log=TRUE)
    ls <- psurvspline(data$time,gamma=gamma,knots=knots,scale=scale,lower.tail=FALSE,log.p=TRUE)
    values <- ifelse(data$event==1,ld,ls)
    if(!all(is.finite(values))) return(1e100)
    -sum(values)
  }
  log_time_sd <- sd(log(data$time))
  initial <- c(-mean(log(data$time))/log_time_sd,1/log_time_sd,rep(0,k+2))
  first <- optim(initial,fn,method="BFGS",control=list(reltol=1e-11,maxit=700,ndeps=rep(1e-5,m+2)))
  fit <- optim(first$par,fn,method="BFGS",control=list(reltol=1e-13,maxit=700,ndeps=rep(1e-5,m+2)))
  stopifnot(fit$convergence==0,is.finite(fit$value),fit$value<1e99)
  min_slope <- minimum_slope(fit$par[seq_len(m)],knots)
  stopifnot(min_slope>0)
  score <- vapply(seq_along(fit$par),function(j) {
    h <- 1e-4*max(1,abs(fit$par[j]))
    values <- vapply(c(-2,-1,1,2),function(offset) {
      shifted <- fit$par; shifted[j] <- shifted[j]+offset*h; fn(shifted)
    },numeric(1))
    (values[1]-8*values[2]+8*values[3]-values[4])/(12*h)
  },numeric(1))
  stopifnot(max(abs(score))<1e-3)
  info <- optimHess(fit$par,fn,control=list(ndeps=rep(5e-5,m+2)))
  stopifnot(all(eigen(info,symmetric=TRUE)$values>0))
  covariance <- solve(info)
  for(metric in c("parameters","information","covariance","log_likelihood")) {
    value <- switch(metric,parameters=fit$par,information=info,covariance=covariance,log_likelihood=-fit$value)
    a <- as.matrix(value)
    for(i in seq_len(nrow(a))) for(j in seq_len(ncol(a))) {
      metric_rows[[length(metric_rows)+1L]] <- data.frame(case=case,metric=metric,row=i,column=j,value=a[i,j])
    }
  }
  knots_rows[[length(knots_rows)+1L]] <- data.frame(case=case,index=seq_along(knots),value=knots)
  profiles <- rbind(c(-.7,.2),c(.4,-.3),c(1.2,.7))
  times <- c(.001,.1,.5,1,2,4,8,100,1e6)
  for(row in 1:3) {
    gamma <- fit$par[seq_len(m)]
    gamma[1] <- gamma[1]+sum(profiles[row,]*tail(fit$par,2))
    eta <- drop(basis(knots,log(times))%*%gamma)
    slope <- drop(dbasis(knots,log(times))%*%gamma)
    predictions[[length(predictions)+1L]] <- data.frame(case=case,row=row,
      x1=profiles[row,1],x2=profiles[row,2],time=times,eta=eta,slope=slope,
      log_survival=logsf_direct(eta,scale),
      native_survival=psurvspline(times,gamma=gamma,knots=knots,scale=scale,lower.tail=FALSE))
  }
  cat(case,"logLik =",format(-fit$value,digits=10),
      "maximum score =",max(abs(score)),"minimum slope =",min_slope,"\n")
}
write.csv(do.call(rbind,knots_rows),"tests/fixtures/survival-spline-knots.csv",row.names=FALSE)
write.csv(do.call(rbind,metric_rows),"tests/fixtures/survival-spline-metric.csv",row.names=FALSE)
write.csv(do.call(rbind,predictions),"tests/fixtures/survival-spline-prediction.csv",row.names=FALSE)
cat("Generated spline basis, distribution and six native-kernel fit references.\n")
