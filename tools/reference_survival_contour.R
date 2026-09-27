# Reference fits and predictions from survival 3.6-4 and unchanged author helpers.
# Only the plotting sinks are replaced to capture numerical surfaces; no plotly
# or other survivalContour dependencies are installed.
options(warn=2, digits=17)
stopifnot(as.character(packageVersion("survival")) == "3.6.4")
library(survival)
root <- "research/raw/survivalContour/R"
stopifnot(unname(tools::md5sum(file.path(root,c("coxContour.R","cox3DContour.R")))) ==
  c("bd284e2c13a7d73382dd95c9e7bae1af","445ffb3920eef6f948fc92ee4c1b748c"))
native <- new.env(parent=globalenv())
sys.source(file.path(root,"coxContour.R"),envir=native)
sys.source(file.path(root,"cox3DContour.R"),envir=native)
native$survContour <- function(x,y,z,...) list(time=x,grid=y,survival=z)
native$contour3D <- function(x,y,z,...) list(time=x,grid=y,prediction=z)

data <- data.frame(
  time=c(1,2,2,3,3,3,4,5,6,7,8,9),
  status=c(1,1,0,1,1,0,1,0,1,1,0,1),
  x1=c(-1,.3,1.2,-.2,.8,-1.3,.4,1.5,-.7,.1,1.1,-.4),
  x2=c(0,1,0,1,0,1,1,0,1,0,1,0))
write.csv(data,"tests/fixtures/survival-contour-input.csv",row.names=FALSE)
fit_rows <- list()
surface_rows <- list()
quantile_rows <- list()
for(ties in c("efron","breslow")) {
  fit <- coxph(Surv(time,status)~x1+x2,data=data,ties=ties,
               x=TRUE,y=TRUE,model=TRUE,
               control=coxph.control(eps=1e-12,toler.chol=1e-14,iter.max=50))
  sm <- summary(fit)
  fit_rows[[length(fit_rows)+1L]] <- data.frame(ties=ties,variable=names(coef(fit)),
    coefficient=coef(fit),se=sqrt(diag(vcov(fit))),pvalue=sm$coefficients[,5],
    covariance_x1=vcov(fit)[,1],covariance_x2=vcov(fit)[,2],
    loglik_null=fit$loglik[1],loglik=fit$loglik[2])
  for(profile_name in c("mean","explicit")) {
    profile <- if(profile_name=="mean") NULL else data.frame(time=0,status=0,x1=0,x2=.25)
    surface <- native$coxContour(data,fit,"x1",nCovEval=5,otherCov=profile)
    interval <- native$cox3DContour(data,fit,"x1",nCovEval=5,otherCov=profile,CI3D=TRUE)
    stopifnot(identical(surface$time,interval$time),
      isTRUE(all.equal(surface$survival,interval$prediction$surv)))
    fixed_x2 <- if(profile_name=="mean") mean(data$x2) else .25
    for(i in seq_along(surface$grid)) {
      surface_rows[[length(surface_rows)+1L]] <- data.frame(
        case=paste(ties,profile_name,sep="_"),ties=ties,profile=profile_name,
        row=i,x1=surface$grid[i],x2=fixed_x2,time=surface$time,
        survival=surface$survival[i,],lower=interval$prediction$lower[i,],
        upper=interval$prediction$upper[i,],
        cumulative_hazard=interval$prediction$cumhaz[,i],
        se_log_survival=interval$prediction$std.err[,i])
    }
    # These five empirical probabilities are explicit Python summary choices,
    # not an assertion that the unavailable app backend uses the same quintet.
    probs <- c(.1,.25,.5,.75,.9)
    new_data <- data.frame(x1=as.numeric(quantile(data$x1,probs)),x2=fixed_x2)
    curves <- survfit(fit,newdata=new_data,conf.type="log",conf.int=.95)
    for(i in seq_along(probs)) {
      quantile_rows[[length(quantile_rows)+1L]] <- data.frame(
        case=paste(ties,profile_name,sep="_"),probability=probs[i],
        x1=new_data$x1[i],x2=fixed_x2,time=curves$time,
        survival=curves$surv[,i],lower=curves$lower[,i],upper=curves$upper[,i])
    }
  }
}
write.csv(do.call(rbind,fit_rows),"tests/fixtures/survival-contour-fits.csv",row.names=FALSE)
write.csv(do.call(rbind,surface_rows),"tests/fixtures/survival-contour-surfaces.csv",row.names=FALSE)
write.csv(do.call(rbind,quantile_rows),"tests/fixtures/survival-contour-quantiles.csv",row.names=FALSE)
cat("Verified original Cox contour helpers for Efron/Breslow and mean/explicit profiles.\n")
