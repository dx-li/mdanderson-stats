# Stratified Cox references from survival 3.6-4, plus an executable audit of
# the unchanged author's stratum-selection defect. No plotly installation.
options(warn=2, digits=17)
stopifnot(as.character(packageVersion("survival")) == "3.6.4")
library(survival)
source_root <- "research/raw/survivalContour/R"
stopifnot(unname(tools::md5sum(file.path(source_root,
  c("coxStrataContour.R", "coxStrata3DContour.R")))) ==
  c("0581156e8c83469a34388a18822d2226", "2ede682ff157b0fad22f78e5d55d4ded"))
native <- new.env(parent=globalenv())
sys.source(file.path(source_root,"coxStrataContour.R"), envir=native)
sys.source(file.path(source_root,"coxStrata3DContour.R"), envir=native)
native$contourPart <- function(x,y,z,...) list(time=x,grid=y,survival=z)
native$`%>%` <- function(x,...) x  # Ignore the hide_colorbar styling call.
native$subplot <- function(x,...) x
native$contour3D <- function(x,y,z,...) list(grid=y,prediction=z)

a <- read.csv("tests/fixtures/survival-contour-input.csv")
a$group <- "A"
b <- data.frame(time=c(0,2,2,4,4,4,5,6,8,9,10,12),
  status=c(1,0,1,1,0,1,0,1,1,0,1,1),
  x1=c(.9,-.1,-1,.7,-.4,1.1,0,-.8,.2,1.3,-1.2,.5),
  x2=c(1,0,1,0,0,1,1,0,1,0,1,0), group="B")
c <- data.frame(time=c(1,5,7,8),status=0,x1=c(.1,.4,-.6,1.5),
  x2=c(0,1,0,1),group="C")
data <- rbind(a,b,c)
data$group <- factor(data$group, levels=c("A","B","C"))
write.csv(data,"tests/fixtures/survival-stratified-input.csv",row.names=FALSE)
grid <- seq(quantile(data$x1,.025),quantile(data$x1,.975),length.out=5)
probs <- c(.1,.25,.5,.75,.9)
qgrid <- as.numeric(quantile(data$x1,probs))
fit_rows <- list()
surface_rows <- list()
quantile_rows <- list()
native_rows <- list()
for(ties in c("efron","breslow")) {
  fit <- coxph(Surv(time,status)~x1+x2+strata(group),data=data,ties=ties,
    x=TRUE,y=TRUE,model=TRUE,
    control=coxph.control(eps=1e-12,toler.chol=1e-14,iter.max=50))
  sm <- summary(fit)
  fit_rows[[length(fit_rows)+1L]] <- data.frame(ties=ties,variable=names(coef(fit)),
    coefficient=coef(fit),se=sqrt(diag(vcov(fit))),pvalue=sm$coefficients[,5],
    covariance_x1=vcov(fit)[,1],covariance_x2=vcov(fit)[,2],
    loglik_null=fit$loglik[1],loglik=fit$loglik[2])
  native_2d <- native$coxStrataContour(data,fit,"x1",nCovEval=5,drawHistogram=FALSE)
  native_3d <- native$coxStrata3DContour(data,fit,"x1",nCovEval=5,CI3D=TRUE)
  stopifnot(identical(native_2d[[1]],native_2d[[2]]),
    identical(native_2d[[1]],native_2d[[3]]),
    identical(native_3d$prediction$surv[[1]],native_3d$prediction$surv[[2]]))
  native_rows[[length(native_rows)+1L]] <- data.frame(ties=ties,
    native_xlevel_name=names(fit$xlevels)[1],
    all_native_2d_surfaces_equal=TRUE,all_native_3d_surfaces_equal=TRUE,
    native_time_count=length(native_2d[[1]]$time),
    native_3d_time_count=length(native_3d$prediction$time[[1]]),
    native_3d_surface_columns=ncol(native_3d$prediction$surv[[1]]))
  for(profile_name in c("mean","explicit")) {
    fixed_x2 <- if(profile_name=="mean") mean(data$x2) else .25
    for(group in levels(data$group)) {
      for(time_kind in c("native","custom")) {
        times <- if(time_kind=="native") sort(unique(c(0,data$time[data$group==group]))) else
          c(0,1,2.5,4,13)
        for(kind in c("surface","quantile")) {
          values <- if(kind=="surface") grid else qgrid
          for(i in seq_along(values)) {
            new_data <- data.frame(x1=values[i],x2=fixed_x2,
              group=factor(group,levels=levels(data$group)))
            prediction <- summary(survfit(fit,newdata=new_data,
              conf.type="log",conf.int=.95), times=times, extend=TRUE)
            # survfit stores SE(log S); summary converts it to SE(S).
            row <- data.frame(case=paste(ties,profile_name,sep="_"),ties=ties,
              profile=profile_name,group=group,time_kind=time_kind,row=i,
              probability=if(kind=="quantile") probs[i] else NA_real_,
              x1=values[i],x2=fixed_x2,time=prediction$time,
              survival=prediction$surv,lower=prediction$lower,upper=prediction$upper,
              cumulative_hazard=prediction$cumhaz,
              se_log_survival=prediction$std.err/prediction$surv)
            if(kind=="surface") surface_rows[[length(surface_rows)+1L]] <- row else
              quantile_rows[[length(quantile_rows)+1L]] <- row
          }
        }
      }
    }
  }
}
for(name in c("fit","surface","quantile","native")) {
  write.csv(do.call(rbind,get(paste0(name,"_rows"))),
    paste0("tests/fixtures/survival-stratified-",name,".csv"),row.names=FALSE)
}
cat("Verified direct stratified Cox references and reproduced native helper defects.\n")
