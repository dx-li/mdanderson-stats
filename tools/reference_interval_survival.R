# Native icenReg 2.0.16 semiparametric PH optimizer and support references.
source("tools/reference_interval_survival_core.R")

id <- 1:96
x <- cbind(x1=cos(id*.71)+id/200,x2=sin(id*1.31)-id/250)
u <- ((id*37) %% 97 + .5)/97
latent <- 3*(-log(u)/exp(.45*x[,1]-.3*x[,2]))^(1/1.4)
lower <- floor(latent); upper <- lower+1
exact <- id%%7==0 & latent<6
lower[exact] <- upper[exact] <- latent[exact]
right <- latent>=6
lower[right] <- 6; upper[right] <- Inf
cases <- list(
  mixed=list(lower=lower,upper=upper,x=x,w=rep(1,96)),
  mixed_weighted=list(lower=lower,upper=upper,x=x,w=1+id%%3),
  turnbull=list(lower=lower,upper=upper,x=matrix(numeric(),96,0),w=rep(1,96)))
inspection <- .5+(id%%12)/2
cases$current_status <- list(lower=ifelse(latent<=inspection,0,inspection),
  upper=ifelse(latent<=inspection,inspection,Inf),x=x,w=rep(1,96))
time <- pmin(ceiling(latent),6)
cases$exact_right <- list(lower=time,upper=ifelse(latent<6,time,Inf),x=x,w=rep(1,96))
inputs <- supports <- metrics <- curves <- list()
for (name in names(cases)) {
  case <- cases[[name]]
  result <- native_fit(case$lower,case$upper,case$x,case$w)
  k <- length(result$fit$mass)
  inputs[[name]] <- data.frame(case=name,row=seq_along(case$lower),
    lower=case$lower,upper=case$upper,weight=case$w,
    x1=if(ncol(case$x)>0)case$x[,1] else 0,
    x2=if(ncol(case$x)>0)case$x[,2] else 0,covariates=ncol(case$x),
    left_index=result$support$l_inds,right_index=result$support$r_inds)
  supports[[name]] <- data.frame(case=name,index=seq_len(k),
    lower=result$support$lower,upper=result$support$upper,
    original_lower=result$original_lower,left_closed=result$left_closed,mass=result$fit$mass)
  values <- c(log_likelihood=result$fit$log_likelihood,iterations=result$fit$iterations,
              if(ncol(case$x))setNames(result$fit$coefficients,paste0("beta",seq_len(ncol(case$x)))),
              if(ncol(case$x))setNames(result$center,paste0("center",seq_len(ncol(case$x)))))
  metrics[[name]] <- data.frame(case=name,metric=names(values),value=unname(values))
  profiles <- rbind(c(0,0),c(-1,.5),c(1,-.5))
  for(j in seq_len(nrow(profiles))) {
    eta <- if(ncol(case$x))sum((profiles[j,]-result$center)*result$fit$coefficients) else 0
    # Native getSCurves uses 1-cumsum; clamp rounding at its fixed endpoints.
    base <- pmax(0,pmin(1,1-c(0,cumsum(result$fit$mass))))
    survival <- reference_curves(base,eta)
    stopifnot(max(abs(survival-base^exp(eta)))<1e-14)
    curves[[length(curves)+1L]] <- data.frame(case=name,profile=j,index=0:k,
      x1=profiles[j,1],x2=profiles[j,2],survival=survival)
  }
  cat(name,"iterations",result$fit$iterations,"logLik",result$fit$log_likelihood,"\n")
}
for (kind in c("inputs","supports","metrics","curves")) {
  write.csv(do.call(rbind,get(kind)),paste0("tests/fixtures/interval-survival-",kind,".csv"),
            row.names=FALSE)
}
