# Independent full-data Delta_n reference: survival::survfit and stats::glm.
# Fixed event/censor counts, fitted observed-case support, separately resampled
# censor records, and exact right-continuous curve integration to original tau.
options(warn=2, digits=17)
suppressPackageStartupMessages(library(survival))

time <- seq_len(20)
event <- c(1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0)
arm <- rep(0:1,10)
input <- data.frame(time=time,event=event,treatment=arm)
support <- sort(unique(time[event==1]))
censor <- list(time[arm==0 & event==0],time[arm==1 & event==0])
tau <- min(tapply(time,arm,max))
# Zero-based indices address pooled failure support or within-arm censor records.
f0 <- rbind(c(0,3,4,6,8,10),c(0,3,4,6,8,10),c(0,2,4,7,9,11),rep(0,6),c(0,0,2,2,4,4))
f1 <- rbind(c(1,2,5,7,9,11),c(1,2,5,7,9,11),c(1,3,5,6,8,10),rep(11,6),c(1,2,5,7,9,11))
c0 <- rbind(0:3,c(0,0,3,3),c(0,1,3,3),0:3,c(0,1,1,1))
c1 <- rbind(0:3,c(1,2,3,3),c(0,0,3,3),0:3,0:3)
tapes <- list(failure0=f0,failure1=f1,censor0=c0,censor1=c1)
rows <- list()
for (name in names(tapes)) for (i in seq_len(nrow(tapes[[name]]))) {
  rows[[length(rows)+1L]] <- data.frame(replicate=i-1L,component=name,
    position=seq_len(ncol(tapes[[name]]))-1L,index=tapes[[name]][i,])
}
write.csv(input,'tests/fixtures/proportional-density-full-inputs.csv',row.names=FALSE)
write.csv(do.call(rbind,rows),'tests/fixtures/proportional-density-full-resamples.csv',row.names=FALSE)

km <- function(t,d,at) {
  model <- survfit(Surv(t,d)~1)
  position <- findInterval(at,model$time)
  c(1,model$surv)[position+1L]
}
fit <- function(t,d,z,pooled) {
  selected <- which(d==1)
  selected <- selected[order(t[selected])]
  y <- t[selected]
  group <- z[selected]
  unique <- sort(unique(y))
  counts <- tabulate(group+1L,nbins=2)
  if (max(y[group==0])<=min(y[group==1]) || max(y[group==1])<=min(y[group==0])) {
    return(list(failure='separation'))
  }
  h <- sapply(0:1,function(j) {
    use <- if (pooled) rep(TRUE,length(t)) else z==j
    km(t[use],1-d[use],y)
  })
  if (any(h<=0)) return(list(failure='unsupported_censoring_tail'))
  offset <- log(counts[2]/counts[1])+log(h[,2]/h[,1])
  model <- glm(group~y+offset(offset),family=binomial(),
    control=glm.control(epsilon=1e-12,maxit=100))
  stopifnot(model$converged)
  eta <- predict(model)
  observed <- cbind(plogis(-eta)/counts[1],plogis(eta)/counts[2])
  disease <- observed/h
  disease <- sweep(disease,2,colSums(disease),'/')
  fitted <- vapply(unique,function(q) sum(disease[y>q,1]),numeric(1))
  base <- km(t[z==0],d[z==0],unique)
  incidence <- 1-km(t[z==0],d[z==0],max(t[z==0]))
  empirical <- pmax(0,pmin(1,(base-(1-incidence))/incidence))
  knots <- c(0,unique[unique<tau],tau)
  left <- head(knots,-1L)
  position <- findInterval(left,unique)
  difference <- c(0,fitted-empirical)[position+1L]
  statistic <- sum(diff(knots)*difference^2)
  list(failure='',statistic=statistic,alpha=unname(coef(model)[1]),
    beta=unname(coef(model)[2]),curve=data.frame(time=unique,
      fitted=fitted,nonparametric=empirical))
}
summaries <- curves <- list()
for (pooled in c(FALSE,TRUE)) for (i in 0:5) {
  if (i==0) {
    t <- time; d <- event; z <- arm
  } else {
    t <- c(support[f0[i,]+1L],support[f1[i,]+1L],censor[[1]][c0[i,]+1L],censor[[2]][c1[i,]+1L])
    d <- c(rep(1,12),rep(0,8))
    z <- c(rep(0,6),rep(1,6),rep(0,4),rep(1,4))
  }
  result <- fit(t,d,z,pooled)
  good <- result$failure==''
  summaries[[length(summaries)+1L]] <- data.frame(equal_censoring=pooled,replicate=i-1L,
    tau=tau,statistic=if(good) result$statistic else NA_real_,
    alpha=if(good) result$alpha else NA_real_,beta=if(good) result$beta else NA_real_,
    failure=result$failure)
  if (good) curves[[length(curves)+1L]] <- cbind(data.frame(equal_censoring=pooled,replicate=i-1L),result$curve)
}
write.csv(do.call(rbind,summaries),'tests/fixtures/proportional-density-full-reference.csv',row.names=FALSE,na='')
write.csv(do.call(rbind,curves),'tests/fixtures/proportional-density-full-curves.csv',row.names=FALSE)
cat('Generated independent GLM/KM curve and full-data bootstrap references.\n')
