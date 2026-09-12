# Native numerical references, using an inspected local PoPdesign 1.1.0 source
# archive (no package installation). Vendor source remains in ignored research/raw.
root <- 'research/raw/PoPdesign/source-1.1.0/PoPdesign/R'
for (name in c('prbf01.R','bound.R','get.boundary.pop.R','select.mtd.pop.R')) {
  source(file.path(root,name))
}
rows <- list()
for (phi in c(.05,.15,.25,.3,.4,.6)) {
  b <- get.boundary.pop(phi,60,3)$out.full.boundary
  rows[[length(rows)+1]] <- data.frame(target=phi,patients=b[1,],
    escalate=b[2,],deescalate=b[3,],exclude_under=b[4,],exclude_over=b[5,])
}
write.csv(do.call(rbind,rows),'tests/fixtures/pop-boundaries.csv',row.names=FALSE)

# The native selector has an untried-dose indexing defect. Compare valid native
# cases here; Python separately checks dose-label mapping for untried doses.
set.seed(175)
rows <- list()
for (i in 1:30) {
  n <- sample(c(3,6,9,12,18),5,replace=TRUE)
  y <- rbinom(5,n,sort(runif(5,0,.7)))
  phi <- c(.15,.25,.3,.4)[1+i%%4]
  fit <- select.mtd.pop(phi,n,y)
  if(is.list(fit)) {
    chosen <- fit$MTD; estimates <- paste(format(fit$p_est,digits=17),collapse=';')
  } else {
    stopifnot(fit==99);chosen <- 0;estimates <- ''
  }
  rows[[i]] <- data.frame(target=phi,patients=paste(n,collapse=';'),
    toxicities=paste(y,collapse=';'),selected=chosen,estimates=estimates)
}
write.csv(do.call(rbind,rows),'tests/fixtures/pop-selection.csv',row.names=FALSE)

# Scalar predictive Bayes factors, including all/zero events.
rows <- list()
for(phi in c(.05,.25,.6))for(n in c(1,3,10,30,100))for(y in unique(c(0,floor(n*phi),n))) {
  rows[[length(rows)+1]] <- data.frame(target=phi,patients=n,toxicities=y,
    log_bayes_factor=log(prbf01(n,y,phi)))
}
write.csv(do.call(rbind,rows),'tests/fixtures/pop-factor.csv',row.names=FALSE)

# Exact enumeration of cohort DLT counts for two-dose, nine-patient trials.
# Native PrBF and MTD kernels provide the numerical oracle. Compacting input
# to the MTD function and mapping labels back avoids its untried-dose bug.
exact <- function(p) {
  total <- rep(0,10) # selections 0..2; mean n1,n2,y1,y2; early; under,over
  terminal <- function(n,y,weight,early) {
    labels <- which(n>0)
    fit <- select.mtd.pop(.3,n[labels],y[labels])
    dose <- if(is.list(fit))labels[fit$MTD] else 0
    values <- c(as.integer(0:2==dose),n,y,early,
      if(which.min(abs(p-.3))==2)as.integer(n[1]>.5*9) else 0,
      if(which.min(abs(p-.3))==1)as.integer(n[2]>.5*9) else 0)
    total <<- total + weight*values
  }
  visit <- function(n,y,d,excluded,left,weight) {
    size <- min(3,left)
    for(k in 0:size) {
      w <- weight*dbinom(k,size,p[d]);if(w==0)next
      nn <- n;yy <- y;bad <- excluded
      nn[d] <- nn[d]+size;yy[d] <- yy[d]+k
      bf <- prbf01(nn[d],yy[d],.3)
      direction <- if(yy[d]/nn[d]<.3)1 else -1
      if(bf<5/24)bad[if(direction==1)1:d else d:2] <- TRUE
      early <- all(bad)
      if(left==size || early) {
        terminal(nn,yy,w,early)
      } else {
        nextdose <- d
        if(bf<2.5) {
          candidate <- max(1,min(2,d+direction))
          if(!bad[candidate])nextdose <- candidate
        }
        visit(nn,yy,nextdose,bad,left-size,w)
      }
    }
  }
  visit(c(0,0),c(0,0),1,c(FALSE,FALSE),9,1)
  stopifnot(abs(sum(total[1:3])-1)<1e-12)
  data.frame(p1=p[1],p2=p[2],no_mtd=total[1],dose1=total[2],dose2=total[3],
    mean_n1=total[4],mean_n2=total[5],mean_y1=total[6],mean_y2=total[7],
    early=total[8],risk_under=total[9],risk_over=total[10])
}
write.csv(do.call(rbind,lapply(list(c(.1,.3),c(.3,.5),c(0,0),c(1,1)),exact)),
          'tests/fixtures/pop-exact-oc.csv',row.names=FALSE)
