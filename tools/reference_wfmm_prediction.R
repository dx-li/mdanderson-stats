# Independent finite-mixture predictive moments under the coefficient model.
# Cov(Ynew) = average conditional covariance + covariance of conditional means.
options(warn = 2, digits = 17)
x <- matrix(c(1,-1, 1,.5, 1,1),3,2,byrow=TRUE)
existing <- matrix(c(1,0, 1,0, 0,1),3,2,byrow=TRUE)
new <- matrix(c(1,0, 1,0, .5,1),3,2,byrow=TRUE)
new_group <- c(2L,1L)
residual_group <- c(1L,2L,1L)
synthesis <- matrix(c(1,.5, -.25,2),2,2,byrow=TRUE)
draws <- 8L
b <- u <- q <- s <- array(0,c(draws,2,2))
input <- list()
for (h in seq_len(draws)) {
  b[h,,] <- matrix(c(.2+.08*h,-.3+.02*h,.4-.03*h,.1+.05*h),2,2,byrow=TRUE)
  u[h,,] <- matrix(c(-.2+.04*h,.1-.01*h,.3-.02*h,-.15+.03*h),2,2,byrow=TRUE)
  q[h,,] <- matrix(c(.3+.02*h,.1+.01*h,.7-.02*h,.4+.03*h),2,2,byrow=TRUE)
  s[h,,] <- matrix(c(.2+.01*h,.5-.02*h,.6-.01*h,.15+.02*h),2,2,byrow=TRUE)
  for (kind in c('b','u','q','s')) for (i in 1:2) for (k in 1:2) {
    input[[length(input)+1L]] <- data.frame(draw=h-1L,kind=kind,row=i-1L,
                                          coefficient=k-1L,value=get(kind)[h,i,k])
  }
}
write.csv(do.call(rbind,input),'tests/fixtures/wfmm-prediction-draws.csv',row.names=FALSE)
write.csv(data.frame(row=0:2,x1=x[,1],x2=x[,2],existing1=existing[,1],
  existing2=existing[,2],new1=new[,1],new2=new[,2],residual_group=residual_group-1L),
  'tests/fixtures/wfmm-prediction-design.csv',row.names=FALSE)
rows <- list()
conditionals <- list()
for (target in c('population','existing','new_latent','replicate')) {
  means <- matrix(0,draws,6)
  conditional_covariance <- array(0,c(draws,6,6))
  for (h in seq_len(draws)) {
    mu <- x %*% b[h,,]
    if (target!='population') mu <- mu + existing %*% u[h,,]
    curve_mu <- mu %*% synthesis
    means[h,] <- as.vector(t(curve_mu))
    for (i in 1:3) for (j in 1:3) {
      variance <- rep(0,2)
      if (target %in% c('new_latent','replicate')) {
        for (level in 1:2) {
          variance <- variance + new[i,level]*new[j,level]*q[h,new_group[level],]
        }
      }
      if (target=='replicate' && i==j) variance <- variance+s[h,residual_group[i],]
      conditional_covariance[h,(2*i-1):(2*i),(2*j-1):(2*j)] <-
        t(synthesis) %*% diag(variance) %*% synthesis
    }
    for (i in 1:3) for (k in 1:2) {
      conditionals[[length(conditionals)+1L]] <- data.frame(target=target,draw=h-1L,
        row=i-1L,time=k-1L,value=curve_mu[i,k])
    }
  }
  mean <- colMeans(means)
  centered <- sweep(means,2,mean)
  covariance <- crossprod(centered)/draws + apply(conditional_covariance,c(2,3),mean)
  stopifnot(max(abs(covariance-t(covariance)))<1e-14,
            min(eigen(covariance,symmetric=TRUE,only.values=TRUE)$values)>-1e-13)
  for (i in 1:6) {
    rows[[length(rows)+1L]] <- data.frame(target=target,metric='mean',i=i-1L,j=-1L,value=mean[i])
    for (j in 1:6) rows[[length(rows)+1L]] <- data.frame(target=target,metric='covariance',i=i-1L,j=j-1L,value=covariance[i,j])
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/wfmm-prediction-moments.csv',row.names=FALSE)
write.csv(do.call(rbind,conditionals),'tests/fixtures/wfmm-prediction-conditional.csv',row.names=FALSE)
cat('Generated 168 analytic predictive mixture moments and 192 conditional means.\n')
