# Independent implementation of equations (1)-(4) and Appendix A, not vendor code.
# Rscript tests/fixtures/u2oet-reference.R > tests/fixtures/u2oet-reference.csv
options(digits=17)
marginal <- function(d1, d2, alpha, beta, powers, phi, interaction, linear) {
  standardize <- function(d, p) {
    d[1]/mean(d) + ((d-d[1])/(tail(d,1)-d[1]))^p * (tail(d,1)-d[1])/mean(d)
  }
  transform <- if (linear) function(d) d-1 else log
  x <- transform(standardize(d1,powers[1]))
  y <- transform(standardize(d2,powers[2]))
  out <- array(0, c(length(x),length(y),length(alpha)+1))
  for (i in seq_along(x)) for (j in seq_along(y)) {
    eta <- alpha + beta[,1]*x[i] + beta[,2]*y[j] + interaction*x[i]*y[j]
    gamma <- 1-(1+phi*exp(eta))^(-1/phi)
    survival <- c(1,cumprod(gamma))
    out[i,j,] <- survival-c(survival[-1],0)
  }
  out
}
rows <- list()
for (case in 1:5) {
  levels <- c(4,3,2,4,4)[case]
  alpha <- c(-.8,.1,-.3)[1:(levels-1)]
  beta <- matrix(c(.7,1.2,.4,.8,.3,1.1),ncol=2,byrow=TRUE)[1:(levels-1),,drop=FALSE]
  powers <- if(case==2) c(1,1) else c(.4,2)
  interaction <- c(0,.6,-.4,0,0)[case]
  e <- marginal(c(4,5,6),c(40,60,80),alpha,beta,powers,.7,interaction,case==4)
  t <- marginal(c(4,5,6),c(40,60,80),-alpha,beta[,2:1,drop=FALSE],
                rev(powers),1.8,-interaction,case==4)
  rho <- c(.6,-.8,1,-1,0)[case]
  C <- function(u,v) u*v*(1+rho*(1-u)*(1-v))
  for(i in 1:3) for(j in 1:3) {
    u <- c(0,cumsum(e[i,j,])); v <- c(0,cumsum(t[i,j,]))
    for(a in 1:levels) for(b in 1:levels) {
      prob <- C(u[a+1],v[b+1])-C(u[a],v[b+1])-C(u[a+1],v[b])+C(u[a],v[b])
      rows[[length(rows)+1]] <- c(case,i-1,j-1,a-1,b-1,prob)
    }
  }
}
answer <- as.data.frame(do.call(rbind,rows))
names(answer) <- c('case','dose1','dose2','efficacy','toxicity','probability')
write.table(answer,stdout(),row.names=FALSE,sep=',',quote=FALSE)
