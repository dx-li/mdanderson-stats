# Independent tensor Gauss-Hermite integration of the two-dose probit posterior.
calculate <- function(n) {
  J <- matrix(0,n,n)
  J[cbind(1:(n-1),2:n)] <- sqrt(1:(n-1))
  J <- J+t(J)
  eigen <- eigen(J,symmetric=TRUE)
  nodes <- eigen$values
  weights <- eigen$vectors[1,]^2
  b1 <- rep(.3+sqrt(1.2)*nodes,each=n)
  b2 <- b1+sqrt(1.2)*rep(nodes,n)
  prior <- rep(weights,each=n)*rep(weights,n)
  likelihood <- pnorm(b1)^1*pnorm(-b1)^3*pnorm(b2)^3*pnorm(-b2)^2
  w <- prior*likelihood
  w <- w/sum(w)
  colSums(cbind(b1,b2,pnorm(b1),pnorm(b2))*w)
}
a <- calculate(80)
b <- calculate(120)
stopifnot(max(abs(a-b))<1e-10)
write.csv(data.frame(quantity=c('beta0','beta1','risk0','risk1'),mean=b,
                     refinement_difference=abs(a-b)),
          'tests/fixtures/prt-fit-r.csv',row.names=FALSE)
print(b)
# Covariance-weighted min-max transformation, independently evaluated in R.
set.seed(6912)
p <- matrix(runif(180,.05,.95),ncol=3)
V <- cov(p)
q <- matrix(NA,nrow(p),3)
for (k in 1:3) {
  choices <- list()
  for (upper in k:3) {
    candidates <- list()
    for (lower in 1:k) {
      ids <- lower:upper
      weight <- solve(V[ids,ids,drop=FALSE],rep(1,length(ids)))
      candidates[[length(candidates)+1]] <- as.vector(p[,ids,drop=FALSE] %*% weight/sum(weight))
    }
    choices[[length(choices)+1]] <- do.call(pmax,candidates)
  }
  q[,k] <- do.call(pmin,choices)
}
stopifnot(all(q>=0),all(q<=1),all(q[,1]<=q[,2]),all(q[,2]<=q[,3]))
write.csv(data.frame(p0=p[,1],p1=p[,2],p2=p[,3],q0=q[,1],q1=q[,2],q2=q[,3]),
          'tests/fixtures/prt-isotonic-r.csv',row.names=FALSE)
