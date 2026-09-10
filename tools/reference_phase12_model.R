# Independent posterior integration: mixture of inflated Laplace and prior normals.
set.seed(8521)
s <- .7071067811865
D <- rbind(c(0,-s,-s,-1),c(0,-s,-s,1),c(-s,s,0,-1),
           c(-s,s,0,1),c(-s,0,s,-1),c(-s,0,s,1))
y <- 1:6
n <- rep(20,6)
mu <- c(6.2445,2.0815,2.0815,0)
sd <- rep(3.16227766,4)
logpost <- function(b) {
 eta <- drop(D %*% b)
 sum(y*plogis(eta,log.p=TRUE)+(n-y)*plogis(-eta,log.p=TRUE)) + sum(dnorm(b,mu,sd,log=TRUE))
}
mode <- optim(mu,function(b) -logpost(b),method='BFGS',hessian=TRUE,
              control=list(reltol=1e-12,maxit=1000))
stopifnot(mode$convergence==0)
R <- chol(1.5*chol2inv(chol(mode$hessian)))
M <- 200000
B <- sweep(matrix(rnorm(M*4),M,4)%*%R,2,mode$par,'+')
use_prior <- runif(M)<.05
B[use_prior,] <- sweep(sweep(matrix(rnorm(sum(use_prior)*4),sum(use_prior),4),2,sd,'*'),2,mu,'+')
eta <- B %*% t(D)
ll <- rowSums(sweep(plogis(eta,log.p=TRUE),2,y,'*') +
              sweep(plogis(-eta,log.p=TRUE),2,n-y,'*'))
lp <- rowSums(sapply(1:4,function(j) dnorm(B[,j],mu[j],sd[j],log=TRUE)))
z <- forwardsolve(t(R),t(sweep(B,2,mode$par,'-')))
lq1 <- log(.95)-2*log(2*pi)-sum(log(diag(R)))-.5*colSums(z*z)
lq2 <- log(.05)+lp
m <- pmax(lq1,lq2)
lq <- m+log(exp(lq1-m)+exp(lq2-m))
lw <- ll+lp-lq
w <- exp(lw-max(lw)); w <- w/sum(w)
values <- cbind(B,plogis(eta))
means <- colSums(values*w)
se <- sqrt(colSums(sweep(values,2,means,'-')^2*w^2))
write.table(cbind(means,se),'tests/fixtures/phase12-model-r.csv',sep=',',row.names=FALSE,col.names=FALSE)
cat('Importance ESS:',1/sum(w^2),'\n')
writeLines(sprintf('{"draws": %d, "seed": 8521, "importance_ess": %.17g, "method": "Independent R mixture importance sampling; diagonal normal prior, six binomial outcomes"}',M,1/sum(w^2)), 'docs/phase12-model-reference.json')
