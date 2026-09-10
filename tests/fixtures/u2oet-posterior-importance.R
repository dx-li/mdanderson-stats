set.seed(7703)
N <- 200000
mu <- rep(0,12); sd <- rep(.2,12)
mu[c(1,7)] <- c(-.3,-.8); sd[c(1,7)] <- .3
mu[c(2,3,8,9)] <- .5
x <- sapply(1:12,function(i) {
  if(i %in% c(2,3,8,9)) qnorm(runif(N,pnorm(0,mu[i],sd[i]),1),mu[i],sd[i])
  else rnorm(N,mu[i],sd[i])
})
rho <- runif(N,-1,1)
prob <- function(block,i,j) {
  d <- c(1,2,3)
  standard <- function(i,p) .5 + ((d[i]-1)/2)^p
  a <- block[,1]; b1 <- block[,2]; b2 <- block[,3]
  eta <- a + b1*log(standard(i,exp(block[,4]))) + b2*log(standard(j,exp(block[,5])))
  phi <- exp(block[,6])
  1-(1+phi*exp(eta))^(-1/phi)
}
w <- rep(1,N)
for (obs in list(c(1,1,0,0),c(2,2,1,0),c(3,3,1,1),c(1,3,1,0))) {
  e <- prob(x[,1:6],obs[1],obs[2]); t <- prob(x[,7:12],obs[1],obs[2])
  pe <- if(obs[3]==1) e else 1-e; pt <- if(obs[4]==1) t else 1-t
  ae <- if(obs[3]==1) e-1 else e; at <- if(obs[4]==1) t-1 else t
  w <- w * pe*pt*(1+rho*ae*at)
}
y <- cbind(x,rho)
m <- colSums(y*w)/sum(w)
mcse <- sqrt(colSums((sweep(y,2,m))^2*w^2))/sum(w)
write.table(data.frame(mean=m,mcse=mcse),stdout(),sep=',',row.names=FALSE,quote=FALSE)
cat('ESS',sum(w)^2/sum(w*w),'\n',file=stderr())
