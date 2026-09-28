# Independent shared-beta interval-PH reference through an exact reduction:
# observing only (0,1] or (1,Inf] gives a binomial complementary-log-log model.
# A separate stratum intercept represents log baseline cumulative hazard at 1.
options(warn=2, digits=17)
d <- data.frame(stratum=factor(c('A','A','B','B')),x=c(0,1,0,1),
                events=c(2,5,4,7),nonevents=c(8,5,6,3))
fit <- glm(cbind(events,nonevents)~0+stratum+x,data=d,
           family=binomial(link='cloglog'),control=glm.control(epsilon=1e-12,maxit=100))
stopifnot(fit$converged)
rows <- list()
for(i in seq_len(nrow(d))) {
  rows[[length(rows)+1L]] <- data.frame(stratum=as.character(d$stratum[i]),
    x=d$x[i],lower=0,upper=1,weight=d$events[i])
  rows[[length(rows)+1L]] <- data.frame(stratum=as.character(d$stratum[i]),
    x=d$x[i],lower=1,upper=Inf,weight=d$nonevents[i])
}
write.csv(do.call(rbind,rows),'tests/fixtures/interval-stratified-input.csv',row.names=FALSE)
co <- coef(fit)
prob <- fitted(fit)
ll <- sum(d$events*log(prob)+d$nonevents*log1p(-prob))
write.csv(data.frame(quantity=c('baseline_A','baseline_B','beta','log_likelihood'),
  value=c(exp(co[1]),exp(co[2]),co[3],ll)),
  'tests/fixtures/interval-stratified-fit.csv',row.names=FALSE)
pred <- expand.grid(stratum=c('A','B'),x=c(-.5,0,.5,1,1.5),stringsAsFactors=FALSE)
pred$survival <- exp(-exp(co[paste0('stratum',pred$stratum)]+co['x']*pred$x))
write.csv(pred,'tests/fixtures/interval-stratified-survival.csv',row.names=FALSE)
cat('Generated weighted 8-row interval reduction, shared fit and 10 predictions.\n')
