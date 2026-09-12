# Independent base-R control selection and beta posterior comparisons.
enrollment <- 0:5
outcome <- c(1,1,0,0,1,0)
observed <- c(.5,1.5,2.5,3.5,6,5.5)
windows <- rbind(c(0,4),c(3,Inf))
a <- c(4,5); b <- c(2,3) # prior (1,1), successes (3,4), failures (1,2)
counts_rows <- comparisons <- list()
for (now in c(5.75,7)) {
  known <- enrollment <= now & observed <= now
  entire <- c(sum(outcome[known]),sum(1-outcome[known]))
  concurrent <- t(vapply(1:2,function(i) {
    eligible <- known & enrollment >= windows[i,1] & enrollment < windows[i,2]
    c(sum(outcome[eligible]),sum(1-outcome[eligible]))
  },numeric(2)))
  for (i in 1:2) counts_rows[[length(counts_rows)+1]] <- data.frame(
    as_of=now,arm=i,successes=concurrent[i,1],failures=concurrent[i,2])
  for (mode in c('entire','concurrent')) for (i in 1:2) {
    counts <- if (mode=='entire') entire else concurrent[i,]
    ca <- counts[1]+1; cb <- counts[2]+1
    upper <- integrate(function(x)dbeta(x,a[i],b[i])*pbeta(x,ca,cb),0,1,rel.tol=1e-12)$value
    lower <- integrate(function(x)dbeta(x,a[i],b[i])*pbeta(x,ca,cb,lower.tail=FALSE),0,1,rel.tol=1e-12)$value
    comparisons[[length(comparisons)+1]] <- data.frame(as_of=now,mode=mode,arm=i,
      control_successes=counts[1],control_failures=counts[2],
      treatment_greater=upper,control_greater=lower,
      futile=lower>.25,efficacious=upper>=.8,final_efficacious=upper>=.7)
  }
}
write.csv(do.call(rbind,counts_rows),'tests/fixtures/plbarpo-control-counts.csv',row.names=FALSE)
write.csv(do.call(rbind,comparisons),'tests/fixtures/plbarpo-control-posterior.csv',row.names=FALSE)
