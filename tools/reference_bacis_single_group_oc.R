# Independent binomial-sum oracle for BaCIS's singleton Beta(1+y,1+n-y).
options(warn=2, digits=17)
n <- 25L
low <- .1
high <- .3
cutoff <- .92
successes <- 0:n
# Beta upper tail equals Pr[Binomial(n+1,p) <= y].
low_tail <- vapply(successes, function(y) sum(dbinom(0:y,n+1,low)), 0.0)
high_tail <- vapply(successes, function(y) sum(dbinom(0:y,n+1,high)), 0.0)
effective <- low_tail > cutoff
out <- data.frame(successes=successes,trials=n,phi_low=low,phi_high=high,
                  cutoff=cutoff,low_tail=low_tail,high_tail=high_tail,
                  effective=as.integer(effective),
                  null_mass=dbinom(successes,n,low),
                  alternative_mass=dbinom(successes,n,high))
write.csv(out,"tests/fixtures/bacis-single-group-oc.csv",row.names=FALSE)
cat("Minimum effective response count:",min(successes[effective]),"\n")
cat("Exact subgroup type I / single-null-group FWER:",sum(out$null_mass[effective]),"\n")
cat("Exact subgroup power:",sum(out$alternative_mass[effective]),"\n")
