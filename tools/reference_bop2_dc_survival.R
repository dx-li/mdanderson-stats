# Independent exponential/inverse-gamma posterior, using base R only.
# Prior is on exponential mean; LRV/CMV are median survival times.
x <- data.frame(
  case=c('interim_futile','interim_continue','final_futile','final_consider',
         'final_go','no_events','no_enrollment'),
  n=c(10,10,40,40,40,10,0), events=c(10,5,40,20,10,0,0),
  total_time=c(10,60,40,140,140,30,0)
)
x$max_subjects <- 40
x$lrv <- 3
x$cmv <- 5
x$prior_shape <- 1e-6
x$prior_scale <- 1e-6
x$lambda_lrv <- .9
x$lambda_cmv <- .5
x$gamma_lrv <- .5
x$gamma_cmv <- .5
x$posterior_shape <- x$prior_shape + x$events
x$posterior_lrv <- pgamma((x$prior_scale+x$total_time)*log(2)/x$lrv, x$posterior_shape)
x$posterior_cmv <- pgamma((x$prior_scale+x$total_time)*log(2)/x$cmv, x$posterior_shape)
x$decision <- 'continue'
for (i in seq_len(nrow(x))) {
  r <- x[i,]
  if (r$n == r$max_subjects) {
    x$decision[i] <- if (r$posterior_lrv > r$lambda_lrv && r$posterior_cmv > r$lambda_cmv) 'final_go' else if (r$posterior_lrv < r$lambda_lrv && r$posterior_cmv < r$lambda_cmv) 'final_no_go' else 'final_consider'
  } else if (r$n %in% c(10,20,30)) {
    if (r$posterior_lrv < r$lambda_lrv*(r$n/r$max_subjects)^r$gamma_lrv && r$posterior_cmv < r$lambda_cmv*(r$n/r$max_subjects)^r$gamma_cmv) x$decision[i] <- 'stop_no_go'
  }
}
write.csv(x, 'tests/fixtures/bop2-dc-survival.csv', row.names=FALSE)
