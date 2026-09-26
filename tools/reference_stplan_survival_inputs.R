# Independent survival-input references using linear systems in cumulative
# hazard space. Base R only. Original STPLAN code is not used here.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=1L) stop('Supply output CSV path')
rows <- list()
add <- function(id, times, survival, change=NA_real_) {
  n <- length(times)
  H <- -log(survival)
  if (is.na(change)) {
    # Fit the late cumulative-hazard line, independently of the Python formula.
    late <- solve(cbind(1,times[2:3]),H[2:3])
    first <- H[1]/times[1]
    second <- late[2]
    identified <- abs(first-second)>64*.Machine$double.eps*max(abs(c(first,second)))
    change <- if (identified) late[1]/(first-second) else mean(times[1:2])
  } else {
    design <- cbind(pmin(times,change),pmax(times-change,0))
    hazards <- solve(design,H)
    first <- hazards[1]; second <- hazards[2]
    identified <- abs(first-second)>64*.Machine$double.eps*max(abs(c(first,second)))
  }
  rows[[length(rows)+1L]] <<- data.frame(case_id=id,points=n,
    t1=times[1],t2=times[2],t3=if(n==3)times[3] else NA_real_,
    s1=survival[1],s2=survival[2],s3=if(n==3)survival[3] else NA_real_,
    hazard_before=first,hazard_after=second,change_time=change,
    identified=identified)
}
curve <- function(t,h1,h2,b) exp(-h1*pmin(t,b)-h2*pmax(t-b,0))
add('known_straddling',c(2,8),curve(c(2,8),.2,.05,4),4)
add('known_both_after',c(6,10),curve(c(6,10),.2,.05,4),4)
add('known_at_break',c(4,10),curve(c(4,10),.2,.05,4),4)
add('known_increasing_hazard',c(1,8),curve(c(1,8),.03,.2,3),3)
add('known_zero_early',c(2,8),curve(c(2,8),0,.1,4),4)
add('known_zero_late',c(2,8),curve(c(2,8),.2,0,4),4)
add('known_constant',c(2,8),curve(c(2,8),.1,.1,4),4)
add('unknown_decreasing',c(2,8,10),curve(c(2,8,10),.2,.05,4))
add('unknown_increasing',c(1,6,10),curve(c(1,6,10),.03,.2,3))
add('unknown_zero_early',c(2,8,10),curve(c(2,8,10),0,.1,4))
add('unknown_zero_late',c(2,8,10),curve(c(2,8,10),.2,0,4))
add('unknown_constant',c(2,8,10),curve(c(2,8,10),.1,.1,4))
add('unknown_no_events',c(2,8,10),c(1,1,1))
add('unknown_tiny_survival',c(.1,1,2),curve(c(.1,1,2),1,735,.999))
add('unknown_scaled_times',c(2,8,10)*1e8,curve(c(2,8,10),.2,.05,4))
options(digits=17)
write.table(do.call(rbind,rows),file=args[1],sep=',',row.names=FALSE,quote=FALSE)
