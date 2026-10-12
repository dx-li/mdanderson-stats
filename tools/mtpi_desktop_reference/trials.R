# Hash verification is performed by reference_mtpi_desktop.py before invocation.
args<-commandArgs(TRUE)
source<-readLines(args[1],warn=FALSE)
helpers<-parse(text=source[grep('^pava <- function',source)[1]:(grep('^################## simulation setup',source)[1]-1)])
eval(helpers)
start<-grep('^    x<-rep',source)[1]
end<-grep('rez\\[sim\\] <- seldose',source)[1]-1
code<-parse(text=c('{',source[start:end],'}'))
rows<-list();k<-0L
for(target in c(.1,.25,.3,.5)) for(cohort in c(1L,3L)) for(cap in c(4L,6L,12L))
for(first in c(1L,2L)) for(truth in c('ascending','zero','one')) for(tape in c('low','high','cycle')) {
 k<-k+1L;D<-3;pT<-target;csize<-cohort;sampsize<-cap;startdose<-first;a<-b<-1;eps1<-eps2<-.05;xi<-.95;xi2<-.8
 p<-switch(truth,ascending=c(.1,.3,.7),zero=c(0,0,0),one=c(1,1,1))
 draws<-0L;path<-c();tt<-NULL
 runif<-function(n){stopifnot(n==1,draws<300);if(draws%%csize==0) path<<-c(path,d);draws<<-draws+1L;switch(tape,low=.01,high=.99,cycle=((draws*37)%%101)/101)}
 eval(code)
 rows[[k]]<-data.frame(target=pT,cohort=csize,cap=sampsize,start=startdose,truth=truth,tape=tape,
  n=paste(n,collapse=';'),y=paste(x,collapse=';'),path=paste(path,collapse=';'),selected=seldose,
  nodose=nodose,mindose=mindose,maxdose=maxdose,toxdose=toxdose,tt=tt)
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
