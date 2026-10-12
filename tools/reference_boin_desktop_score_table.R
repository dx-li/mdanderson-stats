# Execute the hash-pinned original table; distribute synthetic outputs only.
args<-commandArgs(TRUE);stopifnot(length(args)==2)
path<-file.path(args[1],'get.scoretable.R')
stopifnot(unname(tools::md5sum(path))=='58dfed737f8b60734e5e8f025eda48b9')
native<-new.env(parent=globalenv());sys.source(path,envir=native)
rows<-list();case<-0L
for(target in c(.25,.3)) for(cohort in c(1L,3L)) for(maximum in c(12L,30L))
for(upper in c(1.4*target,.5)) {
 case<-case+1L
 tab<-native$get.scoretable(target,.6*target,upper,cohort,maximum%/%cohort,.95)
 rows[[case]]<-data.frame(case=case,target=target,cohort=cohort,maximum=maximum,
  lower=.6*target,upper=upper,patients=tab[,1],toxicities=tab[,2],rank=tab[,3],
  visible=as.numeric(tab[,1])<=12)
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
