x <- as.matrix(read.csv('research/raw/P12Xuelin/beta-comparisons.csv',header=FALSE))
p <- apply(x,1,function(z) integrate(function(t) pbeta(t,z[3],z[4])*dbeta(t,z[1],z[2]),0,1,rel.tol=1e-10,subdivisions=2000)$value)
cat('Comparisons:',length(p),'max absolute error:',max(abs(p-x[,5])),'\n')
write.table(cbind(x,p),'research/raw/P12Xuelin/beta-r-reference.csv',sep=',',row.names=FALSE,col.names=FALSE)
chosen <- unique(c(1,which.min(p),which.max(p),which.min(abs(p-.5)),which.min(abs(p-.8))))
write.table(cbind(x[chosen,1:4],p[chosen]),'tests/fixtures/parallel-phase12-beta.csv',sep=',',row.names=FALSE,col.names=FALSE)

stopifnot(max(abs(p-x[,5])) < 2e-9)
writeLines(sprintf('{"comparisons": %d, "max_absolute_error": %.17g, "reference": "Independent R beta CDF/density quadrature, relative tolerance 1e-10"}',length(p),max(abs(p-x[,5]))),'docs/parallel-phase12-numerics.json')
