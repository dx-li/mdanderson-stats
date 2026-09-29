# Native icenReg 2.0.16 semiparametric PH optimizer and support references.
# Compile the unchanged core with existing RcppEigen; install no packages.
options(warn=2, digits=17)
native <- "research/raw/icenReg"
expected <- c(
  "R/internal_utilities.R"="828ecb5c6afba2be725064b3a7decb505c94e15d",
  "R/ic_sp.R"="0a51f8e92a04249ded93bd6c7476bc6b24d28275",
  "src/icenReg_files/myPAVAalgorithm.cpp"="e2e5a58d1a06502691aadeaa2caee832c8182ec3",
  "src/icenReg_files/basicUtilities.cpp"="e153f08891dce000a0a4d1589dfbc079bc91fec6",
  "src/icenReg_files/ic_sp_ch.h"="0d9cc6875f0c2af7f7b7e3ef30a84fc7ef8e9405",
  "src/icenReg_files/ic_sp_ch.cpp"="140c37b1e3a6a03dcda6157524e10fa612d54c73",
  "src/icenReg_files/ic_sp_gradDescent.cpp"="8136d221ef5988687fb6788f861136d705efb751")
for (name in names(expected)) {
  actual <- system2("git", c("hash-object", shQuote(file.path(native,name))), stdout=TRUE)
  stopifnot(identical(unname(actual), unname(expected[name])))
}
scratch <- "research/raw/interval-reference"
dir.create(scratch, showWarnings=FALSE)
makevars <- file.path(scratch,"Makevars")
writeLines(c("CXXFLAGS=-O0 -g0", "CXX11FLAGS=-O0 -g0", "CXX14FLAGS=-O0 -g0",
             "CXX17FLAGS=-O0 -g0"),makevars)
Sys.setenv(R_MAKEVARS_USER=normalizePath(makevars), MAKEFLAGS="-j1")
# Inline source files only to resolve includes. Function bodies are unchanged.
parts <- c("myPAVAalgorithm.cpp", "basicUtilities.cpp", "ic_sp_ch.h",
           "ic_sp_ch.cpp", "ic_sp_gradDescent.cpp")
bodies <- lapply(parts,function(name) {
  lines <- readLines(file.path(native,"src/icenReg_files",name),warn=FALSE)
  lines[!grepl('^#include "(myPAVAalgorithm.cpp|ic_sp_ch.h)"',lines)]
})
prefix <- c('// [[Rcpp::depends(RcppEigen)]]', '#include <RcppEigen.h>',
            '#include <Rmath.h>', '#include <vector>', 'using namespace std;')
wrappers <- c(
  '// [[Rcpp::export]]',
  'SEXP reference_mi(SEXP values, SEXP il, SEXP ir, SEXP l, SEXP r) {',
  '  return findMI(values, il, ir, l, r);', '}',
  '// [[Rcpp::export]]',
  'SEXP reference_fit(SEXP l, SEXP r, SEXP x, SEXP kind, SEXP w, SEXP ga,',
  '                   SEXP iterations, SEXP updates, SEXP full, SEXP regress, SEXP start) {',
  '  return ic_sp_ch(l, r, x, kind, w, ga, iterations, updates, full, regress, start);', '}',
  '// [[Rcpp::export]]',
  'Rcpp::NumericVector reference_curves(Rcpp::NumericVector baseline, double eta) {',
  '  icm_ph model; Rcpp::NumericVector result(baseline.size());',
  '  for(int i=0; i<baseline.size(); ++i) result[i]=model.baseS2CondS(baseline[i],eta);',
  '  return result;', '}')
cpp <- file.path(scratch,"native.cpp")
writeLines(c(prefix,unlist(bodies),wrappers),cpp)
Rcpp::sourceCpp(cpp,cacheDir=file.path(scratch,"cache"),showOutput=FALSE)
source(file.path(native,"R/internal_utilities.R"))

native_fit <- function(lower,upper,x,w) {
  adjusted <- adjustIntervals(c(0,1),cbind(lower,upper))
  values <- sort(unique(as.numeric(adjusted)))
  mi <- reference_mi(values,values %in% adjusted[,1],values %in% adjusted[,2],
                     adjusted[,1],adjusted[,2])
  names(mi) <- c("l_inds","r_inds","lower","upper")
  center <- colMeans(x)
  xc <- sweep(x,2,center)
  result <- reference_fit(mi$l_inds,mi$r_inds,xc,1L,as.double(w),TRUE,
                          10000L,5L,TRUE,TRUE,rep(0,ncol(x)))
  names(result) <- c("mass","coefficients","log_likelihood","iterations","score")
  # The native C helper returns negative differences; ic_sp.R normalizes them.
  result$mass <- result$mass/sum(result$mass)
  stopifnot(all(is.finite(result$mass)),all(result$mass>=-1e-12),
            is.finite(result$log_likelihood),result$iterations<10000)
  # Map native epsilon-shifted lower boundaries back to input endpoints.
  original_lower <- lower[match(mi$lower,adjusted[,1])]
  left_closed <- mi$lower==original_lower
  list(support=mi,fit=result,center=center,original_lower=original_lower,
       left_closed=left_closed)
}
