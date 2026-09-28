# Small reference cases from the inspected bacistool 1.0.0 ESS function.
# Load only its numerical compESS expression, not the JAGS trial workflow.
# The original package is a local research input and is not redistributed.
options(digits=17, warn=2)
source_path <- "research/raw/BaCIS/bacistool_src/R/internal.R"
expressions <- parse(source_path)
selected <- Filter(function(node) {
  is.call(node) && identical(node[[1]],as.name("<-")) &&
    identical(node[[2]],as.name("compESS"))
}, as.list(expressions))
stopifnot(length(selected)==1L)
eval(selected[[1]])

cases <- data.frame(
  case=c("zero_responses", "two_responses", "ten_responses", "two_roots",
         "more_information", "all_responses", "borrowed_information", "large_sample"),
  responses=c(0,2,10,10,10,25,6,100),
  patients=c(25,25,25,25,25,25,25,5000),
  variance_equivalent_n=c(25,25,25,12,60,25,40,10000))
y <- cases$responses
n <- cases$variance_equivalent_n
cases$posterior_variance <- (y+1)*(n-y+1)/((n+2)^2*(n+3))
cases$native_ess <- vapply(seq_len(nrow(cases)), function(i) {
  compESS(1/cases$posterior_variance[i], y[i], y[i]/cases$patients[i])
}, numeric(1))

# For y=10 and v=11/980, the polynomial factors into
# (N-12)*(N^2+19*N-736). Of its two admissible roots, this root has
# response rate closest to the observed 10/25.
two_root_solution <- (-19+sqrt(3305))/2
expected <- c(0.0001,25,25,two_root_solution,60,25,40,10000)
stopifnot(max(abs(cases$native_ess-expected))<1e-8)
cases$admissible_ess <- expected
# Native clamping/tie ordering gives .0001 at zero responses, which does not
# match the requested variance. The unique admissible solution is exactly25.
cases$admissible_ess[1] <- 25
write.csv(cases,"tests/fixtures/bacis-ess-reference.csv",row.names=FALSE)
cat("Recorded",nrow(cases),"native ESS cases, including the zero-response defect.\n")
