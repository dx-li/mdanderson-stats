# Source-only dataprep/Surv2 reference. Uses base R and the cached pinned source.
options(warn = 1, digits = 17)
all_args <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", all_args[grepl("^--file=", all_args)])
if (length(script_arg) != 1) stop("cannot resolve this script's path")
repo <- normalizePath(file.path(dirname(normalizePath(script_arg)), ".."))
native <- Sys.getenv("INTCCR_R_SOURCE", file.path(repo, "research/raw/intccr/R"))
out <- Sys.getenv("INTCCR_FIXTURE_DIR", file.path(repo, "tests/fixtures"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
expected <- c(
  "dataprep.R" = "371ac55900e8434a264c557e1933b749f3918576",
  "Surv2.R" = "c32f52d7c8c275cda51dd23e9868d9d7e2615fd9"
)
for (name in names(expected)) {
  actual <- system2(
    "git",
    c("-C", shQuote(repo), "hash-object", shQuote(file.path(native, name))),
    stdout = TRUE
  )
  stopifnot(identical(unname(actual), unname(expected[name])))
}
source(file.path(native, "dataprep.R"))
source(file.path(native, "Surv2.R"))
input_path <- Sys.getenv("INTCCR_INPUT_CSV", file.path(repo, "tests/fixtures/intccr-visits-input.csv"))
rows <- read.csv(input_path, na.strings = "NA")
rows <- rows[, c("id", "time", "event", "x")]
fit_sorted <- suppressWarnings(dataprep(rows, ID = id, time = time, event = event, Z = x))
write.csv(fit_sorted, file.path(out, "intccr-visits-native-sorted.csv"), row.names = FALSE)
# Reverse the first subject's rows to expose the source's order(ID & time) defect.
permuted <- rows[c(2, 1, 3, 4, 5, 6, 7, 8, 9, 10), ]
fit_unsorted <- suppressWarnings(dataprep(permuted, ID = id, time = time, event = event, Z = x))
write.csv(fit_unsorted, file.path(out, "intccr-visits-native-unsorted.csv"), row.names = FALSE)
valid_surv <- as.data.frame(Surv2(v = c(0, 1, 2), u = c(1, 2, Inf), event = c(1, 2, 0)))
write.csv(valid_surv, file.path(out, "intccr-visits-surv2.csv"), row.names = FALSE)
exact_error <- tryCatch(
  { Surv2(v = 1, u = 1, event = 1); "ACCEPTED" },
  error = function(e) conditionMessage(e)
)
writeLines(exact_error, file.path(out, "intccr-visits-surv2-exact-error.txt"))
