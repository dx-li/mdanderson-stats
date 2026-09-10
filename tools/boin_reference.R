# Run with BOIN 2.7.2 installed, from the repository root. Independent output fixture.
library(BOIN)
stopifnot(as.character(packageVersion("BOIN")) == "2.7.2")
set.seed(120)
rows <- list()
for (i in 1:64) {
  n <- c(sample(c(3, 6, 9, 12, 15, 24, 30), 5, replace = TRUE), 0)
  y <- rbinom(6, n, sort(runif(6, 0, .8)))
  target <- c(.15, .25, .3, .4)[1 + (i %% 4)]
  extra <- i %% 3 == 0
  bounded <- i %% 2 == 0
  result <- select.mtd(target, n, y, extrasafe = extra, boundMTD = bounded)
  rows[[i]] <- data.frame(
    target = target, patients = paste(n, collapse = ";"),
    toxicities = paste(y, collapse = ";"), extra_safe = extra,
    bound_mtd = bounded, mtd = result$MTD,
    estimate = paste(result$p_est$phat, collapse = ";")
  )
}
write.csv(do.call(rbind, rows), "tests/fixtures/boin-reference.csv", row.names = FALSE)

oc <- get.oc(.3, c(.05,.15,.3,.45,.6), 10, 3,
             n.earlystop = 100, ntrial = 10000, seed = 120)
write.csv(data.frame(dose = 1:5, selection_probability = oc$selpercent / 100,
                    mean_patients = oc$npatients, mean_toxicities = oc$ntox),
          "tests/fixtures/boin-oc-reference.csv", row.names = FALSE)

oc <- get.oc(.3, c(.05,.15,.3,.45,.6), 10, 3, titration = TRUE,
             n.earlystop = 100, ntrial = 10000, seed = 120)
write.csv(data.frame(dose = 1:5, selection_probability = oc$selpercent / 100,
                    mean_patients = oc$npatients, mean_toxicities = oc$ntox),
          "tests/fixtures/boin-titration-reference.csv", row.names = FALSE)
