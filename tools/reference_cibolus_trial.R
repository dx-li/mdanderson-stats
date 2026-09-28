# Independent complete-outcome CiBolus reference using base-R quadrature.
# Fixed priors isolate allocation and observation conversion from Monte Carlo.
options(warn = 2, digits = 17)
concentrations <- c(.2, .4, .8)
bolus <- c(.1, .6)
endpoints <- c(.25, .5, .75, 1)
truth <- c(.5, .7, .8, .08, 1.4, 1.6, .03, .9, .12, .25, .2)
utilities <- cbind(c(rep(100, 5), 0), rep(0, 6))

predict_pair <- function(c, q, p) {
  hazard <- function(t) {
    exposure <- c^p[2] * (q^p[3] + (1-q^p[3])*t)
    p[4] + p[5]*p[6]*exposure^(p[6]-1)/(1+p[5]*exposure^p[6])
  }
  survival <- function(t) exp(-p[1]*c^p[2]*q^p[3] -
    integrate(hazard, 0, t, rel.tol=1e-11, abs.tol=1e-13)$value)
  response_mass <- c(
    -expm1(-p[1]*c^p[2]*q^p[3]),
    vapply(seq_along(endpoints), function(i)
      integrate(function(t) vapply(t, function(s) survival(s)*hazard(s), numeric(1)),
        c(0, endpoints)[i], endpoints[i], rel.tol=1e-10, abs.tol=1e-12)$value,
      numeric(1)),
    survival(1)
  )
  times <- c(0, endpoints, 1)
  failure <- c(rep(0, 5), 1)
  toxic <- -expm1(-(p[7]+p[9]*c^p[8]*q+p[10]*c^p[8]*(1-q)*times+p[11]*failure))
  joint <- cbind(response_mass*(1-toxic), response_mass*toxic)
  stopifnot(abs(sum(joint)-1) < 1e-10)
  list(joint=joint, utility=sum(joint*utilities), efficacy=1-response_mass[6],
    conditional_toxicity=-expm1(-(p[7]+p[9]*c^p[8]*q+p[10]*c^p[8]*(1-q))))
}

grid <- expand.grid(bolus_index=0:1, concentration_index=0:2)
truth_predictions <- lapply(seq_len(nrow(grid)), function(i)
  predict_pair(concentrations[grid$concentration_index[i]+1],
    bolus[grid$bolus_index[i]+1], truth))
joint_rows <- list()
for (i in seq_len(nrow(grid))) {
  joint <- truth_predictions[[i]]$joint
  for (category in 0:5) {
    joint_rows[[length(joint_rows)+1]] <- data.frame(
      concentration_index=grid$concentration_index[i], bolus_index=grid$bolus_index[i],
      category, no_toxicity=joint[category+1,1], toxicity=joint[category+1,2]
    )
  }
}

patient_rows <- look_rows <- summary_rows <- list()
uniforms <- c(0, .2, .6, .95)
for (scenario in c("safe_final_unrestricted", "unsafe_after_first_cohort")) {
  prior <- truth
  if (scenario == "unsafe_after_first_cohort") prior[7] <- 3
  model <- lapply(seq_len(nrow(grid)), function(i)
    predict_pair(concentrations[grid$concentration_index[i]+1],
      bolus[grid$bolus_index[i]+1], prior))
  utility <- vapply(model, function(x) x$utility, numeric(1))
  ptox <- as.integer(vapply(model, function(x) x$conditional_toxicity > .8, logical(1)))
  pineff <- as.integer(vapply(model, function(x) x$efficacy < .01, logical(1)))
  acceptable <- ptox <= .9 & pineff <= .9
  current <- 1L
  treated <- integer(nrow(grid))
  selected <- NA_integer_
  enrolled <- 0L
  for (cohort in 1:2) {
    for (offset in 1:2) {
      enrolled <- enrolled+1L
      joint <- truth_predictions[[current]]$joint
      cdf <- cumsum(as.vector(t(joint)))
      cdf[length(cdf)] <- 1
      cell <- sum(cdf <= uniforms[enrolled])
      category <- cell %/% 2L
      toxicity <- cell %% 2L
      patient_rows[[length(patient_rows)+1]] <- data.frame(scenario, patient=enrolled-1L,
        cohort, concentration_index=grid$concentration_index[current],
        bolus_index=grid$bolus_index[current], uniform=uniforms[enrolled],
        category, toxicity)
      treated[current] <- treated[current]+1L
    }
    final <- enrolled == 4
    eligible <- acceptable
    if (!final) {
      max_tried <- max(grid$concentration_index[treated > 0])
      eligible <- eligible & grid$concentration_index <= max_tried+1L
    }
    selected <- if (any(eligible)) which.max(ifelse(eligible, utility, -Inf)) else NA_integer_
    for (i in seq_len(nrow(grid))) {
      look_rows[[length(look_rows)+1]] <- data.frame(scenario, cohort, final,
        concentration_index=grid$concentration_index[i], bolus_index=grid$bolus_index[i],
        treated=treated[i], utility=utility[i], posterior_toxicity=ptox[i],
        posterior_inefficacy=pineff[i], acceptable=acceptable[i], eligible=eligible[i],
        selected=!is.na(selected) && i == selected)
    }
    if (is.na(selected)) break
    current <- selected
  }
  summary_rows[[length(summary_rows)+1]] <- data.frame(scenario, enrolled,
    fits=cohort, selected_concentration=if (is.na(selected)) NA else grid$concentration_index[selected],
    selected_bolus=if (is.na(selected)) NA else grid$bolus_index[selected])
}
write.csv(do.call(rbind,joint_rows), "tests/fixtures/cibolus-trial-joint.csv", row.names=FALSE)
write.csv(do.call(rbind,patient_rows), "tests/fixtures/cibolus-trial-patients.csv", row.names=FALSE)
write.csv(do.call(rbind,look_rows), "tests/fixtures/cibolus-trial-looks.csv", row.names=FALSE)
write.csv(do.call(rbind,summary_rows), "tests/fixtures/cibolus-trial-summary.csv", row.names=FALSE)
