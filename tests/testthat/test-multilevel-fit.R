# Fitting multilevel (partially pooled) models.
#
# test-RE-stan_data.R covers the standata for random effects, but stops at
# `chains = 0`. Nothing previously fitted a hierarchical model and looked at the
# result, so transform_theta_L_draws() (R/backend.R) -- flagged as a landmine in
# AGENTS.md -- and make_Sigma_nms() (R/epim.R) were untested, and `decov` did not
# appear anywhere in the suite.

groups <- c("Austria", "Germany", "Italy")

uncorrelated_rt <- function() {
  epirt(
    formula = R(country, date) ~ 0 + (1 + lockdown || country) + lockdown,
    prior_covariance = decov(shape = c(2, 0.5), scale = 0.25)
  )
}

correlated_rt <- function() {
  epirt(
    formula = R(country, date) ~ 1 + (1 + lockdown | country),
    prior_covariance = decov(scale = 0.25)
  )
}

fit_with <- function(rt, iter = 200) {
  args <- europe_args(groups, iter = iter)
  args$rt <- rt
  suppressWarnings(do.call(epim, args))
}

test_that("uncorrelated (||) effects give one variance per term, no covariances", {
  skip_on_cran()
  skip_if_no_cmdstan()

  fm <- fit_with(uncorrelated_rt())
  nms <- colnames(as.matrix(fm))

  sigma <- grep("Sigma", nms, value = TRUE)
  # `||` makes the terms independent, so only the diagonal is estimated
  expect_setequal(sigma, c("R|Sigma[country:(Intercept),(Intercept)]",
                           "R|Sigma[country:lockdown,lockdown]"))

  # one b per (level, term). pad_reTrms() adds a `_NEW_` level on top of the
  # three countries, which is what lets unseen levels be represented at all.
  b <- grep("^R\\|b\\[", nms, value = TRUE)
  expect_equal(length(b), (length(groups) + 1L) * 2L)
  for (g in groups) {
    expect_true(any(grepl(g, b, fixed = TRUE)), info = g)
  }
})

test_that("a single intercept-only term (1 | group) fits", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # The plainest multilevel model there is, and it used to fail outright:
  # len_theta_L is 1, so the per-draw function in transform_theta_L_draws()
  # returned a scalar, apply() dropped a dimension, and aperm() errored with
  # "'perm' is of wrong length 3 (!= 2)". This is also the `special_case` branch
  # of the Stan model, so it is worth having pinned.
  fm <- fit_with(epirt(formula = R(country, date) ~ 1 + (1 | country)))

  expect_s3_class(fm, "epimodel")
  sigma <- grep("Sigma", colnames(as.matrix(fm)), value = TRUE)
  expect_equal(sigma, "R|Sigma[country:(Intercept),(Intercept)]")
  expect_true(all(as.matrix(fm, regex_pars = "Sigma") > 0))

  b <- grep("^R\\|b\\[", colnames(as.matrix(fm)), value = TRUE)
  expect_equal(length(b), length(groups) + 1L)
})

test_that("correlated (|) effects additionally estimate the covariance", {
  skip_on_cran()
  skip_if_no_cmdstan()

  fm <- fit_with(correlated_rt())
  sigma <- grep("Sigma", colnames(as.matrix(fm)), value = TRUE)

  # the off-diagonal is the whole difference between `|` and `||`
  expect_setequal(sigma, c("R|Sigma[country:(Intercept),(Intercept)]",
                           "R|Sigma[country:lockdown,(Intercept)]",
                           "R|Sigma[country:lockdown,lockdown]"))
})

test_that("Sigma draws are a valid covariance: positive variances, |corr| <= 1", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # transform_theta_L_draws() rebuilds Sigma from theta_L via lme4::mkVarCorr.
  # If that mapping were wrong the names could still look right, so check the
  # numbers obey the constraints a covariance matrix must satisfy.
  fm <- fit_with(correlated_rt())
  mat <- as.matrix(fm, regex_pars = "Sigma")

  v_int <- mat[, "R|Sigma[country:(Intercept),(Intercept)]"]
  v_lock <- mat[, "R|Sigma[country:lockdown,lockdown]"]
  cov_il <- mat[, "R|Sigma[country:lockdown,(Intercept)]"]

  expect_true(all(v_int > 0))
  expect_true(all(v_lock > 0))
  expect_true(all(abs(cov_il) <= sqrt(v_int * v_lock) + 1e-8))
})

test_that("as.matrix subsets multilevel parameters by model, type and group", {
  skip_on_cran()
  skip_if_no_cmdstan()

  fm <- fit_with(uncorrelated_rt())

  fixed <- as.matrix(fm, par_models = "R", par_types = "fixed")
  expect_equal(colnames(fixed), "R|lockdown")

  italy <- as.matrix(fm, regex_pars = "^R\\|b", par_groups = "Italy")
  expect_equal(nrow(italy), posterior_sample_size(fm))
  expect_true(all(grepl("Italy", colnames(italy), fixed = TRUE)))
  expect_equal(length(colnames(italy)), 2L)

  # the vignette adds the group deviation to the global effect this way, so the
  # two must be conformable
  expect_equal(nrow(fixed), nrow(italy))
})

test_that("posterior_rt returns a series for every fitted group", {
  skip_on_cran()
  skip_if_no_cmdstan()

  args <- europe_args(groups)
  args$rt <- uncorrelated_rt()
  fm <- suppressWarnings(do.call(epim, args))

  rt <- posterior_rt(fm)
  expect_setequal(as.character(unique(rt$group)), groups)
  expect_equal(dim(rt$draws), c(posterior_sample_size(fm), nrow(args$data)))
  # R_t is a rate; the scaled_logit/log link cannot produce negatives
  expect_true(all(rt$draws > 0))
})

test_that("a group absent from the fit is dropped from a forecast, not invented", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # pad_reTrms() carries a `_NEW_` level, so it is reasonable to wonder whether
  # passing newdata with an unseen group predicts for it. It does not: the
  # result is silently restricted to the groups the model was fitted on. Pinned
  # here so the behaviour is deliberate rather than incidental.
  fitted_on <- c("Austria", "Germany")
  args <- europe_args(fitted_on)
  args$rt <- epirt(formula = R(country, date) ~ 1 + (1 | country))
  fm <- suppressWarnings(do.call(epim, args))

  rt <- posterior_rt(fm, newdata = europe_data(groups))
  expect_setequal(as.character(unique(rt$group)), fitted_on)
  expect_false("Italy" %in% as.character(unique(rt$group)))
})

test_that("group-level random walks get one process and one scale per group", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_unless_slow_tests()

  # rw(gr = ...) is documented in man/rw.Rd but used in no vignette and asserted
  # nowhere: test-epim.R only checked that a fit with it returns an epimodel.
  args <- europe_args(groups)
  args$rt <- epirt(formula = R(country, date) ~ 1 + rw(time = week, gr = country))
  fm <- suppressWarnings(do.call(epim, args))

  nms <- colnames(as.matrix(fm))
  label <- "rw(time = week, gr = country)"

  scales <- grep(paste0("^R\\|sigma:", gsub("([().|])", "\\\\\\1", label)),
                 nms, value = TRUE)
  expect_equal(length(scales), length(groups))
  for (g in groups) {
    expect_true(any(grepl(paste0("[", g, "]"), scales, fixed = TRUE)), info = g)
  }

  # and the walk itself is indexed by (time, group), so every group has its own
  walk <- grep("^R\\|rw\\(", nms, value = TRUE)
  expect_gt(length(walk), length(groups))
  for (g in groups) {
    expect_true(any(grepl(paste0(",", g, "]"), walk, fixed = TRUE)), info = g)
  }
})

test_that("a shared random walk has a single scale regardless of group count", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_unless_slow_tests()

  # omitting `gr` puts every group on one walk (get_autocor_gr() falls back to
  # the literal group "all"), which is the contrast that makes rw(gr=) meaningful
  args <- europe_args(groups)
  args$rt <- epirt(formula = R(country, date) ~ 1 + rw(time = week))
  fm <- suppressWarnings(do.call(epim, args))

  scales <- grep("^R\\|sigma:", colnames(as.matrix(fm)), value = TRUE)
  expect_equal(length(scales), 1L)
  expect_true(grepl("all", scales, fixed = TRUE))
})
