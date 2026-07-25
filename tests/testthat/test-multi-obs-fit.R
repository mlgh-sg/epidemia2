# Fitting a model with more than one observation series.
#
# Until now multi-series models were only ever built as standata via the
# `chains = 0` short-circuit and never actually fitted, so nothing exercised the
# concatenated obs/obs_type/obs_group layout from standata_obs()
# (R/stan_data.R) or the unrolled oX1..oX10 block in
# inst/stan/tparameters/make_oeta.stan.

groups <- c("Austria", "Germany")

test_that("standata describes both series", {
  args <- europe_multiobs_args(groups)
  args$chains <- 0
  sdat <- suppressMessages(suppressWarnings(do.call(epim, args)))

  expect_equal(sdat$R, 2L)
  expect_equal(sdat$N_obs, sum(sdat$oN[1:2]))
  # obs_type indexes into the per-series arrays, so it must only ever name a
  # series that exists.
  expect_equal(sort(unique(sdat$obs_type)), 1:2)
  expect_equal(length(sdat$obs), sdat$N_obs)
  expect_equal(length(sdat$obs_group), sdat$N_obs)
  expect_equal(length(sdat$obs_date), sdat$N_obs)
  # both series are neg_binom here, so each needs its own dispersion parameter
  expect_equal(sdat$num_oaux, 2L)
  expect_equal(sdat$ofamily, array(c(2L, 2L)))
})

test_that("a joint two-series model fits and keeps the series distinct", {
  skip_on_cran()
  skip_if_no_cmdstan()

  args <- europe_multiobs_args(groups)
  fm <- suppressWarnings(do.call(epim, args))

  expect_s3_class(fm, "epimodel")
  expect_equal(length(fm$obs), 2)
  expect_equal(all_obs_types(fm), c("deaths", "cases"))

  nms <- colnames(as.matrix(fm))
  # each series gets its own intercept and its own dispersion parameter; if the
  # series were being collapsed into one, these would not both be present
  expect_true(all(c("deaths|(Intercept)", "cases|(Intercept)") %in% nms))
  expect_true(all(c("deaths|reciprocal dispersion",
                    "cases|reciprocal dispersion") %in% nms))
})

test_that("posterior_predict returns each series separately", {
  skip_on_cran()
  skip_if_no_cmdstan()

  args <- europe_multiobs_args(groups)
  fm <- suppressWarnings(do.call(epim, args))
  nobs <- nrow(args$data)

  for (type in c("deaths", "cases")) {
    pp <- posterior_predict(fm, types = type)
    expect_named(pp, c("group", "time", "draws"))
    expect_equal(dim(pp$draws), c(posterior_sample_size(fm), nobs), info = type)
    expect_equal(length(pp$group), nobs, info = type)
    expect_setequal(as.character(unique(pp$group)), groups)
    expect_true(all(is.finite(pp$draws)), info = type)
  }

  # the two series are different data, so their predictions must differ
  expect_false(identical(
    posterior_predict(fm, types = "deaths", seed = 1)$draws,
    posterior_predict(fm, types = "cases", seed = 1)$draws
  ))
})

test_that("asking for an unmodelled series is an error that names the options", {
  skip_on_cran()
  skip_if_no_cmdstan()

  fm <- suppressWarnings(do.call(epim, europe_multiobs_args(groups)))
  expect_error(posterior_predict(fm, types = "hospitalisations"),
               regexp = "not a modeled type of observation")
  # the message should say what IS modelled
  expect_error(posterior_predict(fm, types = "hospitalisations"),
               regexp = "deaths, cases")

  # it used to end in the literal string "FALSE", because call. = FALSE had been
  # placed inside paste0() rather than stop()
  msg <- tryCatch(posterior_predict(fm, types = "hospitalisations"),
                  error = conditionMessage)
  expect_false(grepl("FALSE", msg, fixed = TRUE))
})

test_that("posterior_predict with no `types` returns every series", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # `types` defaults to all modelled types, and the return was `out[[types]]`.
  # With two series that is recursive indexing -- out[["deaths"]][["cases"]] --
  # so the documented default call silently returned NULL.
  fm <- suppressWarnings(do.call(epim, europe_multiobs_args(groups)))

  all <- posterior_predict(fm)
  expect_false(is.null(all))
  expect_named(all, c("deaths", "cases"))
  for (type in c("deaths", "cases")) {
    expect_named(all[[type]], c("group", "time", "draws"), info = type)
  }

  # a single type still unwraps, which is what plot_obs() and
  # evaluate_forecast() consume
  one <- posterior_predict(fm, types = "deaths")
  expect_named(one, c("group", "time", "draws"))
})

test_that("evaluate_forecast scores each series against its own observations", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # `w <- which(type %in% alltypes)` was which() of a length-1 logical, so it
  # was 1 for any modelled type and `y` always came from object$obs[[1]].
  # Asking for "cases" therefore compared case predictions to death counts.
  args <- europe_multiobs_args(groups)
  fm <- suppressWarnings(do.call(epim, args))

  err <- vapply(c("deaths", "cases"), function(ty) {
    mean(evaluate_forecast(fm, type = ty,
                           metrics = "mean_abs_error")$error$mean_abs_error)
  }, numeric(1))

  # Cases outnumber deaths by roughly two orders of magnitude here, so the two
  # error series cannot be interchangeable. Under the bug they were identical.
  expect_false(isTRUE(all.equal(err[["deaths"]], err[["cases"]])))

  # Each series' error must be on the scale of that series' observations.
  for (ty in c("deaths", "cases")) {
    obs_mean <- mean(args$data[[ty]], na.rm = TRUE)
    expect_lt(err[[ty]], obs_mean * 5)
  }
  expect_gt(err[["cases"]], err[["deaths"]])
})

test_that("both series forecast past the fitting window", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_unless_slow_tests()

  args <- europe_multiobs_args(groups)
  fm <- suppressWarnings(do.call(epim, args))

  newdata <- europe_data(groups, end = as.Date("2020-06-01"))
  expect_gt(nrow(newdata), nrow(args$data))

  for (type in c("deaths", "cases")) {
    insample <- posterior_predict(fm, types = type)
    forecast <- posterior_predict(fm, types = type, newdata = newdata)

    expect_equal(ncol(forecast$draws), nrow(newdata), info = type)
    expect_gt(ncol(forecast$draws), ncol(insample$draws))
    expect_gt(max(forecast$time), max(insample$time))
    expect_true(all(is.finite(forecast$draws)), info = type)
  }
})
