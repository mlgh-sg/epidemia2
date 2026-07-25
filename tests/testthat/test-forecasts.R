# Forecast evaluation: evaluate_forecast(), posterior_metrics(),
# posterior_coverage(), plot_metrics(), plot_coverage().
#
# This file was previously commented out in its entirety, so all five exported
# functions had no coverage at all. It could not simply be uncommented: those
# tests were written against an older signature that took an `observation=`
# data frame, which no longer exists -- the observations now come from
# `newdata`. These are rewritten against the current API.
#
# Leaving it dark had a cost: plot_metrics() was broken outright
# (facet_wrap(.data$group, ...) cannot be used outside a data mask) and nothing
# noticed.

forecast_fixture <- function() {
  readRDS("../data/forecast-test-data.rds")
}

test_that("crps scores each observation against its own predictive draws", {
  # Regression test. crps() used to pool the whole [observation, draw] matrix
  # into one empirical distribution and score every observation against that
  # marginal, so the value depended on the other dates and groups in the call.
  crps <- epidemia:::crps

  # A point-mass predictive reduces CRPS to the absolute error.
  draws <- rbind(rep(0, 500), rep(10, 500), rep(10, 500))
  expect_equal(crps(c(0, 10, 60), draws), c(0, 0, 50))

  # Scoring is per row: the result for one observation must not change when
  # unrelated observations are added to the call. Under the pooled version it did.
  alone <- crps(0, draws[1, , drop = FALSE])
  together <- crps(c(0, 10, 60), draws)[1]
  expect_equal(alone, together)

  # Matches the sorted-sample estimator computed independently.
  set.seed(42)
  d <- t(vapply(c(5, 50, 500), function(m) as.numeric(rpois(400, m)), numeric(400)))
  y <- c(5, 50, 500)
  reference <- vapply(seq_along(y), function(i) {
    x <- sort(d[i, ]); n <- length(x)
    a <- seq.int(0.5 / n, 1 - 0.5 / n, length.out = n)
    2 * (1 / n) * sum(((y[i] < x) - a) * (x - y[i]))
  }, numeric(1))
  expect_equal(crps(y, d), reference)

  # CRPS is non-negative and minimised at the truth.
  expect_true(all(crps(y, d) >= 0))
  expect_lt(crps(50, d[2, , drop = FALSE]), crps(500, d[2, , drop = FALSE]))
})

test_that("mean_abs_error is per observation and zero for a perfect forecast", {
  # daily_error() builds `mat` with sweep(t(draws), 1, y); the MARGIN and the
  # orientation of t(draws) have to agree or every date is compared to the
  # wrong observation.
  obs <- list(
    group = factor(c("A", "A", "B")),
    time = as.Date("2020-01-01") + 0:2,
    # draws is [draw, observation]
    draws = cbind(rep(1, 200), rep(5, 200), rep(9, 200))
  )
  out <- epidemia:::daily_error(obs, c("crps", "mean_abs_error", "median_abs_error"),
                                y = c(1, 5, 9))
  expect_equal(out$mean_abs_error, c(0, 0, 0))
  expect_equal(out$median_abs_error, c(0, 0, 0))
  expect_equal(out$crps, c(0, 0, 0))

  out2 <- epidemia:::daily_error(obs, "mean_abs_error", y = c(3, 5, 9))
  expect_equal(out2$mean_abs_error, c(2, 0, 0))
})

test_that("evaluate_forecast returns per-group, per-date error and coverage", {
  skip_on_cran()
  skip_if_no_cmdstan()

  td <- forecast_fixture()
  out <- evaluate_forecast(td$fit, newdata = td$newdata, type = "deaths")

  expect_named(out, c("error", "coverage"))
  expect_s3_class(out$error, "data.frame")
  expect_s3_class(out$coverage, "data.frame")

  expect_true(all(c("group", "date", "crps", "mean_abs_error",
                    "median_abs_error") %in% names(out$error)))
  expect_true(all(c("group", "date", "tag", "in_ci") %in% names(out$coverage)))

  expect_setequal(as.character(unique(out$error$group)), td$fit$groups)

  # one error row per (group, date); coverage repeats that for each level
  expect_equal(nrow(out$error),
               length(unique(out$error$group)) * length(unique(out$error$date)))
  expect_equal(nrow(out$coverage), nrow(out$error) * 2L)

  expect_true(all(out$error$crps >= 0))
  expect_true(all(out$error$mean_abs_error >= 0))
  expect_true(all(out$coverage$in_ci %in% c(0, 1) |
                    is.logical(out$coverage$in_ci)))
})

test_that("metrics and levels arguments are honoured", {
  skip_on_cran()
  skip_if_no_cmdstan()

  td <- forecast_fixture()

  only_crps <- evaluate_forecast(td$fit, newdata = td$newdata, type = "deaths",
                                 metrics = "crps")
  expect_true("crps" %in% names(only_crps$error))
  expect_false("mean_abs_error" %in% names(only_crps$error))

  three <- evaluate_forecast(td$fit, newdata = td$newdata, type = "deaths",
                             levels = c(50, 80, 95))
  expect_equal(length(unique(three$coverage$tag)), 3L)
})

test_that("groups argument restricts the evaluation", {
  skip_on_cran()
  skip_if_no_cmdstan()

  td <- forecast_fixture()
  one <- evaluate_forecast(td$fit, newdata = td$newdata, type = "deaths",
                           groups = "Italy")
  expect_setequal(as.character(unique(one$error$group)), "Italy")
})

test_that("bad type and bad metric are rejected", {
  skip_on_cran()
  skip_if_no_cmdstan()

  td <- forecast_fixture()
  expect_error(
    evaluate_forecast(td$fit, newdata = td$newdata, type = "hospitalisations"),
    regexp = "does not contain any observations"
  )
  expect_error(
    evaluate_forecast(td$fit, newdata = td$newdata, type = "deaths",
                      metrics = "bogus"),
    regexp = "Unrecognised metrics"
  )
})

test_that("rows coded -1 are treated as forecast placeholders, not observations", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # epiobs_() documents -1 as the placeholder for a forecast horizon, and the
  # multiple-observations tutorial builds newdata that way. Scoring those rows
  # took the truth to be -1, inflating the error and collapsing coverage.
  td <- forecast_fixture()
  cutoff <- as.Date("2020-04-15")

  placeheld <- td$newdata
  placeheld$deaths[placeheld$date > cutoff] <- -1

  scored <- evaluate_forecast(td$fit, newdata = placeheld, type = "deaths")

  # nothing after the cutoff should have been scored at all
  expect_true(all(scored$error$date <= cutoff))
  expect_true(all(scored$coverage$date <= cutoff))
  expect_gt(nrow(scored$error), 0)

  # the same rows survive as when those dates are simply not supplied. The
  # metric values themselves are not compared: posterior_predict() samples, and
  # a longer newdata consumes a different stretch of the RNG stream, so two
  # runs give different realisations of the same predictive distribution.
  truncated <- td$newdata[td$newdata$date <= cutoff, ]
  ref <- evaluate_forecast(td$fit, newdata = truncated, type = "deaths")
  expect_equal(nrow(scored$error), nrow(ref$error))
  expect_equal(scored$error$date, ref$error$date)
  expect_equal(as.character(scored$error$group), as.character(ref$error$group))

  expect_true(all(scored$error$mean_abs_error >= 0))
})

test_that("posterior_metrics and posterior_coverage match evaluate_forecast", {
  skip_on_cran()
  skip_if_no_cmdstan()

  td <- forecast_fixture()
  out <- evaluate_forecast(td$fit, newdata = td$newdata, type = "deaths")

  pm <- posterior_metrics(td$fit, newdata = td$newdata, type = "deaths")
  pc <- posterior_coverage(td$fit, newdata = td$newdata, type = "deaths")

  expect_equal(names(pm), names(out$error))
  expect_equal(nrow(pm), nrow(out$error))
  expect_equal(names(pc), names(out$coverage))
  expect_equal(nrow(pc), nrow(out$coverage))
})

test_that("the forecast plots build", {
  skip_on_cran()
  skip_if_no_cmdstan()

  # Regression guard: plot_metrics() and the by_group / by_unseen branches of
  # plot_coverage() all used facet_wrap(.data$col), which errors with
  # "Can't subset `.data` outside of a data mask context".
  td <- forecast_fixture()

  expect_s3_class(plot_metrics(td$fit, newdata = td$newdata, type = "deaths"),
                  "ggplot")
  expect_s3_class(plot_coverage(td$fit, newdata = td$newdata, type = "deaths"),
                  "ggplot")
  expect_s3_class(
    plot_coverage(td$fit, newdata = td$newdata, type = "deaths",
                  by_group = TRUE), "ggplot")
  expect_s3_class(
    plot_coverage(td$fit, newdata = td$newdata, type = "deaths",
                  by_unseen = TRUE), "ggplot")
})
