# Shared test helpers.
#
# testthat sources helper-*.R before running any test file.

# cmdstanr is a Suggests dependency, and CmdStan itself is a separate C++
# toolchain that may be absent even when cmdstanr is installed. Without this the
# fitting tests fail hard rather than skipping, which breaks lightweight CI and
# violates CRAN's rule that Suggests packages be used conditionally.
skip_if_no_cmdstan <- function() {
  testthat::skip_if_not_installed("cmdstanr")
  path <- tryCatch(cmdstanr::cmdstan_path(), error = function(e) NULL)
  if (is.null(path)) {
    testthat::skip("No CmdStan installation found (cmdstanr::install_cmdstan()).")
  }
  invisible(TRUE)
}

# Gate for fits that are too slow for the default loop. Set EPIDEMIA_SLOW_TESTS
# to run them; `make test-slow` and the R-CMD-check workflow both do.
slow_tests_enabled <- function() {
  isTRUE(as.logical(Sys.getenv("EPIDEMIA_SLOW_TESTS", "false")))
}

skip_unless_slow_tests <- function() {
  if (!slow_tests_enabled()) {
    testthat::skip("Slow test; set EPIDEMIA_SLOW_TESTS=true to run.")
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Fixture builders.
#
# Each returns a fresh list of `epim()` arguments so a test can override one
# piece without disturbing its neighbours. Sampler settings are deliberately
# small: these tests check structure (parameter names, dimensions, the shape of
# forecasts), not inference quality.

# EuropeCovid2 filtered the way the multilevel tutorial filters it: seeding
# starts 30 days before the 10th cumulative death, fitting stops on 5 May.
europe_data <- function(groups = NULL, end = as.Date("2020-05-05")) {
  utils::data("EuropeCovid2", package = "epidemia", envir = environment())
  d <- EuropeCovid2$data
  d <- d[d$date > d$date[which(cumsum(d$deaths) > 10)[1] - 30], ]
  d <- d[d$date < end, ]
  if (!is.null(groups)) d <- d[d$country %in% groups, ]
  d$week <- format(d$date, "%V")
  as.data.frame(d)
}

# A single-observation (deaths) argument list over `groups`.
europe_args <- function(groups = c("Austria", "Germany"),
                        iter = 200, chains = 1) {
  utils::data("EuropeCovid2", package = "epidemia", envir = environment())
  utils::data("EuropeCovid", package = "epidemia", envir = environment())
  list(
    rt = epirt(formula = R(country, date) ~ 1 + lockdown),
    inf = epiinf(gen = EuropeCovid$si, seed_days = 6),
    obs = suppressWarnings(epiobs(
      formula = deaths ~ 1,
      i2o = EuropeCovid2$inf2death,
      prior_intercept = normal(0, 0.2),
      link = scaled_logit(0.02)
    )),
    data = europe_data(groups),
    group_subset = groups,
    algorithm = "sampling",
    iter = iter, chains = chains, seed = 12345, refresh = 0
  )
}

# Two jointly-modelled series (deaths + cases) over `groups`. The `cases` i2o is
# deliberately short and crude -- these tests are about the multi-series
# plumbing, not about a defensible ascertainment model.
europe_multiobs_args <- function(groups = c("Austria", "Germany"),
                                 iter = 200, chains = 1) {
  args <- europe_args(groups, iter = iter, chains = chains)
  utils::data("EuropeCovid2", package = "epidemia", envir = environment())
  cases <- suppressWarnings(epiobs(
    formula = cases ~ 1,
    i2o = c(rep(0, 3), rep(1 / 7, 7)),
    prior_intercept = normal(0, 0.5),
    link = scaled_logit(0.5)
  ))
  # unnamed, the way the tutorials write it (`obs = list(cases, ons)`) -- naming
  # the list makes all_obs_types() return a named vector, which users would not see
  args$obs <- list(args$obs, cases)
  args
}

# Number of posterior draws behind a fitted epimodel.
n_draws <- function(fm) posterior_sample_size(fm)
