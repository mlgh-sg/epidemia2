#!/usr/bin/env Rscript
#
# Regenerate the fitted-model fixtures used by the test suite.
#
#     Rscript tests/data/make-fixtures.R            # all
#     Rscript tests/data/make-fixtures.R fm-uk      # a subset
#
# These fixtures are serialised `epimodel` objects, so they are coupled to the
# internal draws representation. When that changes they stop working -- which is
# exactly what happened during the CmdStanR migration: `fm-uk.rds` and
# `forecast-test-data.rds` were left holding rstan S4 `stanfit` objects, and
# `test-newdata.R` failed with "$ operator not defined for this S4 class".
#
# Run this after any change to `build_draws()` / the `epimodel_draws` wrapper,
# then commit the regenerated .rds files.
#
# Requires CmdStan (make setup).

args <- commandArgs(trailingOnly = TRUE)
all_fixtures <- c("fm-uk", "plot_test_fit", "forecast-test-data")
want <- if (length(args)) intersect(args, all_fixtures) else all_fixtures
if (!length(want)) {
  stop("Unknown fixture. Choose from: ", paste(all_fixtures, collapse = ", "))
}

root <- tryCatch(
  rprojroot::find_root(rprojroot::is_r_package),
  error = function(e) normalizePath(".")
)
suppressWarnings(suppressMessages(pkgload::load_all(root, quiet = TRUE)))

out_dir <- file.path(root, "tests", "data")

# Shared data: EuropeCovid2 filtered the same way the multilevel tutorial does --
# seeding starts 30 days before the 10th cumulative death, and the fitting window
# ends on the 5th of May.
data("EuropeCovid2", package = "epidemia")
data("EuropeCovid", package = "epidemia")

europe <- EuropeCovid2$data
europe <- europe[europe$date > europe$date[which(cumsum(europe$deaths) > 10)[1] - 30], ]
fit_window <- europe[europe$date < as.Date("2020-05-05"), ]
fit_window$week <- format(fit_window$date, "%V")

deaths_obs <- suppressWarnings(epiobs(
  formula = deaths ~ 1,
  i2o = EuropeCovid2$inf2death,
  prior_intercept = normal(0, 0.2),
  link = scaled_logit(0.02)
))
basic_inf <- epiinf(gen = EuropeCovid$si, seed_days = 6)

# ---------------------------------------------------------------------------
# fm-uk: single group, random walk on R_t. Used by test-newdata.R to check that
# passing `newdata` equal to the training data reproduces predictions exactly.
if ("fm-uk" %in% want) {
  message("== fm-uk ==")
  fm <- epim(
    rt = epirt(formula = R(country, date) ~ 1 + rw(time = week)),
    inf = basic_inf,
    obs = deaths_obs,
    data = fit_window,
    group_subset = "United_Kingdom",
    algorithm = "sampling",
    iter = 1000, chains = 2, seed = 12345, refresh = 0
  )
  saveRDS(list(data = fit_window, fm = fm),
          file.path(out_dir, "fm-uk.rds"))
  message("   wrote fm-uk.rds")
}

# ---------------------------------------------------------------------------
# plot_test_fit: two groups with a covariate. Used by test-plotting.R.
if ("plot_test_fit" %in% want) {
  message("== plot_test_fit ==")
  fm <- epim(
    rt = epirt(formula = R(country, date) ~ 1 + lockdown),
    inf = basic_inf,
    obs = deaths_obs,
    data = fit_window,
    group_subset = c("Austria", "Germany"),
    algorithm = "sampling",
    iter = 1000, chains = 2, seed = 12345, refresh = 0
  )
  saveRDS(fm, file.path(out_dir, "plot_test_fit.rds"))
  message("   wrote plot_test_fit.rds")
}

# ---------------------------------------------------------------------------
# forecast-test-data: three groups fitted to a short window, plus a `newdata`
# frame that extends past it. Used by test-forecasts.R to exercise
# evaluate_forecast() / posterior_coverage() / posterior_metrics() against
# genuinely out-of-sample observations.
if ("forecast-test-data" %in% want) {
  message("== forecast-test-data ==")
  groups <- c("Austria", "Germany", "Italy")
  short <- fit_window[fit_window$date < as.Date("2020-04-15"), ]
  newdata <- europe[europe$date < as.Date("2020-05-05"), ]
  newdata$week <- format(newdata$date, "%V")

  fm <- epim(
    rt = epirt(formula = R(country, date) ~ 1 + lockdown),
    inf = basic_inf,
    obs = deaths_obs,
    data = short,
    group_subset = groups,
    algorithm = "sampling",
    iter = 1000, chains = 2, seed = 12345, refresh = 0
  )
  saveRDS(list(fit = fm, newdata = newdata[newdata$country %in% groups, ]),
          file.path(out_dir, "forecast-test-data.rds"))
  message("   wrote forecast-test-data.rds")
}

message("Done. Commit the regenerated .rds files.")
