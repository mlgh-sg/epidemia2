data("EuropeCovid")
args <- list()
args$data = EuropeCovid$data
args$inf <- epiinf(gen = EuropeCovid$si)
args$rt <- epirt(R(country, date) ~ 1 + lockdown)
expect_warning(args$obs <- epiobs(deaths~1, i2o = EuropeCovid$inf2death * 0.02))
args$group_subset <- c("Germany", "Italy")
args <- c(args, list(iter=10,chains=1, seed=12345))
args$refresh <- 0

test_that("epim runs through with various rt formula", {
  skip_on_cran()
  skip_if_no_cmdstan()
  run_args <- args

  # just fixed effects
  fm <- suppressWarnings(do.call(epim, run_args))
  expect_true(inherits(fm, "epimodel"))

  # random effects
  run_args$rt <- epirt(formula = R(country, date) ~ (lockdown | country))
  fm <- suppressWarnings(do.call(epim, run_args))
  expect_true(inherits(fm, "epimodel"))

  # random walks
  run_args$data$week <- format(run_args$data$date,"%V")
  run_args$rt <- epirt(formula = R(country, date) ~ (lockdown | country) + rw(time=week) + rw(time=week, gr=country))
  fm <- suppressWarnings(do.call(epim, run_args))
  expect_true(inherits(fm, "epimodel"))

})

test_that("epim accepts an explicit sampling algorithm", {
  skip_on_cran()
  skip_if_no_cmdstan()
  run_args <- args
  run_args$algorithm <- "sampling"
  fm <- suppressWarnings(do.call(epim, run_args))
  expect_true(inherits(fm, "epimodel"))
  expect_equal(fm$algorithm, "sampling")
})

# The variational branch of fit_cmdstan() reaches $variational() rather than
# $sample(), and translate_vb_args() maps a different set of arguments. The
# multilevel tutorial relies on it, so it needs at least a smoke test.
test_that("epim runs through with the variational algorithms", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_unless_slow_tests()

  for (alg in c("meanfield", "fullrank")) {
    run_args <- args
    run_args$algorithm <- alg
    run_args$iter <- 1e4
    fm <- suppressWarnings(do.call(epim, run_args))
    expect_true(inherits(fm, "epimodel"), info = alg)
    expect_equal(fm$algorithm, alg)
    expect_gt(posterior_sample_size(fm), 0)
  }
})








