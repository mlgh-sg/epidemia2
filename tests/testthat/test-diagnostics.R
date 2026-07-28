test_that("sampler_diagnostics rejects things that are not fitted models", {
  expect_error(sampler_diagnostics(list()), "must be a fitted model")
  expect_error(sampler_diagnostics(1:10), "must be a fitted model")
})

# Build an epidemia_diagnostics object directly, so the reporting logic can be
# tested without paying for a fit. The fields are exactly what
# collect_diagnostics() stores.
fake_diag <- function(divergent = c(0, 0), max_treedepth = c(0, 0),
                      ebfmi = c(1.0, 1.0), iter_sampling = 500,
                      worst_rhat = 1.001, min_ess_bulk = 1000,
                      min_ess_tail = 900) {
  structure(
    list(
      per_chain = data.frame(
        chain = seq_along(divergent),
        divergent = as.integer(divergent),
        max_treedepth = as.integer(max_treedepth),
        ebfmi = ebfmi
      ),
      iter_sampling = iter_sampling,
      worst_rhat = worst_rhat,
      worst_rhat_par = "R|lockdown",
      min_ess_bulk = min_ess_bulk,
      min_ess_bulk_par = "R|lockdown",
      min_ess_tail = min_ess_tail
    ),
    class = "epidemia_diagnostics"
  )
}

# print() wraps long warnings with strwrap(), so collapse and squish the
# whitespace before matching -- otherwise a phrase broken across two lines
# picks up the indent and the regex misses it.
diag_text <- function(d) {
  gsub("\\s+", " ", paste(capture.output(print(d)), collapse = " "))
}

test_that("a clean fit reports no problems", {
  out <- capture.output(print(fake_diag()))
  expect_true(any(grepl("No problems detected", out)))
  expect_true(any(grepl("Divergent transitions: 0", out)))
})

test_that("divergences are reported with the right advice", {
  out <- diag_text(fake_diag(divergent = c(3, 1)))
  expect_true(grepl("Divergent transitions: 4", out))
  # a divergence biases the posterior, so the advice must never be "draw more"
  expect_true(grepl("more draws will not help", out))
  expect_true(grepl("adapt_delta", out))
  expect_false(grepl("No problems detected", out))
})

test_that("max treedepth is described as efficiency, not correctness", {
  out <- diag_text(fake_diag(max_treedepth = c(10, 8)))
  expect_true(grepl("Hit max treedepth: 18", out))
  expect_true(grepl("efficiency rather than correctness", out))
})

test_that("low E-BFMI is flagged", {
  out <- diag_text(fake_diag(ebfmi = c(0.11, 0.9)))
  expect_true(grepl("E-BFMI of 0.11 is below 0.2", out))
})

test_that("bad R-hat and low ESS are flagged", {
  out <- diag_text(fake_diag(worst_rhat = 1.27, min_ess_bulk = 40))
  expect_true(grepl("R-hat of 1.270", out))
  expect_true(grepl("have not mixed", out))
  expect_true(grepl("Bulk ESS of 40", out))
})

test_that("percentages are computed against the total post-warmup draws", {
  out <- diag_text(fake_diag(divergent = c(5, 5), iter_sampling = 500))
  # 10 divergences out of 2 chains x 500 draws = 1.0%
  expect_true(grepl("Divergent transitions: 10 \\(1.0%\\)", out))
})

test_that("sampler_diagnostics is reported for a real fit", {
  skip_on_cran()
  skip_if_no_cmdstan()

  fm <- do.call(epim, europe_args(iter = 200, chains = 2))

  d <- sampler_diagnostics(fm)
  expect_s3_class(d, "epidemia_diagnostics")
  expect_equal(nrow(d$per_chain), 2)
  expect_named(d$per_chain,
               c("chain", "divergent", "max_treedepth", "ebfmi"))
  expect_true(all(d$per_chain$divergent >= 0))
  expect_true(is.finite(d$worst_rhat))
  expect_true(is.finite(d$min_ess_bulk))

  # the whole point: the diagnostics survive a save/load round trip, which is
  # what the CmdStan console output does not
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  saveRDS(fm, f)
  expect_equal(sampler_diagnostics(readRDS(f))$per_chain, d$per_chain)
})

test_that("variational fits say why they have no diagnostics", {
  skip_on_cran()
  skip_if_no_cmdstan()

  args <- europe_args()
  args$algorithm <- "meanfield"
  args$iter <- 1e4
  args$chains <- NULL
  fm <- suppressWarnings(do.call(epim, args))

  expect_message(res <- sampler_diagnostics(fm), "not a Hamiltonian sampler")
  expect_null(res)
})

test_that("a fit saved before diagnostics were retained says so accurately", {
  # Older saved models have no $diagnostics. Reporting that as "sampling does
  # not produce them" would be plainly false, so the two absences differ.
  fake <- structure(
    list(algorithm = "sampling",
         stanfit = structure(list(diagnostics = NULL), class = "epimodel_draws")),
    class = "epimodel"
  )
  expect_message(res <- sampler_diagnostics(fake), "fitted before epidemia retained them")
  expect_null(res)
})

test_that("print.epimodel stays silent about a clean fit", {
  x <- structure(list(stanfit = structure(
    list(diagnostics = list(per_chain = data.frame(
      chain = 1:2, divergent = c(0L, 0L),
      max_treedepth = c(0L, 0L), ebfmi = c(0.9, 1.0)))),
    class = "epimodel_draws")), class = "epimodel")
  expect_silent(print_diagnostic_footer(x))
})

test_that("print.epimodel flags a fit the sampler struggled with", {
  x <- structure(list(stanfit = structure(
    list(diagnostics = list(per_chain = data.frame(
      chain = 1:2, divergent = c(3L, 1L),
      max_treedepth = c(0L, 7L), ebfmi = c(0.9, 0.1)))),
    class = "epimodel_draws")), class = "epimodel")
  out <- paste(capture.output(print_diagnostic_footer(x)), collapse = " ")
  expect_true(grepl("4 divergent transitions", out))
  expect_true(grepl("7 iterations at max treedepth", out))
  expect_true(grepl("low E-BFMI", out))
  expect_true(grepl("sampler_diagnostics", out))
})
