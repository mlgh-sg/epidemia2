
test_that("Wrong LHS formula specifications are caught", {
expect_error(rt <- epirt(formula = R(x, y) ~ 1 + cov), NA)
expect_error(rt <- epirt(formula = R() ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(x) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(,y) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(x,) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(x+y,z) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(x,y,z) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(x/y, z) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(x,y) + a ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = a + R(x,y) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = r(x,y) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = Rt(x,y) ~ 1 + cov), regexp = "left hand side")
expect_error(rt <- epirt(formula = R(1,y) ~ 1 + cov), regexp = "left hand side")
})

test_that("Wrong class for formula is caught", {
  expect_error(rt <- epirt(formula = "dummy"), regexp = "must have class")
})

form <- R(x,y) ~ 1 + cov

test_that("link handled correctly", {
  expect_error(rt <- epirt(formula = form, link = "identity"), NA)
  expect_error(rt <- epirt(formula = form, link = scaled_logit(5)), NA)
  expect_error(rt <- epirt(formula = form, link = "dummy"), regexp = "must be either")
  expect_error(rt <- epirt(formula = form, link = 1), regexp = "must be either")
})

test_that("center handled correctly", {
  expect_error(rt <- epirt(formula = form, center = TRUE), NA)
  expect_error(rt <- epirt(formula = form, center = 1), regexp = "logical")
  expect_error(rt <- epirt(formula = form, center = c(TRUE,TRUE)), regexp = "scalar")
})

test_that("handling of prior argument", {
  expect_error(rt <- epirt(formula = form, prior = cauchy()), NA)
  expect_error(rt <- epirt(formula = form, prior = "dummy"), regexp = "prior function")
  # a named list is not enough: check_prior() also requires a `dist` element,
  # which is what distinguishes a prior from any other list the user might pass.
  expect_error(rt <- epirt(formula = form, prior = list(location = 0, scale = 1)),
               regexp = "prior function")
})

test_that("prior_intercept ok dists", {
  expect_error(rt <- epirt(formula = form, prior_intercept = cauchy()), NA)
  expect_error(rt <- epirt(formula = form, prior_intercept = lasso()), "must be one of")
})

test_that("prior_covariance ok dists", {
  expect_error(rt <- epirt(formula = form, prior_covariance = decov()), NA)
  expect_error(rt <- epirt(formula = form, prior_covariance = normal()), "must be one of")
})

test_that("prior_covariance = lkj() is rejected up front", {
  # The Stan program has no lkj branch. Before this check, lkj() was accepted
  # here and then produced standata with shape = 0, so the fit died inside
  # CmdStan with "Unable to retrieve the metadata" only after compiling.
  expect_error(epirt(formula = form, prior_covariance = lkj()),
               regexp = "not supported")
  expect_error(epirt(formula = form, prior_covariance = lkj()),
               regexp = "decov")
})


test_that("correctly storing additional arguments", {
  rt <- epirt(formula=form)
  expect_equal(length(rt$mfargs), 0)
  rt <- epirt(formula = form, na.action = na.fail)
  expect_equal(length(rt$mfargs), 1)
  expect_true(all.equal(rt$mfargs$na.action, na.fail))
})













