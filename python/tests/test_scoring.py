"""Forecast scoring: CRPS, absolute error, credible-interval coverage."""

import numpy as np
import pandas as pd
import pytest

from epidemia.scoring import (
    METRICS,
    ForecastEvaluation,
    absolute_error,
    coverage,
    crps,
    crps_sample,
    drop_placeholders,
    evaluate_forecast,
    posterior_coverage,
    posterior_metrics,
)


# --------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------


def test_point_mass_crps_is_the_absolute_error():
    """A degenerate predictive at x scores |y - x| -- the definition's base case."""
    y = np.array([0.0, 3.0, -2.5, 7.0])
    x = np.array([1.0, 3.0, 4.0, 0.0])
    draws = np.repeat(x[:, None], 25, axis=1)
    np.testing.assert_allclose(crps(y, draws), np.abs(y - x))


def test_point_mass_crps_at_zero_observations():
    """The pooled-draws bug showed up as a non-zero score on days with y == 0."""
    y = np.zeros(5)
    draws = np.zeros((5, 40))
    np.testing.assert_allclose(crps(y, draws), 0.0, atol=1e-12)


def test_crps_is_row_independent():
    """Row i must be scored against row i's draws only, never the pooled matrix."""
    rng = np.random.default_rng(0)
    y0, d0 = 4.0, rng.normal(4.0, 1.0, size=200)

    alone = crps(np.array([y0]), d0[None, :])
    # Other rows on a wildly different scale: pooling would swamp row 0.
    others = rng.normal(5000.0, 300.0, size=(9, 200))
    y = np.concatenate([[y0], rng.normal(5000.0, 300.0, size=9)])
    together = crps(y, np.vstack([d0[None, :], others]))

    np.testing.assert_allclose(together[0], alone[0], rtol=0, atol=0)


def test_crps_matches_the_single_observation_estimator():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(6, 300))
    y = rng.normal(size=6)
    expected = [crps_sample(y[i], draws[i]) for i in range(6)]
    np.testing.assert_allclose(crps(y, draws), expected)


def test_crps_is_non_negative_and_minimised_at_the_truth():
    rng = np.random.default_rng(2)
    truth = 10.0
    draws = rng.normal(truth, 2.0, size=(1, 4000))
    grid = np.linspace(truth - 8, truth + 8, 81)
    scores = crps(grid, np.repeat(draws, grid.size, axis=0))

    assert np.all(scores >= 0)
    assert abs(grid[np.argmin(scores)] - truth) < 0.5
    # ... and it must actually grow as you move away from the predictive centre
    assert scores[0] > scores[grid.size // 2] < scores[-1]


def test_crps_matches_the_closed_form_for_a_normal_predictive():
    """CRPS(N(mu, sigma), y) = sigma * (z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))."""
    from scipy.stats import norm

    rng = np.random.default_rng(3)
    mu, sigma = 2.0, 1.5
    ys = np.array([2.0, 3.5, 0.0])
    draws = rng.normal(mu, sigma, size=(ys.size, 200_000))
    z = (ys - mu) / sigma
    closed = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    np.testing.assert_allclose(crps(ys, draws), closed, atol=0.02)


def test_crps_prefers_the_correct_predictive():
    """Propriety, empirically: the data-generating predictive wins on average."""
    rng = np.random.default_rng(4)
    n = 400
    y = rng.normal(0.0, 1.0, size=n)
    good = rng.normal(0.0, 1.0, size=(n, 500))
    biased = rng.normal(2.0, 1.0, size=(n, 500))
    overdispersed = rng.normal(0.0, 4.0, size=(n, 500))
    assert crps(y, good).mean() < crps(y, biased).mean()
    assert crps(y, good).mean() < crps(y, overdispersed).mean()


def test_crps_rejects_a_transposed_draw_matrix():
    y = np.zeros(5)
    draws = np.zeros((100, 5))  # (draw, observation) -- the usual mistake
    with pytest.raises(ValueError, match="transposed"):
        crps(y, draws)


# --------------------------------------------------------------------------
# Absolute error
# --------------------------------------------------------------------------


def test_absolute_error_is_taken_over_each_observations_draws():
    y = np.array([0.0, 10.0])
    draws = np.array([[1.0, 3.0, 5.0, 7.0], [10.0, 12.0, 6.0, 10.0]])
    out = absolute_error(y, draws)
    np.testing.assert_allclose(out["mean_abs_error"], [4.0, 1.5])
    np.testing.assert_allclose(out["median_abs_error"], [4.0, 1.0])
    assert set(out) == {"mean_abs_error", "median_abs_error"}


def test_absolute_error_of_a_point_mass_predictive():
    y = np.array([2.0, -1.0])
    draws = np.repeat(np.array([[5.0], [3.0]]), 8, axis=1)
    out = absolute_error(y, draws)
    np.testing.assert_allclose(out["mean_abs_error"], [3.0, 4.0])
    np.testing.assert_allclose(out["median_abs_error"], [3.0, 4.0])


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_coverage_table_shape_and_columns():
    rng = np.random.default_rng(5)
    n, s = 7, 300
    y = rng.normal(size=n)
    draws = rng.normal(size=(n, s))
    cov = coverage(y, draws, levels=(50, 95))
    assert isinstance(cov, pd.DataFrame)
    assert list(cov.columns) == [
        "group", "date", "level", "tag", "lower", "upper", "in_ci"
    ]
    assert len(cov) == n * 2                      # one row per (observation, level)
    assert cov["in_ci"].dtype == bool
    assert sorted(cov["level"].unique()) == [50.0, 95.0]
    assert set(cov["tag"]) == {"50% CI", "95% CI"}
    assert (cov["lower"] <= cov["upper"]).all()


def test_coverage_bounds_are_the_central_quantiles():
    y = np.zeros(1)
    draws = np.arange(1001, dtype=float)[None, :]
    cov = coverage(y, draws, levels=(90,))
    np.testing.assert_allclose(cov["lower"].to_numpy(), np.percentile(draws, 5))
    np.testing.assert_allclose(cov["upper"].to_numpy(), np.percentile(draws, 95))
    assert not cov["in_ci"].iloc[0]  # y = 0 sits below the 5th percentile


def test_calibrated_predictive_gives_nominal_coverage():
    rng = np.random.default_rng(6)
    n, s = 3000, 1000
    mu = rng.normal(0.0, 5.0, size=n)          # a different predictive per day
    draws = rng.normal(mu[:, None], 1.0, size=(n, s))
    y = rng.normal(mu, 1.0)
    cov = coverage(y, draws, levels=(50, 95))
    for level, tol in ((50.0, 0.03), (95.0, 0.015)):
        emp = cov.loc[cov["level"] == level, "in_ci"].mean()
        assert abs(emp - level / 100) < tol, f"{level}% CI covered {emp:.3f}"


def test_a_too_narrow_predictive_undercovers():
    rng = np.random.default_rng(7)
    n = 2000
    y = rng.normal(0.0, 1.0, size=n)
    draws = rng.normal(0.0, 0.1, size=(n, 500))  # far too confident
    emp = coverage(y, draws, levels=(95,))["in_ci"].mean()
    assert emp < 0.3


def test_levels_are_validated_and_sorted():
    y, draws = np.zeros(3), np.zeros((3, 10))
    cov = coverage(y, draws, levels=(95, 50))
    assert list(cov["level"].unique()) == [50.0, 95.0]  # ascending, as in R
    with pytest.raises(ValueError, match="between 0"):
        coverage(y, draws, levels=(50, 150))
    with pytest.warns(UserWarning, match="no levels"):
        assert sorted(coverage(y, draws, levels=())["level"].unique()) == [50.0, 95.0]


# --------------------------------------------------------------------------
# Placeholders
# --------------------------------------------------------------------------


def test_placeholder_rows_are_dropped_with_their_draws():
    y = np.array([3.0, -1.0, np.nan, 8.0])
    draws = np.arange(4 * 5, dtype=float).reshape(4, 5)
    ky, kd, kg, kdate = drop_placeholders(y, draws, group=list("abcd"),
                                          date=[0, 1, 2, 3])
    np.testing.assert_allclose(ky, [3.0, 8.0])
    np.testing.assert_allclose(kd, draws[[0, 3]])
    assert list(kg) == ["a", "d"]
    assert list(kdate) == [0, 3]


def test_evaluate_forecast_drops_placeholders():
    rng = np.random.default_rng(8)
    y = np.array([3.0, -1.0, np.nan, 8.0, 0.0])
    draws = rng.normal(5.0, 2.0, size=(5, 400))
    date = pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-03",
                           "2020-03-04", "2020-03-05"])
    out = evaluate_forecast(y, draws, date=date, levels=(50, 95))

    assert len(out.error) == 3                       # -1 and NaN gone
    assert list(out.error["date"]) == list(date[[0, 3, 4]])
    assert len(out.coverage) == 3 * 2
    # ... and the survivors are scored against their OWN (unshifted) draws
    keep = [0, 3, 4]
    np.testing.assert_allclose(out.error["crps"].to_numpy(), crps(y[keep], draws[keep]))
    assert np.isfinite(out.error[list(METRICS)].to_numpy()).all()


def test_placeholder_rows_do_not_inflate_the_score():
    """The whole point of dropping: a -1 truth would be scored as a count."""
    rng = np.random.default_rng(9)
    y = np.array([5.0, 5.0])
    draws = rng.normal(5.0, 1.0, size=(2, 2000))
    clean = evaluate_forecast(y, draws).error["crps"].to_numpy()
    with_ph = evaluate_forecast(np.array([5.0, 5.0, -1.0]),
                                np.vstack([draws, draws[:1]])).error["crps"].to_numpy()
    np.testing.assert_allclose(with_ph, clean)
    assert with_ph.mean() < 1.0  # a scored -1 would push this to ~6


# --------------------------------------------------------------------------
# evaluate_forecast tables
# --------------------------------------------------------------------------


def test_evaluate_forecast_table_shapes_and_columns():
    rng = np.random.default_rng(10)
    n, s = 6, 250
    y = rng.poisson(20, size=n).astype(float)
    draws = rng.poisson(20, size=(n, s)).astype(float)
    group = ["Italy"] * 3 + ["France"] * 3
    date = pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-03"] * 2)

    out = evaluate_forecast(y, draws, group=group, date=date, levels=(50, 95))
    assert isinstance(out, ForecastEvaluation)
    assert list(out.error.columns) == ["group", "date", "crps",
                                       "mean_abs_error", "median_abs_error"]
    assert len(out.error) == n
    assert list(out.coverage.columns) == ["group", "date", "level", "tag",
                                          "lower", "upper", "in_ci"]
    assert len(out.coverage) == n * 2
    assert set(out.coverage["group"]) == {"Italy", "France"}
    # R returns a named list; ["error"] must work as well as .error
    assert out["error"] is out.error
    assert out["coverage"] is out.coverage
    with pytest.raises(KeyError):
        out["nope"]


def test_metrics_subset_and_validation():
    y, draws = np.zeros(4), np.zeros((4, 10))
    err = evaluate_forecast(y, draws, metrics="crps").error
    assert list(err.columns) == ["group", "date", "crps"]
    err = evaluate_forecast(y, draws, metrics=["median_abs_error", "crps"]).error
    # column order follows R's daily_error, not the order the caller asked in
    assert list(err.columns) == ["group", "date", "crps", "median_abs_error"]
    with pytest.raises(ValueError, match="unrecognised metrics"):
        evaluate_forecast(y, draws, metrics="rmse")


def test_default_labels_when_no_group_or_date():
    rng = np.random.default_rng(11)
    y = rng.poisson(4, size=4).astype(float)
    draws = rng.poisson(4, size=(4, 50)).astype(float)
    out = evaluate_forecast(y, draws)
    assert set(out.error["group"]) == {"all"}
    assert list(out.error["date"]) == [0, 1, 2, 3]


def test_any_negative_outcome_counts_as_a_placeholder():
    """Mirrors R's `!is.na(y) & y >= 0`: outcomes are counts, so a negative is a code."""
    y = np.array([2.0, -1.0, -7.0])
    draws = np.zeros((3, 10))
    assert len(evaluate_forecast(y, draws).error) == 1


def test_posterior_metrics_and_coverage_wrappers():
    rng = np.random.default_rng(12)
    y = rng.poisson(9, size=5).astype(float)
    draws = rng.poisson(9, size=(5, 100)).astype(float)
    out = evaluate_forecast(y, draws)
    pd.testing.assert_frame_equal(posterior_metrics(y, draws), out.error)
    pd.testing.assert_frame_equal(posterior_coverage(y, draws), out.coverage)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="one row per observation"):
        crps(np.zeros(3), np.zeros((4, 10)))
    with pytest.raises(ValueError, match="group has length"):
        evaluate_forecast(np.zeros(3), np.zeros((3, 10)), group=["a", "b"])
