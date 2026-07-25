"""Forward simulation, the susceptibility adjustment, and predictive sampling."""

import numpy as np
import pytest

from epidemia.predict import expected_observations, posterior_predict, simulate
from epidemia.renewal import expected_observations as ref_expected_observations
from epidemia.renewal import renewal_infections

SEED = 20200311


# --------------------------------------------------------------------------
# simulate: unadjusted recursion
# --------------------------------------------------------------------------


def test_simulate_matches_the_renewal_reference():
    rng = np.random.default_rng(SEED)
    gen = rng.dirichlet(np.ones(8))
    R = np.exp(rng.normal(0.1, 0.2, size=40))
    sim = simulate(R, gen, seed=12.0, seed_days=6, n_days=40)
    ref = renewal_infections(R, np.full(6, 12.0), gen)
    np.testing.assert_allclose(sim, ref, rtol=1e-12)


def test_simulate_vectorises_over_draws():
    """The batched path must give exactly what looping over draws gives."""
    rng = np.random.default_rng(SEED)
    gen = rng.dirichlet(np.ones(5))
    R = np.exp(rng.normal(0.0, 0.3, size=(7, 30)))
    seeds = rng.uniform(5.0, 20.0, size=7)

    sim = simulate(R, gen, seed=seeds, seed_days=4, n_days=30)
    assert sim.shape == (7, 30)
    for d in range(7):
        ref = renewal_infections(R[d], np.full(4, seeds[d]), gen)
        np.testing.assert_allclose(sim[d], ref, rtol=1e-12)


def test_simulate_forecasts_past_the_fitted_window():
    """A longer Rt (fit + horizon) simulates forward with no re-fit."""
    gen = np.array([1.0])
    R = np.full(10, 2.0)
    fitted = simulate(R, gen, seed=1.0, seed_days=1, n_days=5)
    full = simulate(R, gen, seed=1.0, seed_days=1, n_days=10)
    np.testing.assert_allclose(full[:5], fitted)  # forecast extends, not revises
    np.testing.assert_allclose(full, 2.0 ** np.arange(10))


def test_simulate_rejects_too_short_an_rt():
    with pytest.raises(ValueError, match="covers"):
        simulate(np.ones(5), np.array([1.0]), seed=1.0, seed_days=2, n_days=10)


# --------------------------------------------------------------------------
# simulate: susceptibility adjustment
# --------------------------------------------------------------------------


def _setup(n_days=60):
    rng = np.random.default_rng(SEED)
    gen = rng.dirichlet(np.ones(7))
    R = np.full(n_days, 1.6)
    return gen, R


def test_pop_adjustment_reduces_infections_and_keeps_susceptibles_positive():
    gen, R = _setup()
    plain = simulate(R, gen, seed=50.0, seed_days=6, n_days=60)
    adj = simulate(R, gen, seed=50.0, seed_days=6, n_days=60, pop=1e5)

    assert np.all(adj <= plain + 1e-9)
    assert adj[-1] < plain[-1]  # depletion must actually bite by the end

    susc = 1e5 - np.cumsum(adj)
    assert np.all(susc >= 0.0)
    assert np.all(np.diff(susc) <= 0.0)  # susceptibles only ever decrease
    assert np.all(adj >= 0.0)


def test_pop_adjustment_never_exceeds_the_susceptible_pool():
    """A wildly over-seeded epidemic must not infect more people than exist."""
    gen, R = _setup(n_days=30)
    adj = simulate(R * 10, gen, seed=1e6, seed_days=3, n_days=30, pop=1e4)
    assert adj.sum() <= 1e4 + 1e-6
    assert np.all(1e4 - np.cumsum(adj) >= 0.0)


def test_huge_population_converges_to_the_unadjusted_recursion():
    gen, R = _setup(n_days=40)
    plain = simulate(R, gen, seed=10.0, seed_days=5, n_days=40)
    adj = simulate(R, gen, seed=10.0, seed_days=5, n_days=40, pop=1e18)
    np.testing.assert_allclose(adj, plain, rtol=1e-8)


def test_susceptible0_scales_the_starting_pool():
    """susceptible0 is R's pops * S0: a partly-immune population grows slower."""
    gen, R = _setup(n_days=40)
    full = simulate(R, gen, seed=10.0, seed_days=5, n_days=40, pop=1e5)
    half = simulate(R, gen, seed=10.0, seed_days=5, n_days=40, pop=1e5,
                    susceptible0=5e4)
    assert np.all(half <= full + 1e-9)
    assert half[-1] < full[-1]
    # The seeding days are adjusted too (as in R's Stan), so even they differ.
    assert half[0] < full[0]


def test_pop_adjustment_matches_the_stan_recursion_by_hand():
    """Explicit three-day check of i_t = S_t (1 - exp(-i'_t / pop))."""
    gen = np.array([1.0])  # all weight on yesterday
    pop = 1000.0
    R = np.array([1.0, 2.0, 2.0])
    inf = simulate(R, gen, seed=100.0, seed_days=1, n_days=3, pop=pop)

    s0 = pop
    i0 = s0 * (1 - np.exp(-100.0 / pop))          # seed day, adjusted
    s1 = s0 - i0
    i1 = s1 * (1 - np.exp(-(2.0 * i0) / pop))     # load = i0 (gen = [1])
    s2 = s1 - i1
    i2 = s2 * (1 - np.exp(-(2.0 * i1) / pop))
    np.testing.assert_allclose(inf, [i0, i1, i2], rtol=1e-12)


def test_pop_adjustment_vectorises_over_draws():
    gen, R = _setup(n_days=25)
    pops = np.array([1e4, 1e6])
    batched = simulate(np.stack([R, R]), gen, seed=np.array([20.0, 20.0]),
                       seed_days=4, n_days=25, pop=pops)
    assert batched.shape == (2, 25)
    for d, p in enumerate(pops):
        one = simulate(R, gen, seed=20.0, seed_days=4, n_days=25, pop=p)
        np.testing.assert_allclose(batched[d], one, rtol=1e-12)
    # the small population is the more depleted one
    assert batched[0].sum() / pops[0] > batched[1].sum() / pops[1]


# --------------------------------------------------------------------------
# expected_observations
# --------------------------------------------------------------------------


def test_convolution_hand_computed():
    inf = np.array([1.0, 2.0, 3.0, 4.0])
    i2o = np.array([0.5, 0.3])
    # E_0 = 0; E_1 = .5*1; E_2 = .5*2 + .3*1; E_3 = .5*3 + .3*2
    np.testing.assert_allclose(
        expected_observations(inf, i2o), [0.0, 0.5, 1.3, 2.1]
    )


def test_convolution_never_uses_the_same_day_infection():
    inf = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    y = expected_observations(inf, np.array([1.0]))
    assert y[2] == 0.0
    np.testing.assert_allclose(y, [0.0, 0.0, 0.0, 5.0, 0.0])


def test_convolution_matches_the_renewal_reference():
    rng = np.random.default_rng(SEED)
    inf = rng.uniform(0.0, 100.0, size=50)
    i2o = rng.dirichlet(np.ones(20))
    np.testing.assert_allclose(
        expected_observations(inf, i2o, 0.01),
        ref_expected_observations(inf, i2o, 0.01),
        rtol=1e-12,
    )


def test_convolution_batches_over_draws_with_per_draw_ascertainment():
    rng = np.random.default_rng(SEED)
    inf = rng.uniform(0.0, 100.0, size=(4, 30))
    i2o = rng.dirichlet(np.ones(6))
    ifr = np.array([0.005, 0.01, 0.02, 0.04])[:, None]
    out = expected_observations(inf, i2o, ifr)
    assert out.shape == (4, 30)
    for d in range(4):
        np.testing.assert_allclose(
            out[d], ref_expected_observations(inf[d], i2o, ifr[d, 0]), rtol=1e-12
        )


def test_convolution_handles_a_kernel_longer_than_the_series():
    inf = np.array([1.0, 1.0, 1.0])
    i2o = np.repeat(0.1, 30)
    np.testing.assert_allclose(
        expected_observations(inf, i2o), [0.0, 0.1, 0.2]
    )


# --------------------------------------------------------------------------
# posterior_predict
# --------------------------------------------------------------------------


def test_predict_rejects_unknown_family_and_missing_aux():
    with pytest.raises(ValueError, match="unknown family"):
        posterior_predict(np.ones((2, 3)), "binomial")
    with pytest.raises(ValueError, match="requires aux"):
        posterior_predict(np.ones((2, 3)), "neg_binom")


@pytest.mark.parametrize(
    "family,aux",
    [("poisson", None), ("neg_binom", 8.0), ("quasi_poisson", 3.0),
     ("normal", 5.0)],
)
def test_predict_shape_mean_and_spread(family, aux):
    expected = np.full((4000, 3), 40.0)
    y = posterior_predict(expected, family, aux=aux, rng=SEED)

    assert y.shape == expected.shape
    # mean tracks `expected` ...
    np.testing.assert_allclose(y.mean(axis=0), 40.0, rtol=0.05)
    # ... but the draws are not the mean: this is the entire point of the
    # function -- banding these is strictly wider than banding `expected`.
    assert np.all(y.var(axis=0) > 0.0)
    assert y.std() > 1.0


def test_predict_is_reproducible_and_seed_dependent():
    e = np.full((50, 4), 30.0)
    a = posterior_predict(e, "poisson", rng=SEED)
    b = posterior_predict(e, "poisson", rng=SEED)
    c = posterior_predict(e, "poisson", rng=SEED + 1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_predict_accepts_a_generator():
    e = np.full((10, 3), 5.0)
    a = posterior_predict(e, "poisson", rng=np.random.default_rng(SEED))
    b = posterior_predict(e, "poisson", rng=np.random.default_rng(SEED))
    np.testing.assert_array_equal(a, b)


def test_poisson_variance_equals_the_mean():
    y = posterior_predict(np.full((200_000, 1), 25.0), "poisson", rng=SEED)
    assert y.mean() == pytest.approx(25.0, rel=0.01)
    assert y.var() == pytest.approx(25.0, rel=0.05)


def test_neg_binom_matches_the_pymc_parameterisation():
    """aux must be PyMC's `alpha`: var = mu + mu^2 / alpha."""
    mu, alpha = 30.0, 4.0
    y = posterior_predict(np.full((200_000, 1), mu), "neg_binom", aux=alpha,
                          rng=SEED)
    assert y.mean() == pytest.approx(mu, rel=0.01)
    assert y.var() == pytest.approx(mu + mu**2 / alpha, rel=0.05)
    # and it must be over-dispersed relative to a Poisson with the same mean
    assert y.var() > mu


def test_quasi_poisson_variance_is_proportional_to_the_mean():
    """R uses size = mu / aux, so var = mu * (1 + aux) -- linear in mu."""
    mu, phi = 20.0, 3.0
    y = posterior_predict(np.full((200_000, 1), mu), "quasi_poisson", aux=phi,
                          rng=SEED)
    assert y.mean() == pytest.approx(mu, rel=0.01)
    assert y.var() == pytest.approx(mu * (1 + phi), rel=0.05)


def test_normal_uses_aux_as_the_standard_deviation():
    y = posterior_predict(np.full((100_000, 1), -2.0), "normal", aux=1.5, rng=SEED)
    assert y.mean() == pytest.approx(-2.0, abs=0.02)
    assert y.std() == pytest.approx(1.5, rel=0.02)


def test_log_normal_mean_is_exp_of_expected():
    """R draws rlnorm(meanlog = x - s^2/2, sdlog = s), so E[y] = exp(x)."""
    y = posterior_predict(np.full((200_000, 1), 1.0), "log_normal", aux=0.4,
                          rng=SEED)
    assert y.mean() == pytest.approx(np.exp(1.0), rel=0.01)


def test_per_draw_aux_pairs_with_the_draw_axis():
    """A (draws,) aux vector must vary down the rows, not along time."""
    expected = np.full((2, 20_000), 50.0)
    aux = np.array([0.5, 200.0])  # very over-dispersed vs nearly Poisson
    y = posterior_predict(expected, "neg_binom", aux=aux, rng=SEED)
    assert y.shape == expected.shape
    assert y[0].var() > 10 * y[1].var()
    np.testing.assert_allclose(y.mean(axis=1), 50.0, rtol=0.1)


def test_count_families_tolerate_a_zero_expectation():
    y = posterior_predict(np.zeros((5, 4)), "poisson", rng=SEED)
    np.testing.assert_array_equal(y, 0)
    y = posterior_predict(np.zeros((5, 4)), "neg_binom", aux=2.0, rng=SEED)
    np.testing.assert_array_equal(y, 0)
    y = posterior_predict(np.zeros((5, 4)), "quasi_poisson", aux=2.0, rng=SEED)
    np.testing.assert_array_equal(y, 0)


def test_predictive_bands_are_wider_than_bands_on_the_mean():
    """The bug this module fixes: ribbons on the posterior mean exclude noise."""
    rng = np.random.default_rng(SEED)
    expected = np.exp(rng.normal(np.log(100.0), 0.15, size=(2000, 1)))
    lo_mean, hi_mean = np.percentile(expected, [5, 95])
    y = posterior_predict(expected, "neg_binom", aux=6.0, rng=SEED)
    lo_pred, hi_pred = np.percentile(y, [5, 95])
    assert lo_pred < lo_mean and hi_pred > hi_mean


# --------------------------------------------------------------------------
# end-to-end: draws -> infections -> expected -> predictive
# --------------------------------------------------------------------------


def test_full_forecast_pipeline_over_draws():
    rng = np.random.default_rng(SEED)
    n_draws, n_days = 200, 90
    gen = rng.dirichlet(np.ones(10))
    i2o = rng.dirichlet(np.ones(15))
    Rt = np.exp(rng.normal(0.35, 0.1, size=(n_draws, n_days)))
    seeds = rng.uniform(80.0, 150.0, size=n_draws)
    ifr = rng.uniform(0.005, 0.015, size=n_draws)[:, None]
    phi = rng.uniform(4.0, 12.0, size=n_draws)

    inf = simulate(Rt, gen, seed=seeds, seed_days=6, n_days=n_days, pop=5e6)
    E = expected_observations(inf, i2o, ifr)
    y = posterior_predict(E, "neg_binom", aux=phi, rng=SEED)

    assert inf.shape == E.shape == y.shape == (n_draws, n_days)
    assert np.all(np.isfinite(E)) and np.all(y >= 0)
    # predictive intervals contain the mean-only intervals at every time point
    q_mean = np.percentile(E, [10, 90], axis=0)
    q_pred = np.percentile(y, [10, 90], axis=0)
    # Restrict to days where the expected count is comfortably above 1: at
    # counts of ~0 both sets of quantiles collapse onto 0 and the comparison
    # says nothing (integer draws vs a continuous mean).
    late = np.where(E.mean(axis=0) > 10.0)[0]
    assert late.size > 10
    assert np.all(q_pred[0, late] <= q_mean[0, late])
    assert np.all(q_pred[1, late] >= q_mean[1, late])
