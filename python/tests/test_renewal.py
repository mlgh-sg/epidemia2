"""Deterministic checks of the renewal dynamics (pure NumPy reference)."""

import numpy as np

from epidemia.renewal import expected_observations, random_walk, renewal_infections


def test_renewal_constant_infectiousness():
    # gen puts all weight on the 1-day-ago infection, R = 1 -> infections persist
    R = np.ones(5)
    gen = np.array([1.0])
    seeds = np.array([2.0, 3.0])
    inf = renewal_infections(R, seeds, gen)
    assert np.allclose(inf[:2], [2.0, 3.0])       # seeds preserved
    assert np.allclose(inf[2:], [3.0, 3.0, 3.0])  # i_t = i_{t-1}


def test_renewal_geometric_growth():
    # R = 2 with a 1-day generation -> exact doubling
    R = np.full(4, 2.0)
    gen = np.array([1.0])
    seeds = np.array([1.0])
    inf = renewal_infections(R, seeds, gen)
    assert np.allclose(inf, [1.0, 2.0, 4.0, 8.0])


def test_renewal_multiday_generation():
    # Two-day generation, R = 1: i_t = 0.5 i_{t-1} + 0.5 i_{t-2}
    R = np.ones(5)
    gen = np.array([0.5, 0.5])
    seeds = np.array([4.0, 2.0])
    inf = renewal_infections(R, seeds, gen)
    # i_2 = .5*2 + .5*4 = 3; i_3 = .5*3 + .5*2 = 2.5; i_4 = .5*2.5 + .5*3 = 2.75
    assert np.allclose(inf, [4.0, 2.0, 3.0, 2.5, 2.75])


def test_expected_obs_is_causal_convolution():
    # i2o is lag-1-first (like gen): i2o[k] weights the infection k+1 days ago,
    # matching R's Stan, which sums over infections[start .. t-1] and never lag 0.
    inf = np.array([1.0, 0.0, 0.0, 0.0])
    i2o = np.array([0.5, 0.3, 0.2])  # 1-, 2-, 3-day delay
    y = expected_observations(inf, i2o, 1.0)
    assert np.allclose(y, [0.0, 0.5, 0.3, 0.2])


def test_expected_obs_never_uses_a_same_day_infection():
    """A single infection on day 0 can never be observed on day 0."""
    inf = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    i2o = np.array([1.0])  # all mass on a 1-day delay
    y = expected_observations(inf, i2o, 1.0)
    assert y[2] == 0.0, "the day-2 infection must not be observed on day 2"
    assert np.allclose(y, [0.0, 0.0, 0.0, 5.0, 0.0])


def test_expected_obs_matches_the_generation_kernel_convention():
    """`gen` and `i2o` must index lags identically -- they used to disagree."""
    from epidemia.renewal import infectiousness

    x = np.array([1.0, 2.0, 3.0, 4.0])
    k = np.array([0.6, 0.4])
    np.testing.assert_allclose(expected_observations(x, k, 1.0), infectiousness(x, k))


def test_ascertainment_scales_observations():
    inf = np.ones(4)
    i2o = np.array([1.0])  # 1-day delay, so day 0 has no infection behind it
    y = expected_observations(inf, i2o, 0.1)
    assert np.allclose(y, [0.0, 0.1, 0.1, 0.1])


def test_numpy_reference_matches_the_pytensor_multilevel_model():
    """The NumPy forward simulation and the fitted model must agree exactly.

    The notebooks forward-simulate forecasts/counterfactuals with the NumPy
    reference while the posterior comes from the PyTensor model. If the two use
    different lag conventions the forecast silently contradicts the fit.
    """
    import pymc as pm

    import epidemia as epi

    ec = epi.europe_covid2()
    sub = ec.data[ec.data["country"] == "Italy"].copy()
    data = epi.prepare_panel(sub, ["lockdown"], fit_until="2020-05-05")
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
    model = epi.build_multilevel_model(data, config)
    with model:
        pr = pm.sample_prior_predictive(
            draws=1, random_seed=0,
            var_names=["Rt", "seed", "infections", "E_deaths", "ifr"])

    R = np.asarray(pr.prior["Rt"])[0, 0, 0]
    seed = float(np.asarray(pr.prior["seed"])[0, 0, 0])
    ifr = float(np.asarray(pr.prior["ifr"])[0, 0])
    inf_model = np.asarray(pr.prior["infections"])[0, 0, 0]
    E_model = np.asarray(pr.prior["E_deaths"])[0, 0, 0]

    inf_np = renewal_infections(R, np.full(config.seed_days, seed), np.asarray(ec.si))
    np.testing.assert_allclose(inf_np, inf_model, rtol=1e-6)

    E_np = expected_observations(inf_np, np.asarray(ec.inf2death), ifr) + 1e-15
    np.testing.assert_allclose(E_np, E_model, rtol=1e-6)


def test_single_population_model_convolution_starts_at_lag_one():
    """E_obs[t] must not depend on infections[t] in the SINGLE-population model.

    The multilevel model has its own version of this check. Without one here the
    lag convention in model.py is entirely untested -- reverting it to the old
    same-day form passes every other test in the suite.
    """
    import pymc as pm

    import epidemia as epi

    y = np.concatenate([[np.nan], np.full(29, 5.0)])
    config = epi.EpiConfig(
        gen=np.array([1.0]), i2o=np.array([1.0]),  # all mass at lag 1, both kernels
        seed_days=3, link="log", family="poisson",
    )
    model = epi.build_model(y, config)
    with model:
        pr = pm.sample_prior_predictive(draws=1, random_seed=0,
                                        var_names=["E_obs", "infections"])
    E = np.asarray(pr.prior["E_obs"])[0, 0]
    inf = np.asarray(pr.prior["infections"])[0, 0]
    # i2o = [1.0] at lag 1  =>  E_obs[t] == infections[t-1] (+ the 1e-6 floor)
    np.testing.assert_allclose(E[1:], inf[:-1] + 1e-15, rtol=1e-6)
    np.testing.assert_allclose(E[0], 1e-15, atol=1e-18)  # nothing precedes day 0


def test_single_population_model_matches_the_numpy_reference():
    """build_model's convolution must agree with expected_observations."""
    import pymc as pm

    import epidemia as epi

    d = epi.flu1918()
    y = np.concatenate([[np.nan], d.incidence])
    i2o = np.repeat(0.25, 4)
    config = epi.EpiConfig(gen=d.generation, i2o=i2o, seed_days=6, link="log",
                           family="poisson")
    model = epi.build_model(y, config)
    with model:
        pr = pm.sample_prior_predictive(draws=1, random_seed=0,
                                        var_names=["E_obs", "infections"])
    E = np.asarray(pr.prior["E_obs"])[0, 0]
    inf = np.asarray(pr.prior["infections"])[0, 0]
    np.testing.assert_allclose(E, expected_observations(inf, i2o, 1.0) + 1e-15, rtol=1e-6)


def test_non_centered_random_walk():
    walk = random_walk(0.5, np.array([1.0, 1.0, 1.0]), 2.0)
    assert np.allclose(walk, [2.5, 3.0, 3.5])
