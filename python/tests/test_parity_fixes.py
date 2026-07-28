"""Regression tests for parity gaps found by the R-vs-Python audit."""

import json

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from epidemia.core import ObsModel, PanelData
from epidemia.plots import _draws, _predictive_draws, plot_obs

M, T = 2, 30


def _idata(with_family=True, aux=True):
    rng = np.random.default_rng(0)
    v = {"E_deaths": (("chain", "draw", "region", "region_time"),
                      rng.random((2, 60, M, T)) * 20 + 10)}
    if aux:
        v["deaths|aux"] = (("chain", "draw"), rng.random((2, 60)) * 4 + 4)
    post = xr.Dataset(v, coords={"chain": [0, 1], "draw": range(60),
                                 "region": ["A", "B"], "region_time": range(T)})
    idata = az.InferenceData(posterior=post)
    if with_family:
        idata.attrs["epidemia_families"] = json.dumps({"deaths": "neg_binom"})
    return idata


def _panel():
    return PanelData(X=np.zeros((M, T, 0)), lengths=np.full(M, T),
                     regions=["A", "B"], npis=[],
                     dates=[pd.date_range("2020-03-01", periods=T).values] * M,
                     pops=np.array([1e6, 2e6]))


def _obs_model():
    rng = np.random.default_rng(1)
    return ObsModel("deaths", rng.poisson(15, (M, T)).astype(float),
                    np.ones((M, T), bool), np.array([0.5, 0.5]))


# --- plot_obs bands the predictive, not the mean --------------------------

def test_predictive_band_is_wider_than_the_mean_band():
    """R's plot_obs bands posterior_predict; banding the mean is far too narrow."""
    idata = _idata()
    pred = _predictive_draws(idata, "E_deaths", "deaths")
    assert pred is not None
    mean_arr, _ = _draws(idata, "E_deaths")

    def width(a):
        return (np.percentile(a, 97.5, axis=0) - np.percentile(a, 2.5, axis=0)).mean()

    assert width(pred) > 1.5 * width(mean_arr)


def test_family_is_read_from_the_fit_or_from_the_obs_model():
    assert _predictive_draws(_idata(with_family=True), "E_deaths", "deaths") is not None
    # no family recorded -> cannot draw
    assert _predictive_draws(_idata(with_family=False), "E_deaths", "deaths") is None
    # ...unless the ObsModel supplies it
    got = _predictive_draws(_idata(with_family=False), "E_deaths", "deaths",
                            obs_model=_obs_model())
    assert got is not None


def _single_pop_idata(with_family=False):
    """A one-population fit: no region dim, so no panel counts are needed."""
    rng = np.random.default_rng(3)
    post = xr.Dataset(
        {"E_deaths": (("chain", "draw", "time"), rng.random((2, 40, T)) * 20 + 10)},
        coords={"chain": [0, 1], "draw": range(40), "time": range(T)})
    idata = az.InferenceData(posterior=post)
    if with_family:
        idata.attrs["epidemia_families"] = json.dumps({"deaths": "poisson"})
    return idata


def test_plot_obs_warns_rather_than_silently_banding_the_mean():
    """Falling back to the mean must be loud: the interval is not comparable."""
    with pytest.warns(UserWarning, match="too narrow"):
        plot_obs(_single_pop_idata(with_family=False), save=False)


def test_plot_obs_predictive_false_is_silent():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_obs(_single_pop_idata(with_family=False), save=False,
                 predictive=False)


def test_plot_obs_uses_the_recorded_family_without_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_obs(_single_pop_idata(with_family=True), save=False)


# --- posterior_linpred sums every walk ------------------------------------

def test_linpred_sums_all_rt_walks_and_observation_walks():
    from epidemia.postprocess import _walk_contribution

    rng = np.random.default_rng(2)
    steps = 5
    post = xr.Dataset(
        {"rw": (("chain", "draw", "proc", "step"), rng.normal(size=(1, 8, 1, steps))),
         "rw2": (("chain", "draw", "proc", "step"), rng.normal(size=(1, 8, 1, steps))),
         "deaths|rw": (("chain", "draw", "proc", "step"),
                       rng.normal(size=(1, 8, 1, steps)))},
        coords={"chain": [0], "draw": range(8), "proc": [0], "step": range(steps)})
    panel = _panel()
    panel.rw_index = np.tile(np.repeat(np.arange(steps), T // steps)[:T], (M, 1))

    both = _walk_contribution(post, panel, M, prefix="")
    only_first = _walk_contribution({"rw": post["rw"]}, panel, M, prefix="")
    # rw2 must contribute: summing both differs from summing only rw
    assert not np.allclose(both, only_first)

    # observation walks live under a "<series>|" prefix and were dropped entirely
    obs_walk = _walk_contribution(post, panel, M, prefix="deaths|")
    assert np.any(obs_walk != 0)


# --- autoscale is actually applied by the model builder -------------------

def test_model_builder_autoscales_a_shifted_gamma_covariate_prior():
    """R divides the prior scale by each covariate's own scale; Python never did.

    Checked through the built model rather than the helper, because the gap was
    that nothing *called* the (correct) helper.
    """
    import epidemia.priors as pr
    from epidemia.core import EpiModelConfig, build_epidemia_model

    seen = []
    real = pr.autoscale

    def spy(spec, sd):
        seen.append(np.atleast_1d(np.asarray(sd, dtype=float)).copy())
        return real(spec, sd)

    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 10.0, size=(M, T, 1))       # sd far from 1
    panel = PanelData(X=X, lengths=np.full(M, T), regions=["A", "B"], npis=["x"],
                      dates=[pd.date_range("2020-03-01", periods=T).values] * M,
                      pops=np.array([1e6, 2e6]))
    gen = np.ones(10) / 10
    om = ObsModel("deaths", rng.poisson(10, (M, T)).astype(float),
                  np.ones((M, T), bool), gen)
    cfg = EpiModelConfig(gen=gen, prior_covariates=pr.shifted_gamma(1 / 6, 1.0, 0.0))

    pr.autoscale = spy
    try:
        build_epidemia_model(panel, [om], cfg)
    finally:
        pr.autoscale = real

    assert seen, "the builder never autoscaled the covariate prior"
    assert seen[0][0] == pytest.approx(np.std(X.reshape(-1), ddof=1), rel=1e-6)


def test_autoscale_can_be_switched_off_on_the_config():
    import epidemia.priors as pr
    from epidemia.core import EpiModelConfig, build_epidemia_model

    called = []
    real = pr.autoscale
    pr.autoscale = lambda spec, sd: (called.append(1), real(spec, sd))[1]
    try:
        rng = np.random.default_rng(0)
        panel = PanelData(X=rng.normal(size=(M, T, 1)), lengths=np.full(M, T),
                          regions=["A", "B"], npis=["x"],
                          dates=[pd.date_range("2020-03-01", periods=T).values] * M,
                          pops=np.array([1e6, 2e6]))
        gen = np.ones(10) / 10
        om = ObsModel("deaths", rng.poisson(10, (M, T)).astype(float),
                      np.ones((M, T), bool), gen)
        build_epidemia_model(panel, [om], EpiModelConfig(
            gen=gen, prior_covariates=pr.shifted_gamma(1 / 6, 1.0, 0.0),
            autoscale=False))
    finally:
        pr.autoscale = real
    assert not called


# --- a latent fit can now be forecast -------------------------------------

def test_latent_forecast_is_stochastic_and_reads_the_fit_back():
    """R continues epiinf(latent=TRUE) with normal_lb_rng(mu, sigma, 0).

    Python used to refuse outright. The fitted window must come back from the
    posterior unchanged; the horizon must be drawn, not set to the mean.
    """
    from epidemia.forecast import _renewal

    rng = np.random.default_rng(0)
    S, Mr, Tt = 40, 2, 24
    Rt = np.full((S, Mr, Tt), 1.1)
    gen = np.ones(5) / 5
    seed = np.full((S, Mr), 10.0)
    fitted = rng.uniform(5.0, 15.0, size=(S, Mr, 12))     # the fitted window

    inf, _ = _renewal(Rt, gen, seed, seed_days=3,
                      latent_sd=lambda mean: np.sqrt(2.0 * mean),
                      rng=rng, observed=fitted)

    # in-sample days are read back verbatim
    np.testing.assert_allclose(inf[:, :, :12], fitted)
    # the horizon is not the deterministic mean: repeated draws differ
    again, _ = _renewal(Rt, gen, seed, seed_days=3,
                        latent_sd=lambda mean: np.sqrt(2.0 * mean),
                        rng=np.random.default_rng(1), observed=fitted)
    assert not np.allclose(inf[:, :, 12:], again[:, :, 12:])
    # and never negative -- R truncates at zero
    assert np.all(inf >= 0.0)


def test_latent_forecast_without_noise_matches_the_deterministic_recursion():
    from epidemia.forecast import _renewal

    Rt = np.full((5, 1, 15), 1.2)
    gen = np.ones(4) / 4
    seed = np.full((5, 1), 8.0)
    a, _ = _renewal(Rt, gen, seed, seed_days=3)
    b, _ = _renewal(Rt, gen, seed, seed_days=3, latent_sd=None,
                    rng=np.random.default_rng(0))
    np.testing.assert_allclose(a, b)
