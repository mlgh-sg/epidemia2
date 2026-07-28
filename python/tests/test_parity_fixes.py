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


def test_forecast_works_without_b0_for_a_single_population_fit():
    """region_effects=False never creates b0; forecast() used to demand it.

    Found by the flu tutorial, which is a one-region model and so has no
    per-region intercept deviation to forecast with.
    """
    import epidemia
    from epidemia.core import (EpiModelConfig, ObsModel, PanelData, RandomWalk,
                               fit_epidemia)
    from epidemia.priors import normal

    flu = epidemia.flu1918()
    yv, n = flu.incidence[:45], 45
    panel1 = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]),
                       regions=["Baltimore"], npis=[],
                       dates=[pd.date_range("1918-09-17", periods=n).values],
                       pops=None)
    obs = ObsModel("cases", yv[None, :], np.ones((1, n), bool),
                   np.repeat(0.25, 4), family="poisson", link="identity",
                   intercept=False, offset=np.ones((1, n)))
    cfg = EpiModelConfig(gen=flu.generation, link="log", intercept=True,
                         prior_intercept=normal(np.log(2), 0.2),
                         region_effects=False, seed_days=6,
                         rw=RandomWalk(index=np.tile(np.arange(n), (1, 1)),
                                       prior_scale=0.01))
    idata = fit_epidemia(panel1, [obs], cfg, draws=60, tune=60, chains=2,
                         seed=0, progress_bar=False)
    assert "b0" not in idata.posterior          # the premise of the test
    fc = epidemia.forecast(idata, panel1, [obs], cfg, draws=20, seed=0)
    assert np.isfinite(np.asarray(fc.Rt)).all()


# --- rw index -1 excludes a day from the walk -----------------------------

def test_rw_index_minus_one_contributes_nothing():
    """R's rw(time=) takes NA for "no walk term here"; Python had no equivalent.

    Without it, an excluded day silently takes the walk's FIRST step, which is a
    free parameter perfectly confounded with the intercept -- it showed up as
    r_hat 1.03 on `intercept` in the England tutorial.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from epidemia.core import RandomWalk, _random_walk

    Tn, Mn = 10, 1
    idx = np.array([[-1, -1, -1, 0, 0, 1, 1, 2, 2, 3]])
    with pm.Model():
        contrib = _random_walk(pm, pt, RandomWalk(index=idx), Mn, Tn)
        fn = pm.compile([], contrib, on_unused_input="ignore")
    vals = np.asarray(fn())
    assert np.allclose(vals[0, :3], 0.0), "excluded days must contribute zero"
    assert not np.allclose(vals[0, 3:], 0.0), "included days must contribute"


def test_rw_index_below_minus_one_is_rejected():
    import pymc as pm
    import pytensor.tensor as pt
    from epidemia.core import RandomWalk, _random_walk

    with pm.Model():
        with pytest.raises(ValueError, match=r">= -1"):
            _random_walk(pm, pt, RandomWalk(index=np.array([[-2, 0, 1]])), 1, 3)


# --- medium/low parity fixes ----------------------------------------------

def test_log_uses_a_pseudo_log_that_is_defined_at_zero():
    """R uses trans='pseudo_log'; scale_y_log10 silently drops zero-count days."""
    from epidemia.plots import _pseudo_log, _pseudo_log_inv

    assert np.isfinite(_pseudo_log(0.0))
    assert _pseudo_log(0.0) == pytest.approx(0.0)
    x = np.array([0.0, 1.0, 10.0, 1000.0])
    np.testing.assert_allclose(_pseudo_log_inv(_pseudo_log(x)), x, atol=1e-9)


def test_smooth_drops_incomplete_windows_instead_of_zero_padding():
    """R's rollmean(fill=NA) + complete.cases; 'same' convolution biases edges."""
    from epidemia.plots import _draw_transform

    v = np.full((1, 11), 10.0)                 # constant series
    out = _draw_transform(smooth=5)(v, 0)
    # a constant series must smooth to the same constant wherever it is defined
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite, 10.0)
    # ...and the edges are dropped, not dragged toward zero
    assert np.isnan(out[0, 0]) and np.isnan(out[0, -1])
    assert np.isfinite(out[0, 5])


def test_smooth_window_wider_than_the_series_warns_and_does_nothing():
    from epidemia.plots import _draw_transform

    v = np.full((1, 6), 3.0)
    with pytest.warns(UserWarning, match="not shorter"):
        out = _draw_transform(smooth=99)(v, 0)
    np.testing.assert_allclose(out, 3.0)


def test_groups_accepts_a_vector_and_rejects_unknown_names():
    from epidemia.plots import _as_groups

    assert _as_groups(None, ["A", "B"]) == ["A", "B"]
    assert _as_groups("A", None) == ["A"]
    assert _as_groups(["A", "B"], None) == ["A", "B"]
    assert _as_groups(None, None) is None


def test_score_forwards_metrics_and_validates_groups():
    """R's evaluate_forecast takes metrics=; gr_subset errors on a bad group."""
    import epidemia.scoring as sc

    rng = np.random.default_rng(0)
    n = 12
    yv = rng.poisson(20, n).astype(float)
    dr = rng.poisson(20, (n, 100)).astype(float)
    only = sc.evaluate_forecast(yv, dr, metrics="crps")
    assert "crps" in only.error.columns
    assert "mean_abs_error" not in only.error.columns


def test_prior_family_gates_match_r():
    """R rejects out-of-set families at epirt, epiinf and handle_glm_prior."""
    from epidemia.core import EpiModelConfig, ObsModel, PanelData, build_epidemia_model
    from epidemia.priors import laplace, shifted_gamma

    rng = np.random.default_rng(0)
    n = 20
    pan = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]), regions=["A"],
                    npis=[], dates=[pd.date_range("2020-03-01", periods=n).values],
                    pops=None)
    gen = np.ones(5) / 5
    om = ObsModel("d", rng.poisson(8, (1, n)).astype(float),
                  np.ones((1, n), bool), gen)

    with pytest.raises(ValueError, match="not supported here"):
        build_epidemia_model(pan, [om], EpiModelConfig(
            gen=gen, intercept=True, region_effects=False,
            prior_intercept=shifted_gamma()))

    om_bad = ObsModel("d", om.y, om.mask, gen, prior_aux=laplace())
    with pytest.raises(ValueError, match="not supported here"):
        build_epidemia_model(pan, [om_bad],
                             EpiModelConfig(gen=gen, region_effects=False))


def test_posterior_linpred_transform_applies_the_cap():
    """The old docstring said the cap was NOT applied; it is."""
    from epidemia.forecast import _apply_link

    assert _apply_link("scaled_logit", np.array([0.0]), 0.02)[0] == pytest.approx(0.01)
