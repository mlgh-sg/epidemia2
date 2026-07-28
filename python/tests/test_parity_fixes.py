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
