"""Tests for the per-draw trajectory ("spaghetti") plots.

No model is fitted: the fixtures build InferenceData with the dims and coords the
real fits produce, which is all the plotting code reads.
"""

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

import epidemia as epi
from epidemia.multilevel import MultilevelData
from epidemia.plots import (
    _path_frame,
    _sample_draw_index,
    spaghetti_infections,
    spaghetti_obs,
    spaghetti_rt,
)

M, T, K = 3, 25, 2
C, D = 2, 20
REGIONS = ["Italy", "Sweden", "Norway"]
LENGTHS = np.array([25, 20, 15])


@pytest.fixture(scope="module")
def panel():
    """A hand-built MultilevelData: staggered starts, so per-region dates matter."""
    rng = np.random.default_rng(7)
    deaths = rng.poisson(30, (M, T))
    mask = np.zeros((M, T), dtype=bool)
    for m, n in enumerate(LENGTHS):
        mask[m, :n] = True
        deaths[m, n:] = 0
    dates = [
        np.asarray(pd.date_range(f"2020-02-{10 + 5 * m:02d}", periods=int(n)))
        for m, n in enumerate(LENGTHS)
    ]
    return MultilevelData(deaths=deaths, X=rng.binomial(1, 0.4, (M, T, K)).astype(float),
                          mask=mask, lengths=LENGTHS, regions=list(REGIONS),
                          npis=["lockdown", "schools"], dates=dates)


@pytest.fixture(scope="module")
def fake_multilevel_idata():
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(0)
    post = xr.Dataset(
        {
            "Rt": (("chain", "draw", "region", "region_time"),
                   rng.gamma(4, 0.4, (C, D, M, T))),
            "infections": (("chain", "draw", "region", "region_time"),
                           rng.gamma(4, 50, (C, D, M, T))),
            "E_deaths": (("chain", "draw", "region", "region_time"),
                         rng.gamma(4, 20, (C, D, M, T))),
        },
        coords={"region": REGIONS, "region_time": np.arange(T),
                "chain": np.arange(C), "draw": np.arange(D)},
    )
    return az.InferenceData(posterior=post)


@pytest.fixture(scope="module")
def fake_single_idata():
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(1)
    return az.InferenceData(posterior=xr.Dataset({
        "Rt": (("chain", "draw", "time"), rng.gamma(4, 0.4, (C, D, T))),
        "infections": (("chain", "draw", "time"), rng.gamma(4, 50, (C, D, T))),
        "E_obs": (("chain", "draw", "time"), rng.gamma(4, 20, (C, D, T))),
    }))


# --------------------------------------------------------------------------
# Draw selection
# --------------------------------------------------------------------------


def test_seed_makes_the_selection_reproducible():
    a = _sample_draw_index(200, 10, seed=0)
    b = _sample_draw_index(200, 10, seed=0)
    c = _sample_draw_index(200, 10, seed=1)
    assert np.array_equal(a, b), "the same seed must pick the same paths"
    assert not np.array_equal(a, c), "a different seed must pick different paths"
    assert len(a) == 10 and len(set(a.tolist())) == 10, "no draw twice"
    assert (a >= 0).all() and (a < 200).all()
    assert list(a) == sorted(a), "indices are sorted, so the draw order is stable too"


def test_more_draws_than_exist_is_capped_not_an_error():
    idx = _sample_draw_index(12, 500, seed=0)
    assert len(idx) == 12
    assert sorted(idx.tolist()) == list(range(12)), "a cap means: use them all"


def test_draws_none_uses_every_draw():
    assert len(_sample_draw_index(37, None, seed=0)) == 37


def test_bad_draws_or_alpha_raise(fake_single_idata):
    with pytest.raises(ValueError, match="draws"):
        spaghetti_rt(fake_single_idata, draws=0, save=False)
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="alpha"):
            spaghetti_rt(fake_single_idata, alpha=bad, save=False)


def test_path_frame_keeps_each_draw_a_separate_line():
    draws = np.arange(6 * 4, dtype=float).reshape(6, 4)
    df = _path_frame(draws, np.arange(4), np.array([1, 4]))
    assert len(df) == 2 * 4
    assert set(df["draw"]) == {"1", "4"}
    # values must follow their own draw, not be reshaped across draws
    assert list(df[df["draw"] == "4"]["value"]) == list(draws[4])


# --------------------------------------------------------------------------
# Single-population
# --------------------------------------------------------------------------


def test_single_population_plots_return_ggplots(fake_single_idata):
    for p in (
        spaghetti_rt(fake_single_idata, save=False),
        spaghetti_infections(fake_single_idata, save=False),
        spaghetti_obs(fake_single_idata, observed=np.arange(T) * 2.0, save=False),
    ):
        assert isinstance(p, ggplot)


def test_plots_actually_render(fake_single_idata, fake_multilevel_idata, panel,
                               tmp_path, monkeypatch):
    """A ggplot is lazy: layer errors only surface when it is drawn."""
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    spaghetti_rt(fake_single_idata, draws=5, save="sp-rt")
    spaghetti_obs(fake_single_idata, observed=np.arange(T) * 2.0, draws=5, save="sp-obs")
    # the multi-region obs path draws bars from data.deaths on real dates
    spaghetti_obs(fake_multilevel_idata, data=panel, draws=5, save="ml-obs")
    for f in ("sp-rt.png", "sp-obs.png", "ml-obs.png"):
        assert (tmp_path / f).exists()


def test_multiregion_spaghetti_obs_rejects_an_observed_array(fake_multilevel_idata,
                                                             panel):
    with pytest.raises(ValueError, match="observed"):
        spaghetti_obs(fake_multilevel_idata, observed=np.arange(5), data=panel,
                      save=False)


def test_spaghetti_obs_needs_an_expected_observation_variable():
    import arviz as az
    import xarray as xr

    idata = az.InferenceData(posterior=xr.Dataset({
        "Rt": (("chain", "draw", "time"), np.ones((C, D, T))),
    }))
    with pytest.raises(KeyError, match="E_obs"):
        spaghetti_obs(idata, save=False)


# --------------------------------------------------------------------------
# Multi-region
# --------------------------------------------------------------------------


def test_multiregion_plots_return_ggplots(fake_multilevel_idata, panel):
    for p in (
        spaghetti_rt(fake_multilevel_idata, data=panel, save=False),
        spaghetti_infections(fake_multilevel_idata, data=panel, save=False),
        spaghetti_obs(fake_multilevel_idata, data=panel, save=False),
    ):
        assert isinstance(p, ggplot)


def test_multiregion_requires_data(fake_multilevel_idata):
    with pytest.raises(ValueError, match="region"):
        spaghetti_rt(fake_multilevel_idata, save=False)


def test_region_selection(fake_multilevel_idata, panel):
    from epidemia.plots import _region_path_frame

    idx = np.array([0, 1, 2])
    paths, med, keep = _region_path_frame(fake_multilevel_idata, "Rt", panel, idx,
                                          group="Sweden")
    assert keep == ["Sweden"]
    assert set(paths["region"]) == {"Sweden"} and set(med["region"]) == {"Sweden"}
    n = int(LENGTHS[REGIONS.index("Sweden")])
    assert len(paths) == len(idx) * n, "padded days must be dropped"
    assert len(med) == n
    # each region's own dates, not a shared axis
    assert paths["x"].min() == panel.dates[REGIONS.index("Sweden")][0]

    # `region=` is accepted as an alias of `group=`
    assert isinstance(spaghetti_rt(fake_multilevel_idata, data=panel, region="Sweden",
                                   save=False), ggplot)


def test_unknown_region_raises(fake_multilevel_idata, panel):
    with pytest.raises(ValueError, match="unknown region"):
        spaghetti_rt(fake_multilevel_idata, data=panel, group="Atlantis", save=False)


def test_all_regions_share_the_same_draws(fake_multilevel_idata, panel):
    """A draw is a joint sample; mixing draw ids across regions is not a posterior."""
    from epidemia.plots import _region_path_frame

    idx = _sample_draw_index(C * D, 5, seed=3)
    paths, _, _ = _region_path_frame(fake_multilevel_idata, "Rt", panel, idx)
    per_region = {r: set(g["draw"]) for r, g in paths.groupby("region")}
    assert set(per_region) == set(REGIONS)
    assert all(v == {str(i) for i in idx} for v in per_region.values())


def test_plot_is_reproducible_for_a_fixed_seed(fake_multilevel_idata, panel):
    from epidemia.plots import _region_path_frame

    idx0 = _sample_draw_index(C * D, 7, seed=11)
    a, _, _ = _region_path_frame(fake_multilevel_idata, "Rt", panel, idx0)
    b, _, _ = _region_path_frame(fake_multilevel_idata, "Rt", panel, idx0)
    pd.testing.assert_frame_equal(a, b)


def test_median_uses_all_draws_not_just_the_plotted_ones(fake_multilevel_idata, panel):
    """R computes the overlaid median from the full posterior; so must we."""
    from epidemia.plots import _draws, _region_path_frame

    arr, _ = _draws(fake_multilevel_idata, "Rt")
    idx = _sample_draw_index(arr.shape[0], 3, seed=0)
    _, med, _ = _region_path_frame(fake_multilevel_idata, "Rt", panel, idx)
    m = REGIONS.index("Italy")
    n = int(LENGTHS[m])
    expected = np.median(arr[:, m, :n], axis=0)
    got = med[med["region"] == "Italy"]["median"].to_numpy()
    assert np.allclose(got, expected)
    assert not np.allclose(got, np.median(arr[idx][:, m, :n], axis=0)), \
        "a 3-draw median would not match the full-posterior one"


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def test_spaghetti_plots_are_saved_by_default(fake_multilevel_idata, panel, tmp_path,
                                              monkeypatch):
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    spaghetti_rt(fake_multilevel_idata, data=panel, group="Italy", draws=5)
    assert (tmp_path / "spaghetti-rt.png").exists()
    spaghetti_infections(fake_multilevel_idata, data=panel, group="Italy", draws=5,
                         save="custom-spaghetti")
    assert (tmp_path / "custom-spaghetti.png").exists()


def test_save_false_writes_nothing(fake_multilevel_idata, panel, tmp_path, monkeypatch):
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    spaghetti_rt(fake_multilevel_idata, data=panel, save=False)
    assert not list(tmp_path.glob("*.png"))


def test_exposed_on_the_plots_module():
    for name in ("spaghetti_rt", "spaghetti_infections", "spaghetti_obs"):
        assert hasattr(epi.plots, name)
