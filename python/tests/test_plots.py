"""Tests for the plotting helpers, especially the multi-region (multilevel) path.

These are regression tests for a class of bug that is worse than a crash: a
plot that renders happily while showing the wrong thing.
"""

import numpy as np
import pytest

import epidemia as epi


@pytest.fixture(scope="module")
def panel():
    ec = epi.europe_covid2()
    sub = ec.data[ec.data["country"].isin(["Italy", "Sweden", "Norway"])].copy()
    return epi.prepare_panel(sub, ["lockdown"], seed_offset=30, death_threshold=10,
                             fit_until="2020-05-05")


@pytest.fixture(scope="module")
def fake_multilevel_idata(panel):
    """An InferenceData with the multilevel model's dims, without paying for MCMC."""
    import arviz as az
    import xarray as xr

    M, T, K = panel.X.shape
    C, D = 2, 20
    rng = np.random.default_rng(0)
    post = xr.Dataset(
        {
            "Rt": (("chain", "draw", "region", "region_time"), rng.gamma(4, 0.4, (C, D, M, T))),
            "infections": (("chain", "draw", "region", "region_time"),
                           rng.gamma(4, 50, (C, D, M, T))),
            "E_deaths": (("chain", "draw", "region", "region_time"),
                         rng.gamma(4, 20, (C, D, M, T))),
            "beta": (("chain", "draw", "npi"), rng.normal(-2, 0.2, (C, D, K))),
            "b": (("chain", "draw", "region", "npi"), rng.normal(0, 0.1, (C, D, M, K))),
            "b0": (("chain", "draw", "region"), rng.normal(0.3, 0.2, (C, D, M))),
        },
        coords={"region": panel.regions, "npi": panel.npis,
                "region_time": np.arange(T), "chain": np.arange(C), "draw": np.arange(D)},
    )
    return az.InferenceData(posterior=post)


def test_draws_preserves_region_axis(fake_multilevel_idata, panel):
    """_draws must not fold `region` into the draw axis (it silently did once)."""
    from epidemia.plots import _draws

    M, T, _ = panel.X.shape
    arr, dims = _draws(fake_multilevel_idata, "Rt")
    assert dims == ("region", "region_time")
    assert arr.shape == (2 * 20, M, T), "regions must stay a separate axis"


def test_plot_rt_multiregion_requires_data(fake_multilevel_idata):
    """Refuse to guess dates for a multi-region fit rather than plot nonsense."""
    with pytest.raises(ValueError, match="region"):
        epi.plots.plot_rt(fake_multilevel_idata, save=False)


def test_plot_obs_finds_E_deaths(fake_multilevel_idata, panel):
    """The multilevel model names it E_deaths, not E_obs; plot_obs must cope."""
    p = epi.plots.plot_obs(fake_multilevel_idata, data=panel, save=False)
    assert p is not None


def test_region_frame_uses_each_regions_own_dates(fake_multilevel_idata, panel):
    """Column t is a different date per region; padded days must be dropped."""
    from epidemia.plots import _region_frame

    band, med, keep = _region_frame(fake_multilevel_idata, "Rt", panel, (50, 95), None)
    assert keep == panel.regions
    for m, r in enumerate(panel.regions):
        n = int(panel.lengths[m])
        sub = med[med["region"] == r]
        assert len(sub) == n, f"{r}: expected its own {n} genuine days"
        assert sub["x"].min() == panel.dates[m][0]
        assert sub["x"].max() == panel.dates[m][n - 1]
    # Italy starts earlier than Norway -- if a common axis were assumed these
    # would coincide.
    starts = {r: med[med["region"] == r]["x"].min() for r in panel.regions}
    assert starts["Italy"] < starts["Norway"]


def test_group_selects_one_region(fake_multilevel_idata, panel):
    from epidemia.plots import _region_frame

    _, med, keep = _region_frame(fake_multilevel_idata, "Rt", panel, (50,), "Italy")
    assert keep == ["Italy"]
    assert set(med["region"]) == {"Italy"}


def test_unknown_group_raises(fake_multilevel_idata, panel):
    with pytest.raises(ValueError, match="unknown region"):
        epi.plots.plot_rt(fake_multilevel_idata, data=panel, group="Atlantis", save=False)


def test_plots_are_saved_by_default(fake_multilevel_idata, panel, tmp_path, monkeypatch):
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    epi.plots.plot_rt(fake_multilevel_idata, data=panel)
    assert (tmp_path / "rt.png").exists(), "plots must be written to disk by default"
    epi.plots.plot_rt(fake_multilevel_idata, data=panel, save="custom-name")
    assert (tmp_path / "custom-name.png").exists()


def test_save_false_writes_nothing(fake_multilevel_idata, panel, tmp_path, monkeypatch):
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    epi.plots.plot_rt(fake_multilevel_idata, data=panel, save=False)
    assert not list(tmp_path.glob("*.png"))


def test_effect_plots(fake_multilevel_idata, tmp_path, monkeypatch):
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    epi.plots.plot_effects(fake_multilevel_idata)
    epi.plots.plot_effects(fake_multilevel_idata, group="Italy")
    epi.plots.plot_region_effects(fake_multilevel_idata, "lockdown")
    for f in ("effects.png", "effects_Italy.png", "region_effects_lockdown.png"):
        assert (tmp_path / f).exists()


def test_effect_table_percentages(fake_multilevel_idata, panel):
    """% reduction must come from the scaled-logit counterfactual, not 1-exp(beta)."""
    ec = epi.europe_covid2()
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
    tab = epi.effect_table(fake_multilevel_idata, config, data=panel)

    assert set(tab["region"]) == set(panel.regions)
    pct = tab[tab["kind"] == "pct"]
    assert (pct["lo"] <= pct["median"]).all() and (pct["median"] <= pct["hi"]).all()

    # R_0 and R_t must lie inside the link's range (0, K)
    R = tab[tab["kind"] == "R"]
    assert (R["median"] > 0).all() and (R["median"] < config.R_link_K).all()

    # The reduction must describe the R_0 -> R_t drop it is derived from. The
    # median of a ratio is not the ratio of medians, so this only holds
    # approximately -- it is a sanity check on the direction and magnitude, not
    # an identity.
    for r in panel.regions:
        sub = tab[tab["region"] == r]
        r0 = float(sub[sub["term"] == "R_0 (no measures)"]["median"].iloc[0])
        ra = float(sub[sub["term"] == "R_t (all measures)"]["median"].iloc[0])
        p = float(sub[sub["term"] == "all measures"]["median"].iloc[0])
        assert abs(p - 100 * (1 - ra / r0)) < 5.0, r


def test_effect_table_flags_measures_a_region_never_used(fake_multilevel_idata, panel):
    """Sweden never locked down: its % is a counterfactual, and must say so."""
    ec = epi.europe_covid2()
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
    tab = epi.effect_table(fake_multilevel_idata, config, data=panel)
    lock = tab[(tab["kind"] == "pct") & (tab["term"] == "lockdown")].set_index("region")
    assert lock.loc["Sweden", "enacted"] is False
    assert lock.loc["Italy", "enacted"] is True
    assert lock.loc["Norway", "enacted"] is True
    # "all measures" is only measured where the region used them all
    allm = tab[(tab["kind"] == "pct") & (tab["term"] == "all measures")].set_index("region")
    assert allm.loc["Sweden", "enacted"] is False


def test_effect_table_without_data_leaves_enacted_unknown(fake_multilevel_idata):
    ec = epi.europe_covid2()
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
    tab = epi.effect_table(fake_multilevel_idata, config)
    assert tab["enacted"].isna().all()


def test_percent_effects_plot(fake_multilevel_idata, panel, tmp_path, monkeypatch):
    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    ec = epi.europe_covid2()
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
    epi.plots.plot_percent_effects(fake_multilevel_idata, config, data=panel)
    assert (tmp_path / "percent-effects.png").exists()
    epi.plots.plot_percent_effects(fake_multilevel_idata, config, data=panel,
                                   group="Italy")
    assert (tmp_path / "percent-effects_Italy.png").exists()


def test_single_population_path_still_works(tmp_path, monkeypatch):
    """The multi-region support must not break the single-population plots."""
    import arviz as az
    import xarray as xr

    monkeypatch.setenv("EPIDEMIA_FIGDIR", str(tmp_path))
    rng = np.random.default_rng(1)
    idata = az.InferenceData(posterior=xr.Dataset({
        "Rt": (("chain", "draw", "time"), rng.gamma(4, 0.4, (2, 20, 40))),
        "E_obs": (("chain", "draw", "time"), rng.gamma(4, 20, (2, 20, 40))),
        "infections": (("chain", "draw", "time"), rng.gamma(4, 50, (2, 20, 40))),
    }))
    epi.plots.plot_rt(idata, save="sp_rt")
    epi.plots.plot_infections(idata, save="sp_inf")
    epi.plots.plot_obs(idata, observed=rng.poisson(80, 40), save="sp_obs")
    assert (tmp_path / "sp_rt.png").exists()
    assert (tmp_path / "sp_obs.png").exists()
