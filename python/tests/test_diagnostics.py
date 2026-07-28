"""Tests for the sampler-diagnostics summary."""

import warnings

import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData

from epidemia.diagnostics import sampler_diagnostics


def _idata(diverging, maxdepth=None, energy=None, posterior=True):
    """Build a minimal InferenceData with the sample_stats a NUTS fit produces."""
    diverging = np.asarray(diverging, dtype=bool)
    nchain, ndraw = diverging.shape
    stats = {"diverging": (("chain", "draw"), diverging)}
    if maxdepth is not None:
        stats["maxdepth_reached"] = (("chain", "draw"), np.asarray(maxdepth, dtype=bool))
    if energy is not None:
        stats["energy"] = (("chain", "draw"), np.asarray(energy, dtype=float))
    coords = {"chain": np.arange(nchain), "draw": np.arange(ndraw)}
    groups = {"sample_stats": xr.Dataset(stats, coords=coords)}
    if posterior:
        rng = np.random.default_rng(0)
        groups["posterior"] = xr.Dataset(
            {"theta": (("chain", "draw"), rng.normal(size=(nchain, ndraw)))},
            coords=coords,
        )
    return InferenceData(**groups)


def test_counts_divergences_and_treedepth_per_chain():
    div = [[True, False, False, False], [False, False, True, True]]
    td = [[False, True, False, False], [False, False, False, False]]
    d = sampler_diagnostics(_idata(div, maxdepth=td))

    assert d.divergences == 3
    assert d.max_treedepth_hits == 1
    assert list(d.per_chain["divergent"]) == [1, 2]
    assert list(d.per_chain["max_treedepth"]) == [1, 0]
    assert list(d.per_chain["chain"]) == [1, 2]


def test_a_clean_fit_reports_no_problems():
    n = 400
    rng = np.random.default_rng(1)
    d = sampler_diagnostics(_idata(
        np.zeros((4, n), dtype=bool),
        maxdepth=np.zeros((4, n), dtype=bool),
        energy=rng.normal(size=(4, n)),
    ))
    assert d.divergences == 0
    assert d.max_treedepth_hits == 0
    # theta is iid normal, so R-hat is ~1 and ESS is large: nothing to flag.
    assert d.ok, d.problems


def test_divergences_are_reported_as_a_problem():
    d = sampler_diagnostics(_idata([[True, False], [False, False]]))
    assert not d.ok
    assert any("divergent transition" in p for p in d.problems)
    # a divergence is a bias problem, so the advice must not be "draw more"
    assert any("more draws will not help" in p for p in d.problems)


def test_treedepth_is_flagged_as_efficiency_not_correctness():
    d = sampler_diagnostics(_idata(
        np.zeros((2, 4), dtype=bool), maxdepth=np.ones((2, 4), dtype=bool)))
    msg = " ".join(d.problems)
    assert "max_treedepth" in msg
    assert "efficiency rather than correctness" in msg


def test_low_ebfmi_is_flagged():
    # A slowly drifting energy series has tiny successive differences relative
    # to its variance, which is exactly the low-E-BFMI signature.
    energy = np.linspace(0, 100, 400)[None, :].repeat(2, axis=0)
    d = sampler_diagnostics(_idata(
        np.zeros((2, 400), dtype=bool), energy=energy))
    assert d.per_chain["ebfmi"].min() < 0.2
    assert any("E-BFMI" in p for p in d.problems)


def test_missing_treedepth_variable_is_tolerated():
    # PyMC and nutpie disagree on the name, and some samplers omit it entirely.
    d = sampler_diagnostics(_idata([[False, True]]))
    assert d.max_treedepth_hits == 0


def test_pymc_style_names_are_accepted():
    div = np.array([[True, False, False]])
    coords = {"chain": [0], "draw": [0, 1, 2]}
    idata = InferenceData(sample_stats=xr.Dataset(
        {"divergent": (("chain", "draw"), div),
         "reached_max_treedepth": (("chain", "draw"), np.ones((1, 3), dtype=bool))},
        coords=coords))
    d = sampler_diagnostics(idata)
    assert d.divergences == 1
    assert d.max_treedepth_hits == 3


def test_variational_fit_raises_a_directed_error():
    idata = InferenceData(posterior=xr.Dataset(
        {"theta": (("chain", "draw"), np.zeros((1, 5)))},
        coords={"chain": [0], "draw": range(5)}))
    with pytest.raises(ValueError, match="no NUTS diagnostics|sample_stats"):
        sampler_diagnostics(idata)


def test_warn_emits_one_warning_per_problem():
    with pytest.warns(UserWarning, match="divergent transition"):
        sampler_diagnostics(_idata([[True, False]]), warn=True)


def test_repr_shows_the_totals():
    text = repr(sampler_diagnostics(_idata([[True, False], [False, True]])))
    assert "Sampler diagnostics" in text
    assert "Divergent transitions: 2" in text
    assert "2 chains x 2 post-warmup draws" in text


def test_fit_time_hook_warns_on_divergences():
    """The hook epidemia.fit/fit_multilevel/fit_epidemia call after sampling."""
    from epidemia.multilevel import _warn_on_divergences

    with pytest.warns(UserWarning, match="divergent transition"):
        _warn_on_divergences(_idata([[True, False], [False, False]]))


def test_fit_time_hook_is_silent_when_there_is_nothing_to_report():
    from epidemia.multilevel import _warn_on_divergences

    rng = np.random.default_rng(2)
    clean = _idata(np.zeros((4, 400), dtype=bool),
                   maxdepth=np.zeros((4, 400), dtype=bool),
                   energy=rng.normal(size=(4, 400)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes a failure
        _warn_on_divergences(clean)


def test_fit_time_hook_tolerates_a_fit_with_no_sample_stats():
    """Variational fits have no NUTS diagnostics; that is not an error at fit time."""
    from epidemia.multilevel import _warn_on_divergences

    vi = InferenceData(posterior=xr.Dataset(
        {"theta": (("chain", "draw"), np.zeros((1, 5)))},
        coords={"chain": [0], "draw": range(5)}))
    _warn_on_divergences(vi)   # must not raise
