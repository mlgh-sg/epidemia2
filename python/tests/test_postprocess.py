"""Post-processing helpers, checked against arithmetic done by hand."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import xarray as xr

from epidemia.core import EpiModelConfig, ObsModel, PanelData
from epidemia.postprocess import (
    extract_samples,
    posterior_infectious,
    posterior_linpred,
    prior_summary,
)

M, T, K = 2, 9, 1


def _fixture():
    rng = np.random.default_rng(0)
    post = xr.Dataset(
        {
            "Rt_unadj": (("chain", "draw", "region", "region_time"),
                         rng.random((2, 3, M, T)) + 1),
            "b0": (("chain", "draw", "region"), rng.normal(0, 0.1, (2, 3, M))),
            "beta": (("chain", "draw", "npi"), rng.normal(0, 0.1, (2, 3, K))),
            "b": (("chain", "draw", "region", "npi"),
                  rng.normal(0, 0.1, (2, 3, M, K))),
            "infections": (("chain", "draw", "region", "region_time"),
                           rng.random((2, 3, M, T)) * 100),
        },
        coords={"chain": [0, 1], "draw": range(3), "region": ["A", "B"],
                "npi": ["x"], "region_time": range(T)},
    )
    idata = az.InferenceData(posterior=post)
    panel = PanelData(X=rng.random((M, T, K)), lengths=np.full(M, T),
                      regions=["A", "B"], npis=["x"],
                      dates=[np.arange(T)] * M)
    config = EpiModelConfig(gen=np.array([0.5, 0.3, 0.2]))
    return idata, panel, config


def test_linear_predictor_is_reconstructed_exactly():
    idata, panel, config = _fixture()
    eta = posterior_linpred(idata, panel, config)

    post = idata.posterior
    b0 = np.asarray(post["b0"]).reshape(-1, M)
    beta = np.asarray(post["beta"]).reshape(-1, K)
    b = np.asarray(post["b"]).reshape(-1, M, K)
    expected = b0[:, :, None] + np.einsum(
        "mtk,smk->smt", panel.X, beta[:, None, :] + b)
    np.testing.assert_allclose(eta, expected)


def test_switching_a_component_off_removes_exactly_that_term():
    """R selects pp_eta_fe() and pp_eta_re() independently (R/pp_eta.R:26-63).

    The intercept is a column of the FIXED design, and the region deviations b
    sit with b0 in the RANDOM part -- so fixed=False must drop the intercept and
    keep Z b, and the two halves must add back to the whole.
    """
    idata, panel, config = _fixture()
    full = posterior_linpred(idata, panel, config)
    only_random = posterior_linpred(idata, panel, config, fixed=False)
    only_fixed = posterior_linpred(idata, panel, config, random=False)

    b0 = np.asarray(idata.posterior["b0"]).reshape(-1, M)
    b = np.asarray(idata.posterior["b"]).reshape(-1, M, K)
    beta = np.asarray(idata.posterior["beta"]).reshape(-1, K)

    want_random = b0[:, :, None] + np.einsum("mtk,smk->smt", panel.X, b)
    np.testing.assert_allclose(only_random, want_random)

    want_fixed = np.einsum("mtk,sk->smt", panel.X, beta)
    if "intercept" in idata.posterior:
        want_fixed = want_fixed + np.asarray(
            idata.posterior["intercept"]).reshape(-1)[:, None, None]
    np.testing.assert_allclose(only_fixed, want_fixed)

    # the decomposition is exact
    np.testing.assert_allclose(only_fixed + only_random, full)


def test_fixed_false_keeps_the_region_slope_deviations():
    """The old code folded b in under `fixed`, so fixed=False silently lost it."""
    idata, panel, config = _fixture()
    only_random = posterior_linpred(idata, panel, config, fixed=False)
    b0 = np.asarray(idata.posterior["b0"]).reshape(-1, M)
    assert not np.allclose(only_random,
                           np.broadcast_to(b0[:, :, None], only_random.shape))


def _hand_convolution(idata, config):
    inf = np.asarray(idata.posterior["infections"]).reshape(-1, M, T)
    want = np.zeros_like(inf)
    for k in (1, 2, 3):                     # gen is lag-1-first
        want[..., k:] += config.gen[k - 1] * inf[..., : T - k]
    return want


def test_infectious_matches_a_hand_computed_convolution():
    """R divides the convolution by max(gen) -- epidemia_pp_base.stan:47."""
    idata, panel, config = _fixture()
    want = _hand_convolution(idata, config) / np.max(config.gen)
    np.testing.assert_allclose(posterior_infectious(idata, config), want)


def test_infectious_can_return_the_raw_convolution():
    idata, panel, config = _fixture()
    np.testing.assert_allclose(
        posterior_infectious(idata, config, normalise=False),
        _hand_convolution(idata, config))


def test_infectious_normalisation_is_not_a_no_op():
    """max(gen) is well under 1 for a simplex kernel, so the factor is large."""
    idata, panel, config = _fixture()
    norm = posterior_infectious(idata, config)
    raw = posterior_infectious(idata, config, normalise=False)
    assert np.max(config.gen) < 1.0
    np.testing.assert_allclose(norm * np.max(config.gen), raw)
    assert norm.max() > raw.max()


def test_extract_samples_selects_by_name_and_regex():
    idata, _, _ = _fixture()
    assert list(extract_samples(idata, pars=["beta"]).columns) == ["beta[x]"]
    cols = extract_samples(idata, regex=r"^b").columns
    assert any(c.startswith("b0[") for c in cols)
    assert extract_samples(idata, pars=["beta"]).shape[0] == 6   # 2 chains x 3


def test_prior_summary_reports_defaults_and_overrides():
    from epidemia import priors as pr

    _, panel, config = _fixture()
    obs = ObsModel("deaths", np.zeros((M, T)), np.ones((M, T), bool),
                   np.array([0.5, 0.5]), link_K=0.02)

    default = repr(prior_summary(panel, [obs], config))
    assert "shifted_gamma" in default and "hexp" in default

    swapped = repr(prior_summary(
        panel, [obs],
        EpiModelConfig(gen=config.gen, prior_covariates=pr.hs(),
                       prior_seeds=pr.exponential(0.05))))
    assert "hs" in swapped and "shifted_gamma" not in swapped


def test_series_predictor_needs_the_obs_model():
    idata, panel, config = _fixture()
    with pytest.raises(ValueError, match="obs_models"):
        posterior_linpred(idata, panel, config, series="deaths")
