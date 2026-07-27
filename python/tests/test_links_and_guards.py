"""Link functions, and the guards that stop silently-wrong output.

Both of these were audit findings: the links were hardcoded so R's *default*
R_t link could not be expressed, and forecast() documented a limitation it did
not enforce.
"""

from __future__ import annotations

import numpy as np
import pytest

from epidemia.core import OBS_LINKS, R_LINKS, EpiModelConfig, ObsModel, PanelData
from epidemia.core import build_epidemia_model
from epidemia.forecast import _apply_link

M, T, K = 3, 40, 2
SEED_DAYS = 6


def _gen():
    g = np.exp(-np.arange(1, 21) / 5.0)
    return g / g.sum()


def _panel():
    rng = np.random.default_rng(0)
    return PanelData(X=rng.integers(0, 2, (M, T, K)).astype(float),
                     lengths=np.full(M, T), regions=list("ABC"),
                     npis=["a", "b"], dates=[np.arange(T)] * M,
                     pops=np.array([8e6, 6e7, 8.3e7]))


def _obs(link="scaled_logit"):
    rng = np.random.default_rng(1)
    mask = np.ones((M, T), bool)
    mask[:, :SEED_DAYS] = False
    i2o = np.exp(-((np.arange(1, 31) - 18) ** 2) / 40.0)
    return ObsModel("deaths", rng.poisson(20, (M, T)), mask, i2o / i2o.sum(),
                    family="neg_binom", link=link, link_K=0.02)


@pytest.mark.parametrize("obs_link", OBS_LINKS)
def test_every_observation_link_builds(obs_link):
    model = build_epidemia_model(
        _panel(), _obs(obs_link),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS))
    assert np.isfinite(float(model.compile_logp()(model.initial_point())))


def test_unknown_links_are_rejected_by_name():
    with pytest.raises(ValueError, match="unknown link"):
        build_epidemia_model(_panel(), _obs(),
                             EpiModelConfig(gen=_gen(), link="cloglog"))
    with pytest.raises(ValueError, match="unknown link"):
        build_epidemia_model(_panel(), _obs("log"),
                             EpiModelConfig(gen=_gen()))


def test_the_numpy_links_agree_with_their_closed_forms():
    """forecast's NumPy links must match core's PyTensor ones, or a forecast
    silently disagrees with the fit it came from."""
    eta = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(_apply_link("log", eta, 1.0), np.exp(eta))
    assert np.allclose(_apply_link("identity", eta, 1.0), eta)
    assert np.allclose(_apply_link("logit", eta, 1.0), 1 / (1 + np.exp(-eta)))
    assert np.allclose(_apply_link("scaled_logit", eta, 3.0),
                       3.0 / (1 + np.exp(-eta)))
    # probit / cauchit / cloglog at known points
    assert np.isclose(_apply_link("probit", np.array([0.0]), 1.0)[0], 0.5)
    assert np.isclose(_apply_link("cauchit", np.array([1.0]), 1.0)[0], 0.75)
    assert np.isclose(_apply_link("cloglog", np.array([0.0]), 1.0)[0],
                      1 - np.exp(-1))
    with pytest.raises(ValueError, match="unknown link"):
        _apply_link("nonsense", eta, 1.0)


def test_r_link_set_matches_R():
    """R's epirt accepts exactly these three; log is R's default."""
    assert set(R_LINKS) == {"log", "identity", "scaled_logit"}
    assert set(OBS_LINKS) == {"logit", "probit", "cauchit", "cloglog",
                              "identity", "scaled_logit"}
