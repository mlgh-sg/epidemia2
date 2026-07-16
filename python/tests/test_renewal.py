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
    inf = np.array([1.0, 0.0, 0.0, 0.0])
    i2o = np.array([0.5, 0.3, 0.2])  # same-day, 1-day, 2-day delay
    y = expected_observations(inf, i2o, 1.0)
    assert np.allclose(y, [0.5, 0.3, 0.2, 0.0])


def test_ascertainment_scales_observations():
    inf = np.ones(4)
    i2o = np.array([1.0])
    y = expected_observations(inf, i2o, 0.1)
    assert np.allclose(y, [0.1, 0.1, 0.1, 0.1])


def test_non_centered_random_walk():
    walk = random_walk(0.5, np.array([1.0, 1.0, 1.0]), 2.0)
    assert np.allclose(walk, [2.5, 3.0, 3.5])
