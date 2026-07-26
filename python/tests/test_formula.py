"""Tests for the formula mini-language in :mod:`epidemia.formula`.

Parsing is checked shape by shape against the R syntax (``R/epirt.R``,
``R/autocor.R``, ``vignettes/partial-pooling.Rmd``); the glue is checked by
comparing against the ``prepare_panel`` call one would have written by hand.
Nothing here fits a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from epidemia.core import EpiModelConfig, RandomWalk, prepare_panel
from epidemia.formula import (
    FormulaSpec,
    RandomEffect,
    RwTerm,
    build_from_formula,
    parse_formula,
)


# ---------------------------------------------------------------------------
# left hand side
# ---------------------------------------------------------------------------


def test_rt_sugar_on_lhs():
    spec = parse_formula("R(country, date) ~ 1 + lockdown + public_events")
    assert (spec.response, spec.group, spec.date) == ("R", "country", "date")
    assert spec.is_rt
    assert spec.intercept
    assert spec.fixed == ["lockdown", "public_events"]
    assert spec.random == [] and spec.rw == []
    assert spec.covariates == ["lockdown", "public_events"]


def test_observation_formula_has_no_sugar():
    spec = parse_formula("deaths ~ 1")
    assert (spec.response, spec.group, spec.date) == ("deaths", None, None)
    assert not spec.is_rt
    assert spec.intercept
    assert spec.fixed == []


def test_no_intercept_with_factor_term():
    spec = parse_formula("cases ~ 0 + country")
    assert spec.response == "cases"
    assert spec.intercept is False
    assert spec.fixed == ["country"]


# ---------------------------------------------------------------------------
# group-specific terms
# ---------------------------------------------------------------------------


def test_independent_random_effects_drop_intercept():
    spec = parse_formula(
        "R(country, date) ~ 0 + (1 + lockdown || country) + lockdown"
    )
    assert spec.intercept is False
    assert spec.fixed == ["lockdown"]
    assert spec.random == [
        RandomEffect(terms=["1", "lockdown"], factor="country", correlated=False)
    ]
    assert spec.correlated is False
    assert spec.random[0].intercept is True
    assert spec.random[0].covariates == ["lockdown"]
    # one design matrix serves both the fixed and the group-level effects
    assert spec.covariates == ["lockdown"]


def test_correlated_random_effects():
    spec = parse_formula("R(country, date) ~ 1 + (1 + lockdown | country)")
    assert spec.intercept is True
    assert spec.fixed == []
    assert spec.random == [
        RandomEffect(terms=["1", "lockdown"], factor="country", correlated=True)
    ]
    assert spec.correlated is True


def test_intercept_inside_bar_is_implicit():
    # R parses `expr` into a model matrix, so (lockdown | country) means
    # (1 + lockdown | country) -- see vignettes/partial-pooling.Rmd.
    spec = parse_formula("R(region, date) ~ lockdown + (lockdown | region)")
    assert spec.random[0].terms == ["1", "lockdown"]


@pytest.mark.parametrize("expr", ["0 + lockdown", "-1 + lockdown", "lockdown - 1"])
def test_intercept_inside_bar_can_be_removed(expr):
    spec = parse_formula(f"R(region, date) ~ 1 + ({expr} | region)")
    assert spec.random[0].terms == ["lockdown"]
    assert spec.random[0].intercept is False
    assert spec.intercept is True          # the bar must not touch the global one


def test_several_bars_and_covariate_ordering():
    spec = parse_formula(
        "R(region, date) ~ 1 + lockdown + (1 | region) + (0 + schools || region)"
    )
    assert [r.factor for r in spec.random] == ["region", "region"]
    assert [r.correlated for r in spec.random] == [True, False]
    assert spec.correlated is True         # any single bar makes it correlated
    assert spec.covariates == ["lockdown", "schools"]


@pytest.mark.parametrize(
    "text",
    ["R(region, date) ~ lockdown - 1", "R(region, date) ~ -1 + lockdown",
     "R(region, date) ~ 0 + lockdown"],
)
def test_intercept_removal_spellings(text):
    spec = parse_formula(text)
    assert spec.intercept is False
    assert spec.fixed == ["lockdown"]


# ---------------------------------------------------------------------------
# rw() terms
# ---------------------------------------------------------------------------


def test_rw_with_time_only():
    spec = parse_formula("R(country, date) ~ 1 + rw(time = week) + lockdown")
    assert spec.rw == [RwTerm(time="week", gr=None, prior_scale=0.2)]
    assert spec.fixed == ["lockdown"]
    assert spec.intercept is True


def test_rw_with_time_and_group():
    spec = parse_formula("R(country, date) ~ 1 + rw(time = week, gr = country)")
    assert spec.rw == [RwTerm(time="week", gr="country", prior_scale=0.2)]
    assert spec.fixed == []


@pytest.mark.parametrize(
    "call,expected",
    [
        ("rw()", RwTerm(None, None, 0.2)),
        ("rw(week)", RwTerm("week", None, 0.2)),
        ("rw(week, country)", RwTerm("week", "country", 0.2)),
        ("rw(gr = country)", RwTerm(None, "country", 0.2)),
        ("rw(time = week, prior_scale = 0.05)", RwTerm("week", None, 0.05)),
    ],
)
def test_rw_argument_forms(call, expected):
    assert parse_formula(f"R(country, date) ~ 1 + {call}").rw == [expected]


def test_rw_terms_are_not_fixed_effects():
    spec = parse_formula("R(country, date) ~ rw(time = week)")
    assert spec.fixed == []
    assert spec.covariates == []


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_missing_tilde():
    with pytest.raises(ValueError, match="no '~'"):
        parse_formula("R(country, date) + lockdown")


@pytest.mark.parametrize(
    "text",
    ["R(country) ~ 1", "R(country, date, week) ~ 1", "R() ~ 1",
     "R(country date) ~ 1"],
)
def test_malformed_lhs_sugar(text):
    with pytest.raises(ValueError, match="R\\(group, date\\)"):
        parse_formula(text)


def test_lhs_must_be_a_name():
    with pytest.raises(ValueError, match="column name"):
        parse_formula("log(deaths) ~ 1")


def test_empty_sides():
    with pytest.raises(ValueError, match="left hand side"):
        parse_formula("~ 1")
    with pytest.raises(ValueError, match="right hand side"):
        parse_formula("deaths ~")


def test_unknown_call_on_rhs():
    with pytest.raises(ValueError, match="unknown function 's'"):
        parse_formula("R(country, date) ~ 1 + s(week)")


def test_duplicate_group_specific_terms():
    with pytest.raises(ValueError, match="duplicate group-specific term"):
        parse_formula("R(country, date) ~ 1 + (1 | country) + (1 | country)")


def test_duplicate_group_specific_terms_with_covariates():
    with pytest.raises(ValueError, match="duplicate group-specific term"):
        parse_formula(
            "R(c, date) ~ (1 + lockdown | c) + (lockdown | c)"   # same expansion
        )


def test_distinct_bars_are_fine():
    spec = parse_formula("R(c, date) ~ (1 | c) + (0 + lockdown | c)")
    assert len(spec.random) == 2


def test_unbalanced_parentheses():
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        parse_formula("R(country, date) ~ 1 + (1 | country")


def test_malformed_bar():
    with pytest.raises(ValueError, match="malformed group-specific term"):
        parse_formula("R(country, date) ~ 1 + (1 + lockdown)")


def test_bad_grouping_factor():
    with pytest.raises(ValueError, match="nested or interacted"):
        parse_formula("R(district, date) ~ (1 | county / district)")


def test_stray_plus():
    with pytest.raises(ValueError, match="empty term"):
        parse_formula("R(country, date) ~ 1 + + lockdown")


def test_unparseable_term():
    with pytest.raises(ValueError, match="cannot parse term"):
        parse_formula("R(country, date) ~ 1 + 2.5")


def test_rw_rejects_unknown_argument():
    with pytest.raises(ValueError, match="unknown argument 'by'"):
        parse_formula("R(country, date) ~ rw(by = week)")


def test_rw_prior_scale_must_be_numeric():
    with pytest.raises(ValueError, match="must be a number"):
        parse_formula("R(country, date) ~ rw(prior_scale = week)")


def test_rw_time_must_be_a_column_name():
    with pytest.raises(ValueError, match="must be a column name"):
        parse_formula("R(country, date) ~ rw(time = 3)")


def test_rw_inside_bar_is_rejected():
    with pytest.raises(ValueError, match="rw\\(\\) inside"):
        parse_formula("R(country, date) ~ (rw(time = week) | country)")


def test_formula_must_be_a_string():
    with pytest.raises(TypeError):
        parse_formula(3)


# ---------------------------------------------------------------------------
# build_from_formula
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def europe():
    from epidemia import europe_covid2

    df = europe_covid2().data.copy()
    # rw(time = week) needs a step index; ISO week is what the R vignettes use.
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def test_build_matches_hand_written_call(europe):
    panel, series, cfg = build_from_formula(
        europe,
        "R(country, date) ~ 0 + (1 + lockdown || country) + lockdown "
        "+ public_events",
        responses=["deaths"],
        pop="pop",
    )
    want_panel, want_series = prepare_panel(
        europe, npis=["lockdown", "public_events"], responses=["deaths"],
        group="country", date="date", pop="pop", seed_offset=30, threshold=10,
    )

    assert panel.npis == ["lockdown", "public_events"]
    np.testing.assert_array_equal(panel.X, want_panel.X)
    np.testing.assert_array_equal(panel.lengths, want_panel.lengths)
    assert panel.regions == want_panel.regions
    np.testing.assert_array_equal(panel.pops, want_panel.pops)
    np.testing.assert_array_equal(series["deaths"]["y"], want_series["deaths"]["y"])
    np.testing.assert_array_equal(
        series["deaths"]["mask"], want_series["deaths"]["mask"]
    )

    M, T, K = panel.X.shape
    assert (M, K) == (len(panel.regions), 2)
    assert cfg == {"intercept": False, "correlated": False}
    # the kwargs must drop straight into a config
    config = EpiModelConfig(gen=np.ones(3) / 3, **cfg)
    assert config.intercept is False and config.correlated is False
    assert config.rw is None


def test_build_correlated_flag(europe):
    _, _, cfg = build_from_formula(
        europe, "R(country, date) ~ 1 + (1 + lockdown | country)",
        responses=["deaths"],
    )
    assert cfg == {"intercept": True, "correlated": True}


def test_build_shared_random_walk(europe):
    panel, series, cfg = build_from_formula(
        europe, "R(country, date) ~ 1 + rw(time = week) + lockdown",
        responses=["deaths", "cases"], pop="pop",
    )
    M, T, K = panel.X.shape
    assert K == 1 and panel.npis == ["lockdown"]
    assert set(series) == {"deaths", "cases"}
    assert series["cases"]["y"].shape == (M, T)

    rw = cfg["rw"]
    assert isinstance(rw, RandomWalk)
    assert rw.by_region is False           # no gr= means one shared walk
    assert rw.prior_scale == 0.2
    assert rw.index.shape == (M, T)
    np.testing.assert_array_equal(rw.index, panel.rw_index)
    assert rw.index.min() == 0
    # weekly steps: far fewer than one per day, and non-decreasing within a row
    assert rw.index.max() + 1 < T
    assert (np.diff(rw.index, axis=1) >= 0).all()
    assert cfg["intercept"] is True and cfg["correlated"] is False


def test_build_walk_per_group(europe):
    _, _, cfg = build_from_formula(
        europe, "R(country, date) ~ 1 + rw(time = week, gr = country)",
        responses=["deaths"],
    )
    assert cfg["rw"].by_region is True


def test_build_daily_walk_defaults_to_date(europe):
    # R's rw() with no `time` uses the date column implied by the formula.
    panel, _, cfg = build_from_formula(
        europe, "R(country, date) ~ 1 + rw()", responses=["deaths"],
    )
    M, T, _ = panel.X.shape
    assert cfg["rw"].index.max() + 1 == T   # one step per modelled day


def test_build_prior_scale_is_carried_through(europe):
    _, _, cfg = build_from_formula(
        europe, "R(country, date) ~ rw(time = week, prior_scale = 0.05)",
        responses=["deaths"],
    )
    assert cfg["rw"].prior_scale == 0.05


def test_build_expands_a_factor_covariate(europe):
    # `~ 0 + country` is R's unpooled intercept: one indicator per country.
    panel, _, cfg = build_from_formula(
        europe, "R(country, date) ~ 0 + country", responses=["deaths"],
    )
    M, T, K = panel.X.shape
    assert K == M                            # no intercept: every level kept
    assert panel.npis == [f"country{c}" for c in sorted(panel.regions)]
    # exactly one indicator on per region-day
    assert (panel.X.sum(axis=2)[:, 0] == 1).all()
    assert cfg["intercept"] is False


def test_build_factor_with_intercept_drops_baseline(europe):
    panel, _, _ = build_from_formula(
        europe, "R(country, date) ~ 1 + country", responses=["deaths"],
    )
    M, _, K = panel.X.shape
    assert K == M - 1                        # treatment contrast


def test_build_needs_group_and_date(europe):
    with pytest.raises(ValueError, match="R\\(group, date\\) sugar"):
        build_from_formula(europe, "deaths ~ 1", responses=["deaths"])

    panel, _, cfg = build_from_formula(
        europe, "deaths ~ 1", responses=["deaths"], group="country", date="date",
    )
    assert panel.X.shape[2] == 0
    assert cfg == {"intercept": True, "correlated": False}


def test_build_rejects_unknown_column(europe):
    with pytest.raises(ValueError, match="not in the data"):
        build_from_formula(
            europe, "R(country, date) ~ 1 + nonesuch", responses=["deaths"],
        )


def test_build_rejects_two_rw_terms(europe):
    with pytest.raises(ValueError, match="at most one rw"):
        build_from_formula(
            europe,
            "R(country, date) ~ rw(time = week) + rw(time = date)",
            responses=["deaths"],
        )


def test_build_rejects_mismatched_rw_group(europe):
    with pytest.raises(ValueError, match="does not match the modelled group"):
        build_from_formula(
            europe, "R(country, date) ~ rw(time = week, gr = continent)",
            responses=["deaths"],
        )


def test_build_accepts_a_prespecified_spec(europe):
    spec = parse_formula("R(country, date) ~ 1 + lockdown")
    assert isinstance(spec, FormulaSpec)
    panel, _, cfg = build_from_formula(europe, spec, responses=["deaths"])
    assert panel.npis == ["lockdown"]
    assert cfg["intercept"] is True


def test_build_passes_through_prepare_panel_options(europe):
    panel, _, _ = build_from_formula(
        europe, "R(country, date) ~ 1 + lockdown", responses=["deaths"],
        threshold_on="cases", threshold=100, seed_offset=10,
        fit_until="2020-05-01",
    )
    want, _ = prepare_panel(
        europe, npis=["lockdown"], responses=["deaths"], group="country",
        date="date", threshold_on="cases", threshold=100, seed_offset=10,
        fit_until="2020-05-01",
    )
    np.testing.assert_array_equal(panel.lengths, want.lengths)


def test_build_does_not_mutate_the_input(europe):
    before = list(europe.columns)
    build_from_formula(europe, "R(country, date) ~ 0 + country",
                       responses=["deaths"])
    assert list(europe.columns) == before
