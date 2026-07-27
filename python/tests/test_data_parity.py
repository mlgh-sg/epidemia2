"""Parity checks for the shipped example datasets against R's documentation.

The numbers asserted here are the ones R's ``?EuropeCovid`` / ``?EnglandNewCases``
help pages and the underlying ``.RData`` objects state, so a drift in the CSVs
under ``src/epidemia/data_files/`` shows up as a test failure rather than as a
silently different model fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epidemia.data import (
    EUROPE_COVID_COUNTRIES,
    EUROPE_COVID_NPIS,
    england_new_cases,
    europe_covid,
    europe_covid2,
    flu1918,
)


# ---------------------------------------------------------------- EuropeCovid

def test_europe_covid_shape_and_columns():
    ec = europe_covid()
    df = ec.data
    assert isinstance(df, pd.DataFrame)
    # 899 region-days, ragged because each country starts 30 days before its
    # 10th cumulative death.
    assert df.shape == (899, 9)
    assert list(df.columns) == [
        "country",
        "date",
        "deaths",
        *EUROPE_COVID_NPIS,
        "pop",
    ][:3] + [
        "schools_universities",
        "self_isolating_if_ill",
        "public_events",
        "lockdown",
        "social_distancing_encouraged",
        "pop",
    ][:6] or set(df.columns) == {
        "country", "date", "deaths", "pop", *EUROPE_COVID_NPIS,
    }
    # The above is order-tolerant on the NPI block; the set check is the contract.
    assert set(df.columns) == {"country", "date", "deaths", "pop", *EUROPE_COVID_NPIS}


def test_europe_covid_regions():
    df = europe_covid().data
    assert sorted(df["country"].unique()) == sorted(EUROPE_COVID_COUNTRIES)
    assert len(EUROPE_COVID_COUNTRIES) == 11


def test_europe_covid_dates():
    df = europe_covid().data
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # R's help page: deaths "up until 05/05/2020".
    assert df["date"].min() == pd.Timestamp("2020-01-27")
    assert df["date"].max() == pd.Timestamp("2020-05-05")
    # Every country's series is contiguous daily.
    for _, g in df.groupby("country"):
        d = g["date"].sort_values()
        assert (d.diff().dropna() == pd.Timedelta(days=1)).all()


def test_europe_covid_npis_are_binary_and_deaths_nonnegative():
    df = europe_covid().data
    for npi in EUROPE_COVID_NPIS:
        assert set(np.unique(df[npi])) <= {0, 1}
    assert (df["deaths"] >= 0).all()
    assert df["deaths"].isna().sum() == 0


def test_europe_covid_pop_constant_per_country():
    df = europe_covid().data
    assert df["pop"].isna().sum() == 0
    assert (df.groupby("country")["pop"].nunique() == 1).all()
    pops = df.groupby("country")["pop"].first()
    assert pops["United_Kingdom"] == 67886004
    assert pops["Austria"] == 9006400


def test_europe_covid_kernels():
    ec = europe_covid()
    assert ec.si.shape == (100,)
    assert ec.inf2death.shape == (101,)
    for k in (ec.si, ec.inf2death):
        assert np.all(k >= 0)
        assert k.sum() == pytest.approx(1.0, abs=1e-8)
    # No leading zero: si[0] is P(serial interval = 1 day), so it is already the
    # renewal generation kernel.
    assert ec.si[0] > 0


def test_europe_covid_idempotent():
    a, b = europe_covid(), europe_covid()
    pd.testing.assert_frame_equal(a.data, b.data)
    np.testing.assert_array_equal(a.si, b.si)
    np.testing.assert_array_equal(a.inf2death, b.inf2death)
    # Mutating one copy must not leak into the next call.
    a.data.loc[0, "deaths"] = -999
    a.si[0] = -1.0
    c = europe_covid()
    assert c.data.loc[0, "deaths"] >= 0
    assert c.si[0] > 0


def test_europe_covid_distinct_from_europe_covid2():
    """The two datasets are different vintages of the same countries."""
    ec, ec2 = europe_covid(), europe_covid2()
    assert "cases" not in ec.data.columns  # Flaxman modelled deaths only
    assert "cases" in ec2.data.columns
    assert len(ec.data) != len(ec2.data)
    assert ec.data["date"].max() < ec2.data["date"].max()


# ------------------------------------------------------------ EnglandNewCases

def test_england_new_cases_shape_and_columns():
    df = england_new_cases()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (487, 3)
    assert list(df.columns) == ["date", "region", "cases"]


def test_england_new_cases_dates_and_region():
    df = england_new_cases()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].min() == pd.Timestamp("2020-01-30")
    assert df["date"].max() == pd.Timestamp("2021-05-30")
    # 487 daily rows with no gaps and no duplicates.
    assert df["date"].is_unique
    assert (df["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert (df["date"].max() - df["date"].min()).days + 1 == len(df)
    assert list(df["region"].unique()) == ["England"]


def test_england_new_cases_counts():
    df = england_new_cases()
    assert (df["cases"] >= 0).all()
    assert df["cases"].isna().sum() == 0
    assert df["cases"].iloc[0] == 2  # 2020-01-30


def test_england_new_cases_idempotent():
    a, b = england_new_cases(), england_new_cases()
    pd.testing.assert_frame_equal(a, b)
    a.loc[0, "cases"] = -999
    assert england_new_cases().loc[0, "cases"] == 2


# ------------------------------------------------ existing datasets unchanged

def test_flu1918_and_europe_covid2_still_load():
    """Guard the additive edit: the two pre-existing loaders are untouched."""
    flu = flu1918()
    assert flu.incidence.shape == (92,)
    assert flu.generation.shape == (11,)
    ec2 = europe_covid2()
    assert set(EUROPE_COVID_NPIS) <= set(ec2.data.columns)
    assert ec2.si.sum() == pytest.approx(1.0, abs=1e-8)
