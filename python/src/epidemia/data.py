"""Example datasets.

``flu1918`` is the 1918 influenza pandemic in Baltimore, the same data used in
the R package's basic tutorial (originally from the EpiEstim package).

``europe_covid2`` is the ``EuropeCovid2`` dataset from the R package: daily case
and death counts plus NPI indicators for 11 European countries during the first
wave of COVID-19 (used in the multilevel / partial-pooling example).

``europe_covid`` is the original ``EuropeCovid`` dataset -- the deaths-only data
exactly as used in Flaxman et al. (2020), before the WHO revised the counts
retrospectively. ``england_new_cases`` is ``EnglandNewCases``, PHE "New Cases by
Specimen Date" for England.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np

# Daily case counts, Baltimore 1918 influenza pandemic (EpiEstim::Flu1918$incidence)
_FLU1918_INCIDENCE = np.array([
    5, 1, 6, 15, 2, 3, 8, 7, 2, 15, 4, 17, 4, 10, 31, 11, 13, 36, 13, 33, 17,
    15, 32, 27, 70, 58, 32, 69, 54, 80, 405, 192, 243, 204, 280, 229, 304, 265,
    196, 372, 158, 222, 141, 172, 553, 148, 95, 144, 85, 143, 87, 73, 70, 62,
    116, 44, 38, 60, 45, 60, 27, 51, 34, 22, 16, 11, 18, 11, 10, 8, 13, 3, 3, 6,
    6, 13, 5, 6, 6, 5, 5, 1, 2, 2, 3, 8, 4, 1, 2, 3, 1, 0,
], dtype=float)

# Serial-interval PMF (EpiEstim::Flu1918$si_distr): si[k] = P(serial interval = k days).
# si[0] == 0 (no same-day generation), so the renewal generation kernel drops it.
_FLU1918_SI = np.array([
    0.0, 0.233, 0.359, 0.198, 0.103, 0.053, 0.027, 0.014, 0.007, 0.003, 0.002,
    0.001,
], dtype=float)


@dataclass
class EpiData:
    """A simple container for an example epidemic series."""

    incidence: np.ndarray      # daily observed counts
    serial_interval: np.ndarray  # full SI PMF, si[k] = P(SI = k days)

    @property
    def generation(self) -> np.ndarray:
        """Renewal generation kernel: ``generation[k] = P(SI = k+1 days)``.

        This drops the (zero) same-day entry of the serial interval, so that
        ``generation`` aligns with :func:`epidemia.renewal.renewal_infections`,
        which weights the infection ``k+1`` days in the past by ``generation[k]``.
        """
        return self.serial_interval[1:]


def flu1918() -> EpiData:
    """Return the 1918 Baltimore influenza data (92 days) with its serial interval."""
    return EpiData(
        incidence=_FLU1918_INCIDENCE.copy(),
        serial_interval=_FLU1918_SI.copy(),
    )


# The five NPIs, in the order used throughout the multilevel example.
EUROPE_COVID_NPIS = [
    "schools_universities",
    "self_isolating_if_ill",
    "public_events",
    "social_distancing_encouraged",
    "lockdown",
]


@dataclass
class EuropeCovid2:
    """The ``EuropeCovid2`` dataset (11 countries, first COVID-19 wave).

    Attributes
    ----------
    data : pandas.DataFrame
        Long panel with columns ``id, country, date, cases, deaths, pop`` and the
        five binary NPI indicators in :data:`EUROPE_COVID_NPIS`. ``date`` is a
        ``datetime64`` column.
    si : numpy.ndarray
        Serial-interval PMF (``si[k] = P(serial interval = k days)``), summing to 1.
    inf2death : numpy.ndarray
        Infection-to-death delay PMF, summing to 1.
    """

    data: object
    si: np.ndarray
    inf2death: np.ndarray


def europe_covid2() -> EuropeCovid2:
    """Return the ``EuropeCovid2`` dataset used in the multilevel example.

    Daily case/death counts and NPI indicators for 11 European countries up to
    1 July 2020, together with the serial interval and infection-to-death delay
    distributions from Flaxman et al. (2020). Mirrors ``data("EuropeCovid2")`` in
    the R package.
    """
    import pandas as pd

    files = resources.files("epidemia.data_files")
    with resources.as_file(files / "europe_covid2.csv") as p:
        df = pd.read_csv(p, parse_dates=["date"])
    with resources.as_file(files / "europe_covid2_si.csv") as p:
        si = pd.read_csv(p)["si"].to_numpy(dtype=float).copy()
    with resources.as_file(files / "europe_covid2_inf2death.csv") as p:
        inf2death = pd.read_csv(p)["inf2death"].to_numpy(dtype=float).copy()
    return EuropeCovid2(data=df, si=si, inf2death=inf2death)


@dataclass
class EnglandB117:
    """SGTF-split case counts for England, autumn 2020 to January 2021.

    The data behind Volz et al. (2021), "Assessing transmissibility of
    SARS-CoV-2 lineage B.1.1.7 in England", *Nature* 593, 266-269. Mirrors
    ``data("EnglandB117")`` in the R package.

    Routine PCR testing in England used a three-target assay, and B.1.1.7
    carries a deletion that makes the S-gene target fail while the other two
    still amplify. "S-gene target failure" therefore acts as a proxy for the
    lineage without sequencing every sample.

    Attributes
    ----------
    data : pandas.DataFrame
        ``date, area, corrected_positive, corrected_negative, epiweek`` for 49
        areas over 120 days. ``corrected_negative`` is B.1.1.7 (S-gene
        negative); ``corrected_positive`` is everything else.
    pop : pandas.DataFrame
        ``area, pop`` for 42 of those areas. The other seven are NHS England
        *regions* -- aggregates of the rest -- which the source does not size,
        so they cannot be used with ``pop_adjust``.
    iar : pandas.DataFrame
        ``date, iar``: England-wide daily infection ascertainment rate.
    iar_sd : float
        Its standard deviation, used as the prior scale on the observation
        coefficient.
    i2o : numpy.ndarray
        Infection-to-observation kernel. The observations are weekly case
        totals attached to a daily series, so a daily delay distribution is
        spread over seven offsets and this sums to **7, not 1**.
    published : pandas.DataFrame
        The paper's own fitted estimates: ``area, epiweek, ratio, rt_b117,
        rt_other``. Shipped so a reproduction can check itself against the
        published numbers rather than against a value copied by hand.
    published_england : pandas.DataFrame
        ``epiweek, median, lower, upper``: the England-wide time-varying
        advantage, pooled across areas.
    """

    data: object
    pop: object
    iar: object
    iar_sd: float
    i2o: np.ndarray
    published: object
    published_england: object


def england_b117() -> EnglandB117:
    """Return the :class:`EnglandB117` dataset.

    The counts are aggregates. The raw SGSS line-list behind them is
    disclosure-controlled -- counts below five are suppressed at source -- so
    the raw-to-aggregate step cannot be reproduced outside PHE.
    """
    import pandas as pd

    files = resources.files("epidemia.data_files")

    def _csv(name, **kw):
        with resources.as_file(files / name) as p:
            return pd.read_csv(p, **kw)

    return EnglandB117(
        data=_csv("england_b117.csv", parse_dates=["date"]),
        pop=_csv("england_b117_pop.csv"),
        iar=_csv("england_b117_iar.csv", parse_dates=["date"]),
        iar_sd=float(_csv("england_b117_iar_sd.csv")["iar_sd"].iloc[0]),
        i2o=_csv("england_b117_i2o.csv")["i2o"].to_numpy(dtype=float).copy(),
        published=_csv("england_b117_published.csv"),
        published_england=_csv("england_b117_published_england.csv"),
    )


# The 11 countries in EuropeCovid, in the order R's factor levels put them. The
# factor actually carries 14 levels (Greece, Netherlands and Portugal are unused
# leftovers from the wider ECDC extract), so listing the observed ones here keeps
# callers from being surprised by empty groups.
EUROPE_COVID_COUNTRIES = [
    "Austria",
    "Belgium",
    "Denmark",
    "France",
    "Germany",
    "Italy",
    "Norway",
    "Spain",
    "Sweden",
    "Switzerland",
    "United_Kingdom",
]


@dataclass
class EuropeCovid:
    """The ``EuropeCovid`` dataset (Flaxman et al. 2020, 11 European countries).

    Attributes
    ----------
    data : pandas.DataFrame
        Long panel with columns ``country, date, deaths, pop`` and the five
        binary NPI indicators in :data:`EUROPE_COVID_NPIS`. ``date`` is a
        ``datetime64`` column. Unlike :class:`EuropeCovid2` there are no case
        counts: Flaxman et al. modelled deaths alone. Each country's first row is
        exactly 30 days before it recorded 10 cumulative deaths, so the panel is
        ragged (899 rows in total, running 2020-01-27 to 2020-05-05).
    si : numpy.ndarray
        Serial-interval PMF of length 100, summing to 1. It has no leading
        zero -- ``si[k] = P(serial interval = k+1 days)`` -- so it can be passed
        straight to :func:`epidemia.renewal.renewal_infections` as the
        generation kernel, unlike :attr:`EpiData.serial_interval`.
    inf2death : numpy.ndarray
        Infection-to-death delay PMF of length 101, summing to 1.
    """

    data: object
    si: np.ndarray
    inf2death: np.ndarray


def europe_covid() -> EuropeCovid:
    """Return the ``EuropeCovid`` dataset used in Flaxman et al. (2020).

    Daily death counts, NPI indicators and populations for 11 European countries
    up to 5 May 2020, together with the serial interval and infection-to-death
    delay distributions. Mirrors ``data("EuropeCovid")`` in the R package.

    Note that this is the *original* Flaxman data; :func:`europe_covid2` carries
    the retrospectively revised WHO counts plus case data, and is what the
    multilevel vignette uses.

    Returns
    -------
    EuropeCovid
        Panel, serial interval and infection-to-death delay.
    """
    import pandas as pd

    files = resources.files("epidemia.data_files")
    with resources.as_file(files / "europe_covid.csv") as p:
        df = pd.read_csv(p, parse_dates=["date"])
    # si/inf2death are shipped as their own CSVs (rather than reusing the
    # europe_covid2_* ones they currently duplicate) because R ships them as
    # separate objects and the two datasets are free to diverge.
    with resources.as_file(files / "europe_covid_si.csv") as p:
        si = pd.read_csv(p)["si"].to_numpy(dtype=float).copy()
    with resources.as_file(files / "europe_covid_inf2death.csv") as p:
        inf2death = pd.read_csv(p)["inf2death"].to_numpy(dtype=float).copy()
    return EuropeCovid(data=df, si=si, inf2death=inf2death)


def england_new_cases():
    """Return the ``EnglandNewCases`` dataset as a :class:`pandas.DataFrame`.

    SARS-CoV-2 "New Cases by Specimen Date" for England as published by Public
    Health England, downloaded 2021-06-01. 487 rows covering 2020-01-30 to
    2021-05-30, with columns ``date`` (``datetime64``), ``region`` (always
    ``"England"``) and ``cases``.

    Counts in the last few days of May 2021 are likely under-reported: not every
    specimen had been counted by the download date.

    Returns
    -------
    pandas.DataFrame
        One row per day.
    """
    import pandas as pd

    files = resources.files("epidemia.data_files")
    with resources.as_file(files / "england_new_cases.csv") as p:
        return pd.read_csv(p, parse_dates=["date"])
