"""Example datasets.

``flu1918`` is the 1918 influenza pandemic in Baltimore, the same data used in
the R package's basic tutorial (originally from the EpiEstim package).

``europe_covid2`` is the ``EuropeCovid2`` dataset from the R package: daily case
and death counts plus NPI indicators for 11 European countries during the first
wave of COVID-19 (used in the multilevel / partial-pooling example).
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
        si = pd.read_csv(p)["si"].to_numpy(dtype=float)
    with resources.as_file(files / "europe_covid2_inf2death.csv") as p:
        inf2death = pd.read_csv(p)["inf2death"].to_numpy(dtype=float)
    return EuropeCovid2(data=df, si=si, inf2death=inf2death)
