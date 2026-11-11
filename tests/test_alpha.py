import numpy as np
import pandas as pd
import xarray as xr

from radarwow.estimation import ZDRSlopeAlpha, alpha_from_slope, zdr_slope


def _gates(k, n=200_000, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(15, 55, n)
    zdr = offset + k * (z - 20) + rng.normal(0, 0.3, n)
    return z, zdr


def test_slope_recovered_and_offset_ignored():
    z, zdr = _gates(0.05)
    assert abs(zdr_slope(z, zdr) - 0.05) < 0.003
    z, zdr = _gates(0.05, offset=0.5)
    assert abs(zdr_slope(z, zdr) - 0.05) < 0.003


def test_alpha_bounds_and_regimes():
    assert alpha_from_slope(0.06) == 0.015  # continental, big drops
    assert alpha_from_slope(0.0) == 0.04  # tropical, small drops
    assert np.isclose(alpha_from_slope(0.026), 0.04875 - 0.75 * 0.026)
    assert alpha_from_slope(np.nan) == 0.015


def test_rolling_window():
    def sweep(k, seed):
        z, zdr = _gates(k, n=40_000, seed=seed)
        shape = (200, 200)
        return xr.Dataset({"DBZH": (("azimuth", "range"), z.reshape(shape)), "ZDR": (("azimuth", "range"), zdr.reshape(shape)),
                           "RHOHV": (("azimuth", "range"), np.full(shape, 0.99)), "SNRH": (("azimuth", "range"), np.full(shape, 30.0))},
                          coords={"azimuth": np.arange(200.0), "range": np.arange(200) * 250.0})
    est = ZDRSlopeAlpha(window_min=30)
    t0 = pd.Timestamp("2021-10-24 12:00")
    a1, _ = est.update(t0, sweep(0.06, 1))
    a2, _ = est.update(t0 + pd.Timedelta(minutes=60), sweep(0.01, 2))  # first sweep has left the window
    assert a1 == 0.015 and a2 > 0.035
