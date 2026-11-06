import numpy as np
import pandas as pd
import xarray as xr

from radarwow.estimation import accumulate, to_fixed_azimuths


def _rate(v, az=None):
    az = np.arange(720) * 0.5 + 0.25 if az is None else az
    return xr.DataArray(np.full((az.size, 4), v), dims=("azimuth", "range"), coords={"azimuth": az, "range": np.arange(4) * 250.0})


def test_constant_rain_for_one_hour():
    scans = [(pd.Timestamp("2022-05-05 07:00") + pd.Timedelta(minutes=6 * i), _rate(10.0)) for i in range(11)]
    acc = accumulate(scans, "2022-05-05 07:00", "2022-05-05 08:00")
    assert np.allclose(acc["ACCUM"].values, 10.0) and np.isclose(acc.attrs["coverage"], 1.0)


def test_gap_limited_to_max_gap():
    scans = [(pd.Timestamp("2022-05-05 07:00"), _rate(6.0))]
    acc = accumulate(scans, "2022-05-05 07:00", "2022-05-05 08:00", max_gap_min=15)
    assert np.allclose(acc["ACCUM"].values, 1.5) and np.isclose(acc.attrs["coverage"], 0.25)


def test_fixed_azimuth_regrid_handles_wrap():
    az = np.array([359.9, 0.4, 0.9]) % 360
    da = xr.DataArray(np.array([[1.0], [2.0], [3.0]]), dims=("azimuth", "range"), coords={"azimuth": az, "range": [0.0]})
    out = to_fixed_azimuths(da, n_az=720)
    assert out.values[0, 0] in (1.0, 2.0) and np.isnan(out.values[360, 0])
