import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("pysteps")
pytest.importorskip("cv2")

from radarwow.estimation import accumulate, advection_accumulate, polar_to_cartesian

T0 = pd.Timestamp("2022-05-05 07:00")


def _grid(field):
    n = field.shape[0]
    x = np.arange(n) + 0.5 - n / 2
    return xr.DataArray(field, dims=("y", "x"), coords={"y": x[::-1], "x": x}, attrs={"res_km": 1.0})


def _cell(cx, n=120, r=6, rate=20.0):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.where((xx - cx) ** 2 + (yy - 60) ** 2 <= r ** 2, rate, 0.0)


def test_moving_cell_leaves_a_swath_not_stripes():
    # a 20 mm/h cell moving 1 km/min eastward, scanned every 6 min for 1 h
    scans = [(T0 + pd.Timedelta(minutes=6 * i), _grid(_cell(20 + 6 * i))) for i in range(11)]
    adv = advection_accumulate(scans, T0, T0 + pd.Timedelta(hours=1), return_motion=True)["ACCUM"].values
    eul = accumulate(scans, T0, T0 + pd.Timedelta(hours=1))["ACCUM"].values
    # along the track centre (row 60, x 30..70) the true total is ~ 2 r / 60 h * 20 mm/h = 4 mm everywhere
    a, e = adv[60, 30:70], eul[60, 30:70]
    assert np.std(a) < 0.5 * np.std(e)
    assert abs(a.mean() - 4.0) < 1.0
    # mass is conserved within a few per cent
    assert abs(adv.sum() - eul.sum()) / eul.sum() < 0.05


def test_motion_is_recovered():
    scans = [(T0 + pd.Timedelta(minutes=6 * i), _grid(_cell(20 + 6 * i))) for i in range(3)]
    out = advection_accumulate(scans, T0, T0 + pd.Timedelta(minutes=12), return_motion=True)
    u = out["U"].values[_cell(26) > 0]
    assert abs(np.median(u) - 60.0) < 10.0  # 6 km per 6 min = 60 km/h eastward
    assert abs(np.median(out["V"].values[_cell(26) > 0])) < 5.0


def test_static_rain_matches_eulerian():
    f = _cell(60) + 1.0
    scans = [(T0 + pd.Timedelta(minutes=6 * i), _grid(f)) for i in range(11)]
    adv = advection_accumulate(scans, T0, T0 + pd.Timedelta(hours=1))["ACCUM"].values
    assert np.allclose(adv[20:100, 20:100], f[20:100, 20:100], atol=0.05)


def test_polar_to_cartesian_geometry():
    az = np.arange(720) * 0.5 + 0.25
    r = (np.arange(600) + 0.5) * 250.0
    da = xr.DataArray(np.broadcast_to(az[:, None], (720, 600)).copy(), dims=("azimuth", "range"), coords={"azimuth": az, "range": r})
    c = polar_to_cartesian(da, 0.5, max_km=100, res_km=2.0)
    east = c.sel(x=51, y=1, method="nearest").item()
    north = c.sel(x=1, y=51, method="nearest").item()
    assert abs(east - 90) < 3 and (north < 3 or north > 357)
    assert np.isnan(c.sel(x=-99, y=-99, method="nearest").item())
