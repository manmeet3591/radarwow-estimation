import numpy as np
import pytest
import xarray as xr

from radarwow.estimation import Bech2003BeamBlockage, beam_block_fraction, fulton1998_correction_db, hybrid_scan


def test_bech_fraction_limits_and_half():
    a = 100.0
    assert beam_block_fraction(-200, 0, a) == 0.0
    assert beam_block_fraction(200, 0, a) == 1.0
    assert np.isclose(beam_block_fraction(0, 0, a), 0.5)
    # area of the circular segment below y = -a/2: (2*pi/3 - sqrt(3)/2)/... = 0.1955 of the disc
    assert np.isclose(beam_block_fraction(-50, 0, a), 0.1955, atol=1e-3)


def test_matches_wradlib():
    wrl = pytest.importorskip("wradlib")
    th = np.linspace(-150, 150, 31)
    ours = beam_block_fraction(th, 0.0, 100.0)
    ref = wrl.qual.beam_block_frac(th, np.zeros_like(th), np.full_like(th, 100.0))
    assert np.allclose(ours, ref, atol=1e-6)


def test_fulton_steps():
    out = fulton1998_correction_db(np.array([0.05, 0.2, 0.35, 0.5, 0.58, 0.7]))
    assert np.allclose(out[:5], [0, 1, 2, 3, 4]) and np.isnan(out[5])


def _flat_dem(height_m, ridge=None):
    lat = np.arange(34.0, 36.0, 0.01)
    lon = np.arange(-98.5, -96.0, 0.01)
    z = np.full((lat.size, lon.size), height_m, dtype="float32")
    if ridge:  # a north-south ridge east of the radar
        z[:, (lon > ridge[0]) & (lon < ridge[0] + 0.05)] = ridge[1]
    return xr.DataArray(z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def _sweep(el=0.5):
    az = np.arange(360) + 0.5
    r = (np.arange(400) + 0.5) * 250.0
    return xr.Dataset({"DBZH": (("azimuth", "range"), np.full((360, 400), 30.0))},
                      coords={"azimuth": az, "range": r, "elevation": ("azimuth", np.full(360, el))})


def test_ridge_blocks_only_its_sector_and_beyond():
    site = (35.0, -97.3, 400.0)
    out = Bech2003BeamBlockage(dem=_flat_dem(300.0, ridge=(-96.9, 1200.0)), site=site)(_sweep(), band="S")
    cbb = out["CBB"].values
    assert cbb[270].max() < 0.01  # west: open
    east = cbb[90]
    ridge_gate = int(np.argmax(out["PBB"].values[90] > 0.5))
    assert ridge_gate > 0 and east[-1] >= 0.99 and np.all(np.diff(east) >= 0)
    assert np.isnan(out["DBZH_BB"].values[90, -1]) and np.isclose(out["DBZH_BB"].values[270, -1], 30.0)


def test_hybrid_scan_takes_lowest_unblocked():
    f0, f1 = np.full((2, 2), 1.0), np.full((2, 2), 2.0)
    c0 = np.array([[0.0, 0.7], [0.2, 0.9]])
    c1 = np.array([[0.0, 0.1], [0.0, 0.6]])
    field, idx = hybrid_scan([(f0, c0), (f1, c1)])
    assert np.array_equal(idx, [[0, 1], [0, -1]]) and np.isnan(field[1, 1]) and field[0, 1] == 2.0
