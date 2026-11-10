import numpy as np
import pytest
import xarray as xr

from radarwow.estimation import ZhangQi2010VPR, fit_tilt
from radarwow.estimation.vpr import beam_height_km

P = ZhangQi2010VPR.params


def _true_vpr(h, z0=35.0, hb=1.6, hp=2.5, ht=3.4, peak=8.0):
    """Apparent (beam-broadened) bright band."""
    up = z0 + peak * (h - hb) / (hp - hb)
    down = z0 + peak + (z0 - 4 - z0 - peak) * (h - hp) / (ht - hp)
    z = np.where(h < hb, z0, np.where(h <= hp, up, down))
    return np.where(h > ht, z0 - 4 - 6 * (h - ht), z)


def _volume(elevations=(0.5, 0.9, 1.3, 1.8, 2.4), n_az=360, ng=920, dr=250.0):
    r = (np.arange(ng) + 0.5) * dr
    az = np.arange(n_az) + 0.5
    tree = {"/": xr.Dataset(attrs={"radarwow_band": "S"})}
    for i, el in enumerate(elevations):
        z = np.broadcast_to(_true_vpr(beam_height_km(r, el)), (n_az, ng)).astype(float).copy()
        tree[f"sweep_{i}"] = xr.Dataset({"DBZH": (("azimuth", "range"), z)},
                                        coords={"azimuth": az, "range": r, "elevation": ("azimuth", np.full(n_az, el))})
    dt = xr.DataTree.from_dict(tree)
    dt.attrs["radarwow_band"] = "S"
    return dt


def test_fit_recovers_linear_bright_band():
    h = np.arange(0.2, 3.4, 0.05)
    t = fit_tilt(h, _true_vpr(h), 3.4, P)
    assert t.corrected and not t.symmetric
    assert abs(t.h_peak - 2.5) < 0.2 and abs(t.h_bottom - 1.6) < 0.3
    assert t.alpha < 0 < t.beta
    corrected = _true_vpr(h) - t.delta_db(h)
    m = (h >= 1.6) & (h <= 3.4)
    assert np.mean(np.abs(corrected[m] - 35.0)) < 1.5 < np.mean(np.abs(_true_vpr(h)[m] - 35.0))


def test_symmetric_model_when_bottom_not_seen():
    h = np.arange(2.45, 3.4, 0.02)  # profile starts at the peak: no bottom visible
    t = fit_tilt(h, _true_vpr(h), 3.4, {**P, "min_points": 5})
    assert t.corrected and t.symmetric
    assert np.isclose(t.beta, -t.alpha)
    assert np.isclose(t.h_bottom, 2 * t.h_peak - 3.4)


def test_no_bright_band_is_not_corrected():
    rng = np.random.default_rng(0)
    h = np.arange(0.2, 3.2, 0.05)
    z = 35 - 1.0 * h + rng.normal(0, 0.3, h.size)  # Z decreasing with height, no bright band
    t = fit_tilt(h, z, 3.2, P)
    assert not t.corrected


def test_outliers_are_flagged_and_too_many_reject_the_tilt():
    h = np.arange(0.2, 3.4, 0.05)
    z = _true_vpr(h).copy()
    z[[5, 20, 50]] += 15.0
    t = fit_tilt(h, z, 3.4, P)
    assert t.corrected and t.bad_fraction > 0
    t = fit_tilt(h, z, 3.4, {**P, "max_bad_fraction": 0.01})
    assert not t.corrected and "flagged" in t.reason


def test_volume_correction_flattens_profile():
    dt = _volume()
    alg = ZhangQi2010VPR(freezing_level_km=2.4)  # apparent top 2.4 + 1.4 km half beam depth
    out = alg(dt)
    assert any(f.corrected for f in alg.fits)
    ds = out["sweep_0"].to_dataset()
    h = beam_height_km(ds["range"].values, 0.5)
    m = (h > 1.8) & (h < 3.2)
    before = np.abs(ds["DBZH"].values[:, m] - 35).mean()
    after = np.abs(ds["DBZH_VPR"].values[:, m] - 35).mean()
    assert after < 0.5 * before
    # below the bottom nothing changes
    assert np.allclose(ds["VPR_CORRECTION"].values[:, h < 1.3], 0.0)


def test_convective_columns_are_left_alone():
    dt = _volume()
    out = ZhangQi2010VPR(freezing_level_km=3.0, vil_convective=0.0)(dt)
    assert np.allclose(out["sweep_0"].to_dataset()["VPR_CORRECTION"].values, 0.0)


def test_needs_volume_and_freezing_level():
    dt = _volume()
    with pytest.raises(ValueError):
        ZhangQi2010VPR()(dt)
    with pytest.raises(TypeError):
        ZhangQi2010VPR(freezing_level_km=3.0)(dt["sweep_0"].to_dataset(), band="S")


def test_split_cut_prefers_dual_pol_sweep():
    from radarwow.estimation import unique_tilts

    dt = _volume(elevations=(0.5, 0.5, 0.9))
    ds = dt["sweep_1"].to_dataset()
    dt["sweep_1"] = xr.DataTree(ds.assign(PHIDP=ds["DBZH"] * 0))
    assert [n for _, n in unique_tilts(dt)] == ["sweep_1", "sweep_2"]
