import numpy as np
import pytest
import xarray as xr

from radarwow.attenuation import synthetic_attenuated_sweep
from radarwow.estimation import RainRateA, RainRateKDP, RainRateZZDR, SyntheticRainRate, ZRRelation, coefficients


def _sweep(z, **extra):
    z = np.asarray(z, dtype=float)
    data = {"DBZH": (("azimuth", "range"), z), "SNRH": (("azimuth", "range"), np.full(z.shape, 30.0))}
    data.update({k: (("azimuth", "range"), np.asarray(v, dtype=float)) for k, v in extra.items()})
    return xr.Dataset(data, coords={"azimuth": np.arange(z.shape[0]) + 0.5, "range": (np.arange(z.shape[1]) + 0.5) * 250.0,
                                    "elevation": ("azimuth", np.full(z.shape[0], 0.5))})


def test_zr_inverts_power_law():
    r_true = 10.0
    z = 10 * np.log10(200 * r_true ** 1.6)
    out = ZRRelation()(_sweep([[z]]), band="C")
    assert np.isclose(float(out["RATE_Z"][0, 0]), r_true, rtol=1e-6)


def test_nexrad_hail_cap():
    out = ZRRelation(preset="nexrad_convective")(_sweep([[53.0, 65.0]]), band="S")
    assert np.isclose(float(out["RATE_Z"][0, 0]), float(out["RATE_Z"][0, 1]))


def test_ra_coefficients_table_and_interpolation():
    assert coefficients.ra_coefficients("S", 10) == (3100.0, 1.03)
    e, f = coefficients.ra_coefficients("C", 15)
    assert 250 < e < 294 and 0.89 < f < 0.91


def test_rate_a_follows_specific_attenuation():
    ds = synthetic_attenuated_sweep(alpha_true=0.08)
    out = RainRateA()(ds, band="C")
    e, f = coefficients.ra_coefficients("C", 10)
    ah_true = ds["KDP_TRUE"].values * 0.08
    m = np.isfinite(out["RATE_A"].values) & (ah_true > 0.05)
    expect = e * ah_true[m] ** f
    assert np.median(np.abs(out["RATE_A"].values[m] - expect) / expect) < 0.15


def test_melting_layer_masks_far_gates():
    ds = _sweep(np.full((2, 600), 30.0))
    out = ZRRelation(freezing_level_km=1.5)(ds, band="C")
    r = out["RATE_Z"].values
    assert np.isfinite(r[:, 10]).all() and np.isnan(r[:, -1]).all()


def test_zzdr_is_s_band_only():
    with pytest.raises(ValueError):
        RainRateZZDR()(_sweep([[30.0]], ZDR=[[0.5]]), band="C")


def test_synthetic_picks_kdp_in_hail_and_z_without_phase():
    ds = _sweep([[55.0, 30.0]], KDP=[[3.0, 0.5]])
    out = SyntheticRainRate()(ds, band="S")
    assert out["RATE_SOURCE"].values.tolist() == [[2, 3]]
    k = RainRateKDP()(ds, band="S")
    assert np.isclose(float(out["RATE"][0, 0]), float(k["RATE_KDP"][0, 0]))


def test_synthetic_caps_hail_when_no_kdp():
    ds = _sweep([[69.0, 53.0]])
    out = SyntheticRainRate()(ds, band="C")
    r = out["RATE"].values[0]
    assert out["RATE_SOURCE"].values.tolist() == [[3, 3]] and np.isclose(r[0], r[1])
