"""Rain-rate estimators: R(Z), R(KDP), R(A), R(Z,ZDR) and a synthetic blend.

Every estimator writes its own field (RATE_Z, RATE_KDP, RATE_A, RATE_ZZDR) so they can be compared;
`SyntheticRainRate` combines them into RATE with RATE_SOURCE telling which one was used per gate.
Relations hold for rain only: gates above the melting layer are set to NaN when the height of the
0 C level is known (``height_over_iso0`` in the file, or the ``freezing_level_km`` parameter).
"""

from __future__ import annotations

import numpy as np

from radarwow.core import Algorithm, Band, Reference, register
from radarwow.core.noise import signal_mask

from . import coefficients as co

DIMS = ("azimuth", "range")


def below_melting_layer(ds, freezing_level_km=None, ml_depth_km=0.5, ml_bottom_km=None):
    """True where the beam centre is below the melting-layer bottom (or everywhere if unknown).

    Priority: ``ml_bottom_km`` (height above the radar, e.g. from radarwow.classification QVP detection),
    then the file's ``height_over_iso0``, then ``freezing_level_km`` minus ``ml_depth_km``.
    """
    if ml_bottom_km is not None and np.isfinite(ml_bottom_km):
        r = ds["range"].values
        el = np.deg2rad(ds["elevation"].values if "elevation" in ds else np.zeros(ds.sizes["azimuth"]))
        ke_re = 4.0 / 3.0 * 6371e3
        beam_h = np.sqrt(r[None, :] ** 2 + ke_re ** 2 + 2 * r[None, :] * ke_re * np.sin(el[:, None])) - ke_re
        return beam_h < ml_bottom_km * 1e3
    if "height_over_iso0" in ds:
        h = ds["height_over_iso0"].values
        return ~(np.isfinite(h) & (h > -ml_depth_km * 1e3))
    if freezing_level_km is None:
        return np.ones(ds["DBZH"].shape, dtype=bool)
    r = ds["range"].values
    el = np.deg2rad(ds["elevation"].values if "elevation" in ds else np.zeros(ds.sizes["azimuth"]))
    ke_re = 4.0 / 3.0 * 6371e3
    beam_h = np.sqrt(r[None, :] ** 2 + ke_re ** 2 + 2 * r[None, :] * ke_re * np.sin(el[:, None])) - ke_re
    alt = float(ds.attrs.get("altitude", 0.0) or 0.0)
    return beam_h + alt < (freezing_level_km - ml_depth_km) * 1e3


def _rate(values, attrs, name):
    return (DIMS, values.astype("float32"), {"units": "mm/h", "long_name": name, **attrs})


@register
class ZRRelation(Algorithm):
    name = "zr_relation"
    stage = "estimation"
    reference = Reference("J. S. Marshall, W. M. Palmer", 1948, "The distribution of raindrops with size",
                          "J. Meteor. 5, 165-166", "10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2")
    requires = ("DBZH",)
    provides = ("RATE_Z",)
    notes = "Presets in coefficients.ZR_PRESETS (marshall_palmer, nexrad_convective, chennai_suresh2005), or pass a and b."
    params = dict(preset="marshall_palmer", a=None, b=None, z_max_dbz=None, freezing_level_km=None, ml_bottom_km=None)

    def run_sweep(self, ds, cfg):
        p = dict(co.ZR_PRESETS[self.params["preset"]])
        a = self.params["a"] or p["a"]
        b = self.params["b"] or p["b"]
        cap = self.params["z_max_dbz"] if self.params["z_max_dbz"] is not None else p["z_max_dbz"]
        z = ds["DBZH"].values.astype(float)
        if cap is not None:
            z = np.minimum(z, cap)
        r = (10 ** (0.1 * z) / a) ** (1.0 / b)
        ok = signal_mask(ds) & below_melting_layer(ds, self.params["freezing_level_km"], ml_bottom_km=self.params.get("ml_bottom_km"))
        out = ds.copy()
        out["RATE_Z"] = _rate(np.where(ok, r, np.nan), {"relation": f"Z = {a} R^{b}", "source": p["source"]}, "rain rate from Z")
        return out


@register
class RainRateKDP(Algorithm):
    name = "rain_rate_kdp"
    stage = "estimation"
    reference = Reference("A. V. Ryzhkov, D. S. Zrnic", 1996, "Assessment of rainfall measurement that uses specific differential phase",
                          "J. Appl. Meteor. 35, 2080-2090", "10.1175/1520-0450(1996)035<2080:AORMTU>2.0.CO;2")
    requires = ("KDP",)
    provides = ("RATE_KDP",)
    notes = "Band defaults from coefficients.RKDP; presets nexrad_dpqpe and mrms_hail (S band). Negative KDP gives 0."
    params = dict(preset=None, freezing_level_km=None, ml_bottom_km=None)

    def run_sweep(self, ds, cfg):
        if self.params["preset"]:
            p = co.RKDP_PRESETS[self.params["preset"]]
            if p["band"] != cfg.band.value:
                raise ValueError(f"preset {self.params['preset']} is for band {p['band']}")
        else:
            p = co.RKDP[cfg.band.value]
        k = np.clip(ds["KDP"].values.astype(float), 0, None)
        r = p["c"] * k ** p["d"]
        ok = np.isfinite(ds["KDP"].values) & below_melting_layer(ds, self.params["freezing_level_km"], ml_bottom_km=self.params.get("ml_bottom_km"))
        out = ds.copy()
        out["RATE_KDP"] = _rate(np.where(ok, r, np.nan), {"relation": f"R = {p['c']} KDP^{p['d']}", "source": p["source"]}, "rain rate from KDP")
        return out


@register
class RainRateA(Algorithm):
    name = "ryzhkov2014_rain_rate_a"
    stage = "estimation"
    reference = Reference("A. Ryzhkov, M. Diederich, P. Zhang, C. Simmer", 2014,
                          "Potential utilization of specific attenuation for rainfall estimation, mitigation of partial beam "
                          "blockage, and radar networking", "J. Atmos. Oceanic Technol. 31, 599-619", "10.1175/JTECH-D-13-00038.1")
    requires = ("DBZH", "PHIDP")
    optional = ("AH", "PHIDP_REC", "RHOHV", "height_over_iso0")
    provides = ("RATE_A",)
    notes = ("A_H from radarwow-attenuation (ZPHI) when absent. Coefficients from Ryzhkov 2014 Table 1 for the band, "
             "interpolated to temperature_c. Used only on rays whose delta PHIDP >= ra_min_dphi (band config): with a "
             "small total phase shift the attenuation is too small to measure and R(A) is dominated by noise. "
             "Also not used where PIA > ra_max_pia (ZPHI unstable on strongly attenuated paths).")
    params = dict(temperature_c=10.0, freezing_level_km=None, ml_bottom_km=None, min_dphi=None)

    def run_sweep(self, ds, cfg):
        if "AH" not in ds:
            from radarwow.attenuation import Testud2000ZPHI
            ds = Testud2000ZPHI()(ds, band=cfg)
        from radarwow.attenuation.common import path_dphi, phidp_rec, rain_path

        e, f = co.ra_coefficients(cfg.band.value, self.params["temperature_c"])
        a = ds["AH"].values.astype(float)
        r = e * np.clip(a, 0, None) ** f
        _, _, i0, i1 = rain_path(ds, cfg)
        dphi = path_dphi(phidp_rec(ds, cfg), i0, i1)
        min_dphi = self.params["min_dphi"] if self.params["min_dphi"] is not None else cfg["ra_min_dphi"]
        ok = np.isfinite(a) & (dphi >= min_dphi)[:, None] & below_melting_layer(ds, self.params["freezing_level_km"], ml_bottom_km=self.params.get("ml_bottom_km"))
        if "PIA" in ds:  # ZPHI amplifies noise where the path is strongly attenuated
            ok &= ds["PIA"].values <= cfg["ra_max_pia"]
        if "PHASE_CONSISTENT" in ds:  # PHIDP rise far larger than the Z can explain: not rain attenuation
            ok &= ds["PHASE_CONSISTENT"].values.astype(bool)[:, None]
        out = ds.copy()
        out["RATE_A"] = _rate(np.where(ok, r, np.nan), {"relation": f"R = {e:.1f} A^{f:.3f} ({self.params['temperature_c']} C)",
                                                       "source": co.RA_SOURCE}, "rain rate from specific attenuation")
        return out


@register
class RainRateZZDR(Algorithm):
    name = "rain_rate_z_zdr"
    stage = "estimation"
    reference = Reference("A. V. Ryzhkov, S. E. Giangrande, T. J. Schuur", 2005, "Rainfall estimation with a polarimetric prototype of WSR-88D",
                          "J. Appl. Meteor. 44, 502-515", "10.1175/JAM2213.1")
    bands = frozenset({Band.S})
    requires = ("DBZH", "ZDR")
    provides = ("RATE_ZZDR",)
    notes = "S band only (NEXRAD DP-QPE coefficients); needs calibrated ZDR: a 0.2 dB ZDR bias changes R by ~15 %."
    params = dict(freezing_level_km=None, ml_bottom_km=None)

    def run_sweep(self, ds, cfg):
        p = co.RZZDR["S"]
        z = 10 ** (0.1 * ds["DBZH"].values.astype(float))
        zdr = 10 ** (0.1 * ds["ZDR"].values.astype(float))
        r = p["g"] * z ** p["h"] * zdr ** p["k"]
        ok = signal_mask(ds) & np.isfinite(r) & below_melting_layer(ds, self.params["freezing_level_km"], ml_bottom_km=self.params.get("ml_bottom_km"))
        out = ds.copy()
        out["RATE_ZZDR"] = _rate(np.where(ok, r, np.nan), {"relation": f"R = {p['g']} Z^{p['h']} ZDR^{p['k']}", "source": p["source"]},
                                 "rain rate from Z and ZDR")
        return out


SOURCE_CODES = {0: "none", 1: "R(A)", 2: "R(KDP)", 3: "R(Z)"}


@register
class SyntheticRainRate(Algorithm):
    name = "synthetic_rain_rate"
    stage = "estimation"
    reference = Reference("J. Zhang, L. Tang, S. Cocks, P. Zhang, A. Ryzhkov, K. Howard, C. Langston, B. Kaney", 2020,
                          "A dual-polarization radar synthetic QPE for operations", "J. Hydrometeor. 21, 2507-2521",
                          "10.1175/JHM-D-19-0194.1")
    requires = ("DBZH",)
    optional = ("PHIDP", "KDP", "AH", "RHOHV", "height_over_iso0")
    provides = ("RATE", "RATE_SOURCE", "RATE_A", "RATE_KDP", "RATE_Z")
    notes = ("Per gate, below the melting layer: R(KDP) where Z >= z_hail_dbz (hail possible), else R(A) where A_H is "
             "defined, else R(Z) with Z capped at z_max_dbz (53 dBZ, the WSR-88D hail cap of Fulton et al. 1998). The MRMS 50 km blending zone and evaporation correction are not implemented. "
             "Works with Z only (single-pol data) by falling back to R(Z).")
    params = dict(z_hail_dbz=50.0, zr_preset="marshall_palmer", z_max_dbz=53.0, temperature_c=10.0, freezing_level_km=None, ml_bottom_km=None)

    def run_sweep(self, ds, cfg):
        p = self.params
        ml = dict(freezing_level_km=p["freezing_level_km"], ml_bottom_km=p["ml_bottom_km"])
        out = ZRRelation(preset=p["zr_preset"], z_max_dbz=p["z_max_dbz"], **ml)(ds, band=cfg)
        if "PHIDP" in ds:
            out = RainRateA(temperature_c=p["temperature_c"], **ml)(out, band=cfg)
        if "KDP" in out:
            out = RainRateKDP(**ml)(out, band=cfg)
        z = ds["DBZH"].values
        rz = out["RATE_Z"].values
        ra = out["RATE_A"].values if "RATE_A" in out else np.full(z.shape, np.nan)
        rk = out["RATE_KDP"].values if "RATE_KDP" in out else np.full(z.shape, np.nan)
        hail = (z >= p["z_hail_dbz"]) & np.isfinite(rk)
        use_a = ~hail & np.isfinite(ra)
        use_z = ~hail & ~use_a & np.isfinite(rz)
        rate = np.where(hail, rk, np.where(use_a, ra, np.where(use_z, rz, np.nan)))
        src = np.where(hail, 2, np.where(use_a, 1, np.where(use_z, 3, 0))).astype("int8")
        out["RATE"] = _rate(rate, {"comment": "synthetic: R(KDP) if Z >= z_hail_dbz, else R(A), else R(Z)"}, "rain rate")
        out["RATE_SOURCE"] = (DIMS, src, {"flag_values": list(SOURCE_CODES), "flag_meanings": " ".join(SOURCE_CODES.values())})
        return out
