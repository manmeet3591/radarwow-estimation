"""Rain-rate relation coefficients, each with the source it was read from.

R in mm/h, Z in mm^6 m^-3 (linear), KDP in deg/km, A in dB/km (one-way), ZDR linear.
Sources are as transcribed in radar-wow/survey/03_estimation_merging_mosaic.md (coefficient table),
plus the cross-checks noted per entry. Relations not read in a source are not included.
"""

from __future__ import annotations

#: Z = a R^b
ZR_PRESETS = {
    "marshall_palmer": dict(a=200.0, b=1.6, z_max_dbz=None,
                            source="Marshall & Palmer 1948, doi:10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2; "
                                   "stratiform rain, any band; wradlib default"),
    "nexrad_convective": dict(a=300.0, b=1.4, z_max_dbz=53.0,
                              source="Fulton et al. 1998, doi:10.1175/1520-0434(1998)013<0377:TWRA>2.0.CO;2; "
                                     "WSR-88D default with the 53 dBZ hail cap; Py-ART ZtoR default"),
    "chennai_suresh2005": dict(a=267.0, b=1.345, z_max_dbz=None,
                               source="Suresh et al. 2005, MAUSAM 56, doi:10.54302/mausam.v56i2.950 (abstract); "
                                      "Chennai S-band DWR vs gauges <= 100 km, NE monsoon 2001"),
}

#: R = c KDP^d (KDP >= 0)
RKDP = {
    "S": dict(c=50.70, d=0.85, source="Py-ART est_rain_rate_kdp S default ('Beard and Chuang'), docstring cites Figueras i Ventura et al. 2012"),
    "C": dict(c=29.70, d=0.85, source="Py-ART est_rain_rate_kdp C default ('Beard and Chuang'), Figueras i Ventura et al. 2012"),
    "X": dict(c=15.81, d=0.7992, source="Py-ART est_rain_rate_kdp X default ('Brandes coefficients')"),
}
RKDP_PRESETS = {
    "nexrad_dpqpe": dict(band="S", c=44.0, d=0.822,
                         source="NEXRAD DP-QPE (Berkowitz et al. 2013) as quoted by Dixon et al. 2015 and Grams et al. 2019 (multiplier 44)"),
    "mrms_hail": dict(band="S", c=29.0, d=0.77,
                      source="MRMS R(KDP) for Z >= 50 dBZ, Zhang et al. 2017 AMS 38th Radar Conf. 22B.3"),
}

#: R = e A^f, by temperature (deg C). Ryzhkov et al. 2014 Table 1, doi:10.1175/JTECH-D-13-00038.1.
#: The 10 C row of every band is cross-checked against Py-ART's est_rain_rate_a defaults.
RA = {
    "S": {0: (2230.0, 1.03), 10: (3100.0, 1.03), 20: (4120.0, 1.03), 30: (5330.0, 1.03)},
    "C": {0: (221.0, 0.92), 10: (250.0, 0.91), 20: (294.0, 0.89), 30: (352.0, 0.89)},
    "X": {0: (49.1, 0.87), 10: (45.5, 0.83), 20: (43.5, 0.79), 30: (43.0, 0.76)},
}
RA_SOURCE = "Ryzhkov, Diederich, Zhang & Simmer 2014, JTECH 31, doi:10.1175/JTECH-D-13-00038.1, Table 1"

#: R = g Z^h ZDR^k (S band only; C/X need a DSD re-fit)
RZZDR = {"S": dict(g=0.0067, h=0.927, k=-3.43,
                   source="NEXRAD DP-QPE (Berkowitz et al. 2013) as quoted by Dixon et al. 2015; also CSU_RadarTools default")}


def ra_coefficients(band: str, temperature_c: float = 10.0) -> tuple[float, float]:
    """R(A) coefficients for a band, linearly interpolated in temperature between table rows (0-30 C)."""
    table = RA[band]
    ts = sorted(table)
    t = min(max(float(temperature_c), ts[0]), ts[-1])
    for lo, hi in zip(ts[:-1], ts[1:]):
        if lo <= t <= hi:
            w = (t - lo) / (hi - lo)
            return tuple((1 - w) * x + w * y for x, y in zip(table[lo], table[hi]))
    return table[ts[-1]]
