# radarwow-estimation

**radar-wow stage 6: Rain-rate estimators, VPR, blockage, hybrid scan.** R(Z), R(KDP), R(Z,ZDR), R(A); vertical profile correction; beam blockage; lowest-usable-tilt selection.

Part of [radar-wow](https://github.com/radar-wow), a set of repos that together build a radar QPE system that works for S, C and X band radars.
Every algorithm is one class implementing one paper, with the same interface: it takes an xradar sweep or volume with canonical moment names, declares the bands and moments it needs, and records provenance.

## Sub-components

| sub-component | algorithm | source | DOI | status |
|---|---|---|---|---|
| power-law estimators | `RZ`, `RKDP`, `RZZDR` | Ryzhkov 2022 / dual-pol QPE review (talk) |  | planned |
| VPR correction | `ZhangQi2010VPR` | Zhang & Qi 2010, J. Hydrometeor. | [doi:10.1175/2010JHM1201.1](https://doi.org/10.1175/2010JHM1201.1) | planned |
| beam blockage | `Bech2003Blockage` | Bech et al. 2003, JTECH | [doi:10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2](https://doi.org/10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2) | planned |
| hybrid scan | `HybridScan` | NMQ / MRMS (talk) |  | planned |

## Install

```bash
pip install git+https://github.com/radar-wow/radarwow-core git+https://github.com/radar-wow/radarwow-estimation
```

## Usage

```python
from radarwow.core import load_sample
from radarwow.estimation import *  # see table above

vol = load_sample('arm_csapr2_c')        # or open_volume('my_file')
# out = SomeAlgorithm()(vol)            # band read from the file, or band='S'|'C'|'X'
```

## Status legend

*implemented + notebook*: code, tests, executed notebook. *planned*: interface fixed, code to write. *stub*: class exists, raises `NotImplementedError`. *to review*: paper to read before deciding.

## License

MIT. Data used in the notebooks comes from public archives (NEXRAD on AWS, ARM, IMD via open-radar-data) under their own terms.
