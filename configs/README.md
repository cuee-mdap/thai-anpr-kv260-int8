# configs

| file | what it is | used for |
|---|---|---|
| *(no file)* | **min-max calibration is the Vitis-AI 2.5 default** - `y_b_qt.py`/`c.qt.py` without `--quant_config` reproduce every headline number in the paper | Tables I-IV, VI, and the ablation of Table III |
| `quant_cfg_entropy.json` | entropy calibration, float scales | calibration control, Section V.C |
| `quant_cfg_entropy_pot.json` | entropy calibration, power-of-two scales (DPU-compatible) | calibration control, Section V.C |
| `quant_cfg_percentile.json` | percentile (99.99) scale selection | the excluded control of Section V.C; it collapses the sanity reference as well, which is a defect of that calibration path in this toolchain build |
| `anchors.json` | anchor sets: Thai deployed, CCPD refit, Open Images refit | detector training and the cross-domain experiment of Section VI.E |
