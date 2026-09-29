# INT8-Quantization Robustness for Edge-FPGA License-Plate Recognition

Code, configurations, compiled INT8 models and per-run results behind the paper

> **INT8-Quantization Robustness for Edge-FPGA License-Plate Recognition: A Two-Stage Thai ANPR System on the AMD Kria KV260** Watchara Ruangsang and Supavadee Aramvith · IEEE Access (under revision)

Every number reported in the paper was produced by the scripts in `scripts/`; nothing here is re-derived or hand-edited. Where a result file is per-scene, it contains a correct/incorrect flag instead of the recognized text, because Thai plate numbers are personal data (see [Privacy](#privacy-what-is-not-released)).

---

## What is here

| folder | contents |
| :---- | :---- |
| `scripts/` | 19 scripts: training, INT8 post-training quantization, quantization-aware training, evaluation, activation analysis, dataset preparation and on-board measurement |
| `configs/` | calibration configurations (entropy, entropy + power-of-two, percentile) and the anchor sets; min–max is the toolchain default and needs no file — see `configs/README.md` |
| `weights/` | the four compiled INT8 `.xmodel` files that run on the board, plus `MANIFEST.csv` listing all 48 FP32 checkpoints with sizes and MD5 checksums |
| `results/` | 16 result files: per-seed, per-layer and per-scene outputs behind Tables I–VI and Figs. 7–11 |

## Environment

| stage | environment |
| :---- | :---- |
| Training, activation analysis, GPU/CPU benchmarks | PyTorch 2.4.1 + CUDA 12.1 (conda env `lpr_pt`), NVIDIA RTX 3060 12 GB |
| INT8 PTQ, compilation | Docker `xilinx/vitis-ai-cpu:2.5.0`, conda env `vitis-ai-pytorch` (`pytorch_nndct`, `vai_c_xir`) |
| Quantization-aware training | Docker `xilinx/vitis-ai-pytorch-gpu:3.0.0.001` (training only; every reported number is evaluated through the 2.5 flow) |
| Board | AMD Kria KV260, PetaLinux 2022.1 (kernel 5.15.19-xilinx-v2022.1), DPUCZDX8G B4096, Vitis-AI 2.5 runtime (VART/XIR) |

```sh
conda create -n lpr_pt python=3.8 -y && conda activate lpr_pt
pip install torch==2.4.1 torchvision opencv-python pycocotools numpy matplotlib pandas
```

## Reproducing the results

```sh
# ---- detector: train one seed (proposed model and the csp4-free ablation) ----
python scripts/y_a_train.py --seed 0 --epochs 300 --output_dir runs/plus_seed0
python scripts/y_a_train.py --seed 0 --epochs 300 --no_csp4 --output_dir runs/base_seed0
python scripts/y_a_train.py --seed 0 --epochs 300 --no_csp4 --width_mult 1.5 \
       --output_dir runs/basewide_seed0                       # capacity-matched control

# ---- INT8 PTQ + COCO mAP (inside the Vitis-AI 2.5 container) ----
python scripts/y_b_qt.py --mode float_test -d runs/plus_seed0 --max_images 0 --eval_conf 0.001
python scripts/y_b_qt.py --mode calib -q calib -d runs/plus_seed0 --max_images 0
python scripts/y_b_qt.py --mode test  -q test  -d runs/plus_seed0 --max_images 0 --eval_conf 0.001
#   calibration controls: add --quant_config configs/quant_cfg_entropy_pot.json (or _entropy / _percentile)

# ---- recognizer ----
python scripts/b.train.py                                      # CRNN + CTC (deployed recognizer)
python scripts/models_ocr_baseline.py                          # LPRNet-style deployable peer
python scripts/c.qt.py --calib_split train --calib_size 100     # INT8 PTQ + sensitivity sweep

# ---- quantization-aware training (Table III) ----
python scripts/y_c_qat.py --ckpt runs/base_seed0/yolov7_tiny_small_best.pth \
       --no_csp4 --epochs 30 --lr 3e-5 --out_dir runs/qat_base

# ---- mechanism: per-layer activation statistics and simulated INT8 SQNR (Fig. 8) ----
python scripts/act_stats.py --max_images 500

# ---- cross-domain check (Section VI.E) ----
python scripts/prepare_ccpd.py         # downloads/converts CCPD2020
python scripts/prepare_oid_plates.py   # downloads/converts Open Images V7 "Vehicle registration plate"
python scripts/fit_anchors.py --dataset data/ccpd_plate_yolov7      # k-means anchors per domain
ANCHORS="11.7,2.4;16.1,3.1;22.0,3.9" python scripts/finetune_public.py \
       --dataset_path data/ccpd_plate_yolov7 --out runs/ccpd_plus/best.pth
python scripts/collect_e6.py

# ---- on board (run these on the KV260) ----
python scripts/kv260_tail_latency.py --iters 1000     # Table IV incl. p95/p99/p99.9
python scripts/kv260_power_bench.py                   # INA260 power, whole SOM
python scripts/run_e2e_int8.py                        # end-to-end INT8 over the test scenes

# ---- host-side GPU/CPU comparison (Table V) ----
python scripts/bench_platform_power.py                # throughput and NVML power in one loop
```

The Thai dataset is not redistributable, so the scripts above reproduce the *method*. The cross-domain experiment runs end to end from public data alone (CCPD2020 and Open Images V7).

## Which file backs which table or figure

| paper item | result file | produced by |
| :---- | :---- | :---- |
| Table I — detection accuracy, proposed vs YOLOv4-tiny peer | `results/ablation_per_seed.csv`, `results/baseline_yolov4tiny.csv` | `y_a_train.py` → `y_b_qt.py`, `models_baseline.py` |
| Table II — recognizer vs LPRNet peer and EasyOCR | `results/baseline_lprnet.csv`, `results/recognizer_int8_sensitivity.csv` | `models_ocr_baseline.py`, `b.train.py`, `c.qt.py` |
| Table III — csp4 ablation (22 seeds) + QAT rows | `results/ablation_per_seed.csv`, `results/ablation_summary.json`, `results/qat_paired.csv` | `y_a_train.py`, `y_b_qt.py`, `y_c_qat.py`, `aggregate_ablation.py` |
| Table IV — on-board latency, p95/p99/p99.9, application level | `results/tail_latency.json`, `results/tail_latency_breakdown.json` | `kv260_tail_latency.py` |
| Table V — cross-platform throughput and power | `results/platform_power_gpu_cpu.json` (+ board rows of `tail_latency.json`) | `bench_platform_power.py`, `kv260_power_bench.py` |
| Table VI — end-to-end accuracy and the McNemar tests | `results/mcnemar_per_scene_4cfg.csv`, `results/mcnemar_power.json`, `results/board_int8_e2e_per_scene.json` | `run_e2e_int8.py` |
| Fig. 7 — ablation across IoU thresholds | `results/ablation_per_seed.csv` | `aggregate_ablation.py` |
| Fig. 8 — SQNR along the network, head-input distribution | `results/activation_per_layer.csv`, `results/activation_prehead_summary.csv` | `act_stats.py` |
| Figs. 9–10 — on-board efficiency and latency breakdown | `results/tail_latency.json`, `results/tail_latency_breakdown.json` | `kv260_tail_latency.py` |
| Fig. 11 — platform comparison | `results/platform_power_gpu_cpu.json` | `bench_platform_power.py` |
| Section V.F — IoU versus reading accuracy, per scene | `results/iou_vs_accuracy_per_scene.csv` | `iou_coupling.py` |
| Section VI.E — cross-domain CCPD2020 and Open Images | `results/cross_domain_ccpd_oid.csv` | `prepare_ccpd.py`, `prepare_oid_plates.py`, `fit_anchors.py`, `finetune_public.py`, `collect_e6.py` |
| Deployed INT8 models | `weights/dt_model.xmodel` (detector), `weights/lpr_model.xmodel` (recognizer), `weights/dt_v4tiny_seed0.xmodel`, `weights/lpr_baseline_seed0.xmodel` (peers) | `y_b_qt.py`/`c.qt.py` → `vai_c_xir` |
| All 48 FP32 checkpoints (sizes + MD5) | `weights/MANIFEST.csv` | — |

## Privacy: what is *not* released

The Thai plate images come from an industrial partner and are personal data under Thailand's Personal Data Protection Act (PDPA, B.E. 2562). They are not redistributable, and neither is the recognized text. Concretely, in this repository:

* no plate imagery of any kind;  
* per-scene result files carry a **correct/incorrect flag (0/1)**, never the predicted or ground-truth plate string — this is enough to recompute every accuracy, CER-free metric and McNemar test reported in the paper;  
* the example training log in `scripts/b.train.py` keeps its loss and accuracy lines but its per-sample `Predict / Label` lines were removed for the same reason.

FP32 checkpoints (48 files, 366 MB) exceed what belongs in a code repository; `weights/MANIFEST.csv` lists every one with its size and MD5 so that a copy can be verified, and they accompany the archived record of this repository.

## How to cite

Please cite the article:

```bibtex
@article{ruangsang_int8_kv260,
  author  = {Ruangsang, Watchara and Aramvith, Supavadee},
  title   = {{INT8-Quantization} Robustness for {Edge-FPGA} License-Plate Recognition:
             A Two-Stage Thai {ANPR} System on the {AMD} {Kria} {KV260}},
  journal = {IEEE Access},
  year    = {2026},
  note    = {under revision}
}
```

If you use the cross-domain preparation scripts, please also cite Open Images V7 and CCPD as the dataset sources. Machine-readable metadata for this repository is in `CITATION.cff`.

Repository: [https://github.com/cuee-mdap/thai-anpr-kv260-int8](https://github.com/cuee-mdap/thai-anpr-kv260-int8) (citable snapshot: release `v1.0.0`)

## License

Code in `scripts/` and `configs/`: MIT (see `LICENSE`). The compiled `.xmodel` files in `weights/` are released under separate terms stated in `LICENSE`.
