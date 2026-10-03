#!/bin/bash
# run_s1_ciou.sh — [S1 / R3-3] เทรน Plus ด้วย CIoU 3 ซีด → PTQ INT8 → compile → รันบนบอร์ด → ให้คะแนน
#
# วางโฟลเดอร์นี้ไว้ใน testocr แล้วเรียกจาก testocr:
#     cd <repo root>
#     bash s1_260930/run_s1_ciou.sh <ขั้น>
#
# ขั้น (รันตามลำดับ · แต่ละขั้นรันซ้ำได้ ข้ามซีดที่ทำเสร็จแล้ว):
#   test     [HOST lpr_pt] ทดสอบ loss (ไม่กี่วินาที)
#   smoke    [HOST lpr_pt] เทรน 1 epoch ซีด 0 ดูว่ารันผ่าน (ลบผลทิ้งเอง)
#   train    [HOST lpr_pt] เทรน 3 ซีด (ใช้เวลานาน → ใช้ nohup ดู README)
#   quant    [DOCKER vitis-ai-cpu:2.5.0, conda vitis-ai-pytorch] FP32 test + PTQ INT8 (mAP ครบ 3 ค่า)
#   compile  [DOCKER เดิม] vai_c_xir → dt_ciou_seedN.xmodel
#   board    [HOST] ส่งขึ้น KV260 · รัน 648 ฉาก · ดึง json กลับ
#   score    [HOST] คิดความถูกต้อง end-to-end + IoU bin (ต้องมี outputs/e4_653/per_image_4cfg.json)
#   summary  [HOST] ตารางสรุปเทียบ MSE (ซีด 0–2 เดิม) → runs/ciou/S1_SUMMARY.md
set -eo pipefail
PKG="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PKG/.." && pwd)"          # = testocr
cd "$ROOT"

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-300}"                 # Table VIII: ≤300 epoch + early stop (ตรวจกับ log เดิมก่อน ดู README)
BOX_W="${BOX_W:-1.0}"
OUT=runs/ciou
DATA=data/thai_license_plate_dataset_for_yolov7
PY="${PY:-python3}"   # ตั้ง PY=<path> ถ้าใช้ conda env เฉพาะ
[ -x "$PY" ] || PY=python3
BOARD="${BOARD:-kv260}"
BOARD_MODEL_DIR=/home/root/kv260_yolov7_ocr
BOARD_EVAL_DIR=/home/root/e2e_eval
GT_JSON="${GT_JSON:-outputs/e4_653/per_image_4cfg.json}"
mkdir -p "$OUT"

install_wrapper() {
  cp "$PKG/y_a_train_ciou.py" "$PKG/test_ciou_loss.py" gpu/train/
}

case "$1" in
test)
  install_wrapper
  $PY gpu/train/test_ciou_loss.py
  ;;

smoke)
  install_wrapper
  $PY gpu/train/y_a_train_ciou.py --seed 0 --epochs 1 --box_weight "$BOX_W" \
      --dataset_path "$DATA" --output_dir "$OUT/_smoke" 2>&1 | tail -30
  echo "ถ้าเห็น loss และ Validation ของ epoch 1 = ใช้ได้ · ลบทิ้ง: rm -rf $OUT/_smoke"
  ;;

train)
  install_wrapper
  for s in $SEEDS; do
    d="$OUT/plus_seed$s"
    if [ -f "$d/yolov7_tiny_small_best.pth" ] && [ -f "$d/.done" ]; then echo "ข้าม $d (เสร็จแล้ว)"; continue; fi
    mkdir -p "$d"
    echo "=== train CIoU plus seed $s → $d  ($(date)) ==="
    $PY gpu/train/y_a_train_ciou.py --seed "$s" --epochs "$EPOCHS" --box_weight "$BOX_W" \
        --dataset_path "$DATA" --output_dir "$d" 2>&1 | tee "$d/train.log"
    touch "$d/.done"
  done
  ;;

quant)   # ใน docker: /workspace ต้องเป็น testocr
  command -v vai_c_xir >/dev/null || { echo "ต้องรันใน docker vitis-ai-cpu:2.5.0 + conda activate vitis-ai-pytorch"; exit 1; }
  python3 -c "import pycocotools" 2>/dev/null || pip install pycocotools
  for s in $SEEDS; do
    d="$OUT/plus_seed$s"
    [ -f "$d/yolov7_tiny_small_best.pth" ] || { echo "ไม่มี checkpoint $d"; continue; }
    common="-d $d --max_images 0 --eval_conf 0.001 --test_images_dir $DATA/images/test --test_labels_dir $DATA/labels/test"
    [ -s "$d/float.log" ] || python3 gpu/quantize/y_b_qt.py --mode float_test $common 2>&1 | tee "$d/float.log"
    [ -s "$d/int8.log" ]  || python3 gpu/quantize/y_b_qt.py --mode full       $common 2>&1 | tee "$d/int8.log"
    ls -la "$d/quantize/"*.xmodel
  done
  ;;

compile)
  command -v vai_c_xir >/dev/null || { echo "ต้องรันใน docker"; exit 1; }
  for s in $SEEDS; do
    d="$OUT/plus_seed$s"
    x=$(ls "$d"/quantize/*_int.xmodel | head -1)
    vai_c_xir -x "$x" -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
              -o "$d/compiled" -n "dt_ciou_seed$s" 2>&1 | tail -5
    ls -la "$d/compiled/dt_ciou_seed$s.xmodel"
  done
  ;;

board)
  ssh "$BOARD" "xmutil unloadapp; xmutil loadapp kv260-benchmark-b4096" || true
  scp "$PKG/run_e2e_int8.py" "$BOARD:$BOARD_EVAL_DIR/run_e2e_int8_s1.py"
  mkdir -p outputs/e2e_board_int8
  for s in $SEEDS; do
    d="$OUT/plus_seed$s"
    scp "$d/compiled/dt_ciou_seed$s.xmodel" "$BOARD:$BOARD_MODEL_DIR/"
    ssh "$BOARD" "cd $BOARD_MODEL_DIR && python3 $BOARD_EVAL_DIR/run_e2e_int8_s1.py 0 \
        $BOARD_MODEL_DIR/dt_ciou_seed$s.xmodel $BOARD_EVAL_DIR/board_ciou_seed$s.json"
    scp "$BOARD:$BOARD_EVAL_DIR/board_ciou_seed$s.json" outputs/e2e_board_int8/
  done
  ;;

score)
  # ตรวจตัวให้คะแนนก่อน: ไฟล์ของโมเดลที่ deploy จริงต้องได้เลขเดียวกับในเปเปอร์
  $PY "$PKG/score_e2e.py" --board outputs/e2e_board_int8/board_int8_e2e_n648.json --gt "$GT_JSON" \
      --tag deployed_MSE --out "$OUT/score_deployed.json"
  for s in $SEEDS; do
    $PY "$PKG/score_e2e.py" --board "outputs/e2e_board_int8/board_ciou_seed$s.json" --gt "$GT_JSON" \
        --tag "ciou_seed$s" --out "$OUT/score_ciou_seed$s.json"
  done
  ;;

summary)
  $PY "$PKG/summarize_s1.py" --runs "$OUT" --seeds $SEEDS --baseline "$PKG/ablation_per_seed.csv"
  ;;

*)
  sed -n 2,20p "$0"; exit 1 ;;
esac
