#!/bin/bash
# run_b2_full.sh — [B2 · 1 ต.ค.] ปิดประเด็น anchor ให้ครบด้วยผลรันจริง
#
#   ส่วนที่ 1  FP32 mAP@0.75 / @0.5:0.95 ของ 8 checkpoint ที่เปเปอร์ใช้ (ตอนนี้มีแค่ @0.5)
#   ส่วนที่ 2  ประเมิน 4 checkpoint ที่ fit anchor ใหม่ (b_oid_*) ทั้ง FP32 และ INT8
#              — E6b เทรนไว้ 25 ก.ย. แต่ถูกขัดจังหวะก่อนถึงขั้น eval จึงยังไม่มีตัวเลข
#
#   ต้องรันใน docker: xilinx/vitis-ai-cpu:2.5.0 + conda activate vitis-ai-pytorch
#   ผลออกที่ runs/e6_ood/B2_CHECK/  (ตัวเลขล้วน)
set -e
ROOT=/workspace
cd "$ROOT"
E6=runs/e6_ood
OUT=$E6/B2_CHECK
mkdir -p "$OUT"
A_OID="1.6,1.3;5.3,3.7;11.0,5.0"

ap() {  # $1=IoU spec  $2=log
  grep -E "Average Precision.*IoU=$1 \| area=   all" "$2" 2>/dev/null | tail -1 \
    | awk '{printf "%.2f", $NF*100}'
}

CSV=$OUT/b2_full_results.csv
echo "group,anchors,dataset,variant,seed,precision,mAP50,mAP75,mAP5095" > "$CSV"

# ---------- ส่วนที่ 1: FP32 ครบ 3 ค่า ของชุดที่เปเปอร์ใช้ (anchor ไทยเดิม) ----------
for D in $E6/ccpd_*_seed*/ $E6/oid_*_seed*/; do
  D=${D%/}; B=$(basename "$D")
  [ -f "$D/yolov7_tiny_small_best.pth" ] || continue
  DSN=${B%%_*}; REST=${B#*_}; VAR=${REST%%_*}; S=${B##*seed}
  case "$VAR" in base) XA="--no_csp4";; *) XA="";; esac
  case "$DSN" in oid) DS=data/oid_plate_yolov7;; ccpd) DS=data/ccpd_plate_yolov7;; esac
  unset ANCHORS
  if [ ! -s "$D/float_full.log" ]; then
    echo "=== [1] FP32 ครบ 3 ค่า: $B (anchor ไทยเดิม)"
    python3 gpu/quantize/y_b_qt.py --mode float_test -d "$D" $XA --max_images 0 --eval_conf 0.001 \
        --test_images_dir $DS/images/test --test_labels_dir $DS/labels/test > "$D/float_full.log" 2>&1 || true
  else
    echo "=== [1] ข้าม $B (มี float_full.log แล้ว)"
  fi
  echo "paper,thai-default,$DSN,$VAR,$S,FP32,$(ap 0.50 "$D/float_full.log"),$(ap 0.75 "$D/float_full.log"),$(ap 0.50:0.95 "$D/float_full.log")" >> "$CSV"
done

# ---------- ส่วนที่ 2: ชุด anchor ที่ fit ใหม่ (b_oid_*) — FP32 + INT8 ----------
export ANCHORS="$A_OID"
for D in $E6/b_oid_*_seed*/; do
  D=${D%/}; B=$(basename "$D")
  [ -f "$D/yolov7_tiny_small_best.pth" ] || continue
  REST=${B#b_oid_}; VAR=${REST%%_*}; S=${B##*seed}
  case "$VAR" in base) XA="--no_csp4";; *) XA="";; esac
  DS=data/oid_plate_yolov7
  echo "=== [2] refit anchors: $B  ANCHORS=$ANCHORS"
  [ -s "$D/float_full.log" ] || python3 gpu/quantize/y_b_qt.py --mode float_test -d "$D" $XA \
      --max_images 0 --eval_conf 0.001 \
      --test_images_dir $DS/images/test --test_labels_dir $DS/labels/test > "$D/float_full.log" 2>&1 || true
  echo "refit,oid-kmeans,oid,$VAR,$S,FP32,$(ap 0.50 "$D/float_full.log"),$(ap 0.75 "$D/float_full.log"),$(ap 0.50:0.95 "$D/float_full.log")" >> "$CSV"

  if [ ! -s "$D/int8_full.log" ]; then
    python3 gpu/quantize/y_b_qt.py --mode calib -q calib -d "$D" $XA --max_images 500 \
        --test_images_dir $DS/images/train --test_labels_dir $DS/labels/train > "$D/calib_full.log" 2>&1 || true
    python3 gpu/quantize/y_b_qt.py --mode test -q test -d "$D" $XA --max_images 0 --eval_conf 0.001 \
        --test_images_dir $DS/images/test --test_labels_dir $DS/labels/test > "$D/int8_full.log" 2>&1 || true
  fi
  echo "refit,oid-kmeans,oid,$VAR,$S,INT8,$(ap 0.50 "$D/int8_full.log"),$(ap 0.75 "$D/int8_full.log"),$(ap 0.50:0.95 "$D/int8_full.log")" >> "$CSV"
done
unset ANCHORS

echo; echo "=== ผลรวม $CSV ==="
cat "$CSV"
