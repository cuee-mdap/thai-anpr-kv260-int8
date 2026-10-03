#!/bin/bash
# qat_s2_eval.sh — [S2b] รอ QAT train จบ แล้ว PTQ + วัด COCO mAP ด้วย flow 2.5 เดิม
#   y_c_qat.py เขียนผลเป็น yolov7_tiny_small_best.pth (ชื่อเดียวกับ flow เดิม) จึงใช้ชื่อนี้ตรง ๆ
#   ผลสรุป -> runs/qat_s2_261002/results_s2b.csv
set -u
cd "$(dirname "$0")/.."            # รากโปรเจกต์
OUT=runs/qat_s2_261002

while pgrep -f "run_qat_s2.sh" >/dev/null; do sleep 60; done
echo "[$(date '+%F %H:%M')] train จบ -> เริ่ม eval"

docker run --rm -v "$(pwd)":/workspace -w /workspace xilinx/vitis-ai-cpu:2.5.0 bash -lc '
source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch
(python3 -c "import pycocotools" 2>/dev/null || pip install pycocotools -q)
OUT=runs/qat_s2_261002
DATA=data/thai_license_plate_dataset_for_yolov7
# เฉพาะ area=all เท่านั้น: ถ้าไม่กรอง tail -1 จะไปโดน "area= large" = -1.000 (ชุดทดสอบไม่มีวัตถุใหญ่)
  ap() { grep -E "Average Precision +\(AP\) @\[ IoU=$1 *\| area= *all" "$2" 2>/dev/null | tail -1 | awk "{printf \"%.2f\", \$NF*100}"; }
echo "model,precision,mAP50,mAP75,mAP5095" > $OUT/results_s2b.csv
for d in $OUT/*_seed*/; do
  d=${d%/}; b=$(basename $d)
  [ -f "$d/yolov7_tiny_small_best.pth" ] || { echo "ข้าม $b (ไม่มี checkpoint)"; continue; }
  case "$b" in base*) XA="--no_csp4";; *) XA="";; esac
  echo "=== eval $b"
  python3 gpu/quantize/y_b_qt.py --mode float_test -d "$d" $XA --max_images 0 --eval_conf 0.001 \
      --test_images_dir $DATA/images/test --test_labels_dir $DATA/labels/test > "$d/float.log" 2>&1
  echo "$b,FP32,$(ap "0.50 " "$d/float.log"),$(ap 0.75 "$d/float.log"),$(ap 0.50:0.95 "$d/float.log")" >> $OUT/results_s2b.csv
  python3 gpu/quantize/y_b_qt.py --mode full -d "$d" $XA --max_images 0 --eval_conf 0.001 \
      --test_images_dir $DATA/images/test --test_labels_dir $DATA/labels/test > "$d/int8.log" 2>&1
  echo "$b,INT8,$(ap "0.50 " "$d/int8.log"),$(ap 0.75 "$d/int8.log"),$(ap 0.50:0.95 "$d/int8.log")" >> $OUT/results_s2b.csv
done
cat $OUT/results_s2b.csv'

echo "[$(date '+%F %H:%M')] S2b EVAL DONE"
