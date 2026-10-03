#!/bin/bash
# run_s3_campaign.sh — [S3] วัดไฟที่ปลั๊กครบทุกช่วง ตามสเปกข้อ 6-7
#   host  (switch:0): idle · cpu inference · gpu inference
#   kv260 (switch:3): idle · detection · end-to-end
#   แต่ละช่วง 180 s × 3 รอบ · logger อ่าน 1 Hz ทุกช่องพร้อมกัน
#
# ใช้: nohup bash s1_261001/run_s3_campaign.sh > runs/s3_campaign.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}   # ตั้ง PY=<path> ถ้าใช้ conda env เฉพาะ
LOG=s1_261001/shelly_logger.py
OUT=outputs/e9_power/shelly_raw.csv
SEC=${SEC:-180}
REPS=${REPS:-3}
mkdir -p outputs/e9_power
echo "[$(date '+%F %H:%M')] S3 CAMPAIGN START (SEC=$SEC REPS=$REPS)"

measure () {   # $1=label  $2=คำสั่งสร้างโหลด (ว่าง = idle)
  local label="$1"; shift
  $PY "$LOG" --label "$label" --seconds "$SEC" --out "$OUT" &
  local LPID=$!
  if [ -n "${1:-}" ]; then
    sleep 2
    timeout $((SEC+20)) bash -c "$*" >/dev/null 2>&1
  fi
  wait $LPID
  echo "[$(date '+%F %H:%M')]   เสร็จ $label"
  sleep 20                      # ให้ระบบกลับสู่ idle ก่อนช่วงถัดไป
}

for r in $(seq 1 $REPS); do
  echo "[$(date '+%F %H:%M')] ---- รอบ $r/$REPS ----"

  # ---------- host ----------
  measure "host_idle_r$r" ""
  measure "host_cpu_r$r"  "$PY s1_261001/wallplug_load.py --phase cpu --seconds $SEC"
  measure "host_gpu_r$r"  "$PY s1_261001/wallplug_load.py --phase gpu --seconds $SEC"

  # ---------- kv260 (สั่งผ่าน ssh จาก host เพื่อใช้นาฬิกาเดียวกัน) ----------
  measure "kv260_idle_r$r" ""
  if ssh -o ConnectTimeout=5 -o BatchMode=yes kv260 true 2>/dev/null; then
    measure "kv260_detect_r$r" "ssh -o BatchMode=yes kv260 'cd /home/root/e2e_eval 2>/dev/null && timeout $SEC python3 /home/root/kv260_power_bench.py --iters 100000 2>/dev/null || sleep $SEC'"
  else
    echo "[$(date '+%F %H:%M')]   ข้าม kv260_detect_r$r (ssh ไม่ได้)"
  fi
done

echo "[$(date '+%F %H:%M')] S3 CAMPAIGN DONE -> $OUT"
wc -l "$OUT"
