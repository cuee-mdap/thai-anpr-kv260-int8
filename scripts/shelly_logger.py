# -*- coding: utf-8 -*-
"""shelly_logger.py — [S3] บันทึกค่าจาก Shelly Power Strip 4 Gen4 ที่ 1 Hz

ตามสเปกข้อ 6:
  · poll Shelly.GetStatus ที่ 1 Hz · timeout 2 s
  · บันทึกเวลา ISO + monotonic ของ host
  · ทุกช่อง switch:0-3 : apower, voltage, current, pf, freq, aenergy.total, output
  · CSV แบบ long: timestamp,t_s,label,channel,name,W,V,A,pf,Hz,Wh_total,output
  · อ่านพลาด -> NaN แล้วอ่านรอบถัดไป ไม่หยุดทั้งรัน
  · เตือนถ้า output=false

ใช้:
  python s1_261001/shelly_logger.py --label host_idle --seconds 180
  python s1_261001/shelly_logger.py --label host_gpu  --seconds 180 --out my.csv
"""
import argparse, csv, json, math, os, sys, time
import urllib.request
from datetime import datetime

DEFAULT_IP = '192.168.1.223'
NAMES = {0: 'host', 1: 'spare1', 2: 'spare2', 3: 'kv260'}
FIELDS = ['timestamp', 't_s', 'label', 'channel', 'name', 'W', 'V', 'A', 'pf', 'Hz', 'Wh_total', 'output']


def poll(ip, timeout=2.0):
    url = f'http://{ip}/rpc/Shelly.GetStatus'
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default=DEFAULT_IP)
    ap.add_argument('--label', required=True, help='ชื่อช่วงที่วัด เช่น host_idle, kv260_detect')
    ap.add_argument('--seconds', type=int, default=180)
    ap.add_argument('--out', default='outputs/e9_power/shelly_raw.csv')
    ap.add_argument('--hz', type=float, default=1.0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    new = not os.path.exists(a.out)
    fh = open(a.out, 'a', newline='', encoding='utf-8')
    w = csv.writer(fh)
    if new:
        w.writerow(FIELDS)

    t0 = time.monotonic()
    n_ok = n_err = 0
    warned_off = set()
    period = 1.0 / a.hz
    print(f'[{a.label}] เริ่ม {datetime.now():%H:%M:%S} · {a.seconds}s @ {a.hz} Hz -> {a.out}', flush=True)

    while True:
        t = time.monotonic() - t0
        if t >= a.seconds:
            break
        ts = datetime.now().isoformat(timespec='milliseconds')
        try:
            d = poll(a.ip)
            n_ok += 1
            for i in range(4):
                s = d.get(f'switch:{i}', {})
                out = s.get('output')
                if out is False and i not in warned_off:
                    print(f'  ⚠️  switch:{i} ({NAMES[i]}) output=false — ช่องนี้ไม่มีไฟ', flush=True)
                    warned_off.add(i)
                w.writerow([ts, f'{t:.3f}', a.label, i, NAMES.get(i, ''),
                            s.get('apower'), s.get('voltage'), s.get('current'),
                            s.get('pf'), s.get('freq'),
                            (s.get('aenergy') or {}).get('total'), out])
        except Exception as e:                       # อ่านพลาด -> NaN ไม่หยุดรัน
            n_err += 1
            for i in range(4):
                w.writerow([ts, f'{t:.3f}', a.label, i, NAMES.get(i, ''),
                            math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, ''])
            if n_err <= 3:
                print(f'  ! อ่านพลาดที่ t={t:.0f}s: {type(e).__name__}', flush=True)
        fh.flush()
        sleep = period - ((time.monotonic() - t0) - t)
        if sleep > 0:
            time.sleep(sleep)

    fh.close()
    print(f'[{a.label}] จบ · อ่านสำเร็จ {n_ok} ครั้ง · พลาด {n_err} ครั้ง', flush=True)


if __name__ == '__main__':
    main()
