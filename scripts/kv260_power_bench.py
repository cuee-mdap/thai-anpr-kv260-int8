#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kv260_power_bench.py — [E6] วัด power(W) / FPS / FPS-per-W / latency แยก stage บน KV260
#
# *** รันบนบอร์ด KV260 เท่านั้น (ต้องมี xir/vart + DPU overlay โหลดแล้ว) ***
#   xmutil unloadapp; xmutil loadapp kv260-benchmark-b4096   # โหลด DPU สดก่อน
#   python3 -u kv260_power_bench.py --dt dt_model.xmodel --ocr lpr_model.xmodel --iters 500
#
# หมายเหตุ: อ่าน power ใน main thread (สลับกับ exec) — ห้ามใช้ thread แยกตอน VART รัน (ทำให้ค้าง)
import os
import sys
import time
import glob
import json
import argparse
import numpy as np

try:
    import xir
    import vart
except ImportError:
    print("ERROR: ต้องรันบนบอร์ด KV260 (มี xir/vart). บน host รันไม่ได้")
    sys.exit(1)

POWER_PATHS = sorted(glob.glob('/sys/class/hwmon/hwmon*/power*_input'))  # KV260: ina260 (uW)


def read_power():
    """อ่าน power รวม (W) จาก hwmon — เรียกใน main thread เท่านั้น"""
    tot, ok = 0.0, False
    for p in POWER_PATHS:
        try:
            tot += int(open(p).read().strip()) / 1e6  # uW -> W
            ok = True
        except Exception:
            pass
    return tot if ok else None


def get_dpu_subgraph(xmodel_path):
    g = xir.Graph.deserialize(xmodel_path)
    return [s for s in g.get_root_subgraph().toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def bench_stage(name, xmodel_path, iters):
    sgs = get_dpu_subgraph(xmodel_path)
    if not sgs:
        raise RuntimeError(f"ไม่พบ DPU subgraph ใน {xmodel_path}")
    runner = vart.Runner.create_runner(sgs[0], "run")
    it = runner.get_input_tensors()
    ot = runner.get_output_tensors()
    ish = tuple(it[0].dims)
    osh = tuple(ot[0].dims)
    xin = [np.zeros(ish, dtype=np.int8)]
    xout = [np.zeros(osh, dtype=np.int8)]

    for _ in range(10):  # warmup
        jid = runner.execute_async(xin, xout)
        runner.wait(jid)

    samples = []
    step = max(1, iters // 30)  # อ่าน power ~30 ครั้งระหว่าง loop (main thread)
    t0 = time.time()
    for i in range(iters):
        jid = runner.execute_async(xin, xout)
        runner.wait(jid)
        if i % step == 0:
            w = read_power()
            if w is not None:
                samples.append(w)
    dt = time.time() - t0

    lat_ms = dt / iters * 1000.0
    fps = iters / dt
    pw = sum(samples) / len(samples) if samples else None
    size_mb = os.path.getsize(xmodel_path) / 1e6
    res = {
        "stage": name, "xmodel": os.path.basename(xmodel_path),
        "input": list(ish), "output": list(osh), "model_MB": round(size_mb, 2),
        "iters": iters, "latency_ms": round(lat_ms, 3), "fps": round(fps, 1),
        "power_W": round(pw, 2) if pw else None,
        "fps_per_W": round(fps / pw, 2) if pw else None,
        "energy_mJ_per_inf": round(pw / fps * 1000, 2) if pw else None,
    }
    del runner
    return res


def main():
    ap = argparse.ArgumentParser(description='[E6] power/FPS-W/latency บน KV260')
    ap.add_argument('--dt', default='dt_model.xmodel')
    ap.add_argument('--ocr', default='lpr_model.xmodel')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--out', default='kv260_power_bench.json')
    args = ap.parse_args()

    print("=" * 64)
    print("KV260 POWER / FPS / LATENCY BENCHMARK [E6]")
    print("=" * 64)
    if POWER_PATHS:
        print(f"power sensors: {POWER_PATHS}")
        time.sleep(1)
        idle = read_power()
        print(f"idle power ~ {idle:.2f} W" if idle else "อ่าน power ไม่ได้")
    else:
        print("⚠️ ไม่พบ hwmon power sensor — วัดได้แค่ FPS/latency")
    print("-" * 64)

    results = []
    for name, path in [("DETECT", args.dt), ("OCR", args.ocr)]:
        if not os.path.exists(path):
            print(f"ข้าม {name}: ไม่พบ {path}")
            continue
        print(f">> bench {name} ({os.path.basename(path)}) x{args.iters} ...", flush=True)
        r = bench_stage(name, path, args.iters)
        results.append(r)
        print(f"   latency={r['latency_ms']}ms | FPS={r['fps']} | power={r['power_W']}W | "
              f"FPS/W={r['fps_per_W']} | E={r['energy_mJ_per_inf']}mJ/inf | size={r['model_MB']}MB",
              flush=True)

    if len(results) == 2:
        e2e = results[0]['latency_ms'] + results[1]['latency_ms']
        print(f">> END-TO-END (DPU detect+OCR): ~{e2e:.2f} ms => ~{1000.0/e2e:.1f} FPS"
              f"  (ยังไม่รวม CPU crop/decode)", flush=True)
        results.append({"stage": "END2END_DPU", "latency_ms": round(e2e, 2),
                        "fps": round(1000.0 / e2e, 1)})

    json.dump(results, open(args.out, 'w'), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
