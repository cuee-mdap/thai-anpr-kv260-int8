#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kv260_tail_latency.py — [E10 / ตอบ R1-6] วัด latency ราย iteration บน KV260 -> mean/p50/p95/p99/p99.9/max
#
# *** รันบนบอร์ด KV260 เท่านั้น *** (ต้องมี xir/vart + overlay kv260-benchmark-b4096 โหลดอยู่)
#   python3 -u kv260_tail_latency.py --iters 1000
#
# วัด 3 แบบ (เก็บเวลาแต่ละรอบทั้งหมด ไม่ใช่ค่าเฉลี่ยรวม):
#   1) detect DPU-only  2) ocr DPU-only  3) end-to-end ต่อเฟรมจากภาพจริง (preprocess+detect+crop+ocr+CTC)
# บันทึก power (INA260) + อุณหภูมิ ควบคู่ เพื่อดูว่า tail เกิดตอนร้อนหรือไม่
import os, sys, json, time, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e_int8 import DPU, detect_best_box, ocr_read, MODEL_DIR, IMG_DIR, LIST_FILE  # noqa: E402
from PIL import Image  # noqa: E402

POWER_PATHS = sorted(glob.glob('/sys/class/hwmon/hwmon*/power*_input'))
TEMP_PATHS = sorted(glob.glob('/sys/class/hwmon/hwmon*/temp*_input'))


def read_power():
    tot, ok = 0.0, False
    for p in POWER_PATHS:
        try:
            tot += int(open(p).read().strip()) / 1e6; ok = True
        except Exception:
            pass
    return tot if ok else None


def read_temp():
    vals = []
    for p in TEMP_PATHS:
        try:
            vals.append(int(open(p).read().strip()) / 1000.0)
        except Exception:
            pass
    return max(vals) if vals else None


def summarize(lat_ms, label, extra=None):
    a = np.asarray(lat_ms, dtype=np.float64)
    d = {
        "stage": label, "n": int(a.size),
        "mean_ms": float(a.mean()), "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
        "p999_ms": float(np.percentile(a, 99.9)), "max_ms": float(a.max()), "min_ms": float(a.min()),
        "fps_mean": float(1000.0 / a.mean()), "fps_p99": float(1000.0 / np.percentile(a, 99)),
    }
    d["jitter_p99_over_p50"] = d["p99_ms"] / d["p50_ms"]
    if extra:
        d.update(extra)
    return d


def bench_dpu(dpu, iters, warmup=200, label="stage"):
    """DPU-only: execute_async + wait ด้วยบัฟเฟอร์ที่จองไว้แล้ว (นิยามเดียวกับ Table IV)"""
    xin = [np.zeros(dpu.in_dims, dtype=np.int8, order='C')]
    xout = [np.zeros(dpu.out_dims, dtype=np.int8, order='C')]
    for _ in range(warmup):
        jid = dpu.r.execute_async(xin, xout); dpu.r.wait(jid)
    lat = np.empty(iters, dtype=np.float64)
    pw, tp = [], []
    step = max(1, iters // 50)
    for i in range(iters):
        t0 = time.perf_counter()
        jid = dpu.r.execute_async(xin, xout); dpu.r.wait(jid)
        lat[i] = (time.perf_counter() - t0) * 1000.0
        if i % step == 0:
            w = read_power(); t = read_temp()
            if w: pw.append(w)
            if t: tp.append(t)
    extra = {"power_W_mean": round(float(np.mean(pw)), 2) if pw else None,
             "temp_C_max": round(float(np.max(tp)), 1) if tp else None}
    return lat, summarize(lat, label, extra)


def bench_e2e(det, ocr, files, iters, warmup=50):
    """end-to-end ต่อเฟรมจากภาพจริง (ไม่รวมเวลาอ่านดิสก์ — โหลดภาพก่อนจับเวลา เหมือน Table IV)"""
    dsc, osc = det.in_scale, ocr.in_scale
    lat = np.empty(iters, dtype=np.float64)
    lat_det = np.empty(iters, dtype=np.float64)
    lat_ocr = np.full(iters, np.nan, dtype=np.float64)
    pw, tp = [], []
    step = max(1, iters // 50)
    nfiles = len(files)
    for i in range(-warmup, iters):
        img = Image.open(os.path.join(IMG_DIR, files[(i + warmup) % nfiles]))
        img.load()
        t0 = time.perf_counter()
        box, conf = detect_best_box(det, img, dsc)
        t1 = time.perf_counter()
        if box is not None:
            x1, y1, x2, y2 = box
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            _ = ocr_read(ocr, crop, osc)
        t2 = time.perf_counter()
        if i >= 0:
            lat[i] = (t2 - t0) * 1000.0
            lat_det[i] = (t1 - t0) * 1000.0
            if box is not None:
                lat_ocr[i] = (t2 - t1) * 1000.0
            if i % step == 0:
                w = read_power(); t = read_temp()
                if w: pw.append(w)
                if t: tp.append(t)
    extra = {"power_W_mean": round(float(np.mean(pw)), 2) if pw else None,
             "temp_C_max": round(float(np.max(tp)), 1) if tp else None,
             "det_part_mean_ms": float(np.nanmean(lat_det)),
             "ocr_part_mean_ms": float(np.nanmean(lat_ocr)),
             "frames_with_detection": int(np.sum(~np.isnan(lat_ocr)))}
    return lat, summarize(lat, "end-to-end (per frame, real images)", extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--out', default='/home/root/e2e_eval/tail_latency.json')
    ap.add_argument('--raw', default='/home/root/e2e_eval/tail_latency_raw.npz')
    ap.add_argument('--breakdown', action='store_true', help='วัด breakdown ต่อขั้นแทน (E10b)')
    args = ap.parse_args()

    print(f"power sensors: {len(POWER_PATHS)} | temp sensors: {len(TEMP_PATHS)}", flush=True)
    det = DPU(os.path.join(MODEL_DIR, "dt_model.xmodel"))
    ocr = DPU(os.path.join(MODEL_DIR, "lpr_model.xmodel"))
    files = [l.strip() for l in open(LIST_FILE, encoding='utf-8') if l.strip()]
    print(f"images: {len(files)} | iters: {args.iters}", flush=True)

    if args.breakdown:
        br = bench_breakdown(det, ocr, files, args.iters)
        print(json.dumps(br, indent=1), flush=True)
        json.dump({"iters": args.iters, "date": time.strftime("%Y-%m-%d %H:%M"), "breakdown": br},
                  open(args.out.replace('.json', '_breakdown.json'), "w"), indent=1)
        print(f"DONE -> {args.out.replace('.json','_breakdown.json')}", flush=True)
        return

    out = {"iters": args.iters, "date": time.strftime("%Y-%m-%d %H:%M"), "stages": []}
    raw = {}

    lat, s = bench_dpu(det, args.iters, label="detect (DPU only)")
    print(json.dumps(s, indent=1), flush=True); out["stages"].append(s); raw["detect"] = lat

    lat, s = bench_dpu(ocr, args.iters, label="ocr (DPU only)")
    print(json.dumps(s, indent=1), flush=True); out["stages"].append(s); raw["ocr"] = lat

    lat, s = bench_e2e(det, ocr, files, args.iters)
    print(json.dumps(s, indent=1), flush=True); out["stages"].append(s); raw["e2e"] = lat

    json.dump(out, open(args.out, "w"), indent=1)
    np.savez_compressed(args.raw, **raw)
    print(f"DONE -> {args.out} , {args.raw}", flush=True)




# ---------------------------------------------------------------------------
# [E10b] breakdown ต่อเฟรม: preprocess / DPU / postprocess / crop / OCR / CTC
#   ใช้ตรวจว่าเวลาจริงต่อเฟรมไปอยู่ส่วนไหน (Fig. 9 เดิมบอก CPU ~0.04 ms ซึ่งวัดเฉพาะ crop)
#   python3 -u kv260_tail_latency.py --breakdown --iters 500
# ---------------------------------------------------------------------------
def bench_breakdown(det, ocr, files, iters, warmup=30):
    import run_e2e_int8 as E
    dsc, osc = det.in_scale, ocr.in_scale
    keys = ["det_pre", "det_dpu", "det_post", "crop", "ocr_pre", "ocr_dpu", "ocr_ctc", "total"]
    acc = {k: np.full(iters, np.nan) for k in keys}
    nfiles = len(files)
    for i in range(-warmup, iters):
        img = Image.open(os.path.join(E.IMG_DIR, files[(i + warmup) % nfiles])); img.load()
        W0, H0 = img.size
        t = [time.perf_counter()]
        # --- detect preprocess ---
        im = img.convert('RGB').resize((E.IMG_SIZE, E.IMG_SIZE), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        xi = np.clip(np.round(x * dsc), -128, 127).astype(np.int8)
        t.append(time.perf_counter())
        # --- detect DPU ---
        out = det.run(xi)
        t.append(time.perf_counter())
        # --- detect postprocess (decode + NMS) ---
        p = out[0].transpose(2, 0, 1).reshape(3, 5, 40, 40)
        gh, gw = 40, 40; stride = E.IMG_SIZE / gh
        gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
        bx = (E.sigmoid_np(p[:, 0]) + gx[None]) * stride
        by = (E.sigmoid_np(p[:, 1]) + gy[None]) * stride
        bw = E.ANCHORS[:, 0][:, None, None] * np.exp(p[:, 2]) * stride
        bh = E.ANCHORS[:, 1][:, None, None] * np.exp(p[:, 3]) * stride
        conf = E.sigmoid_np(p[:, 4]); m = conf >= E.CONF_THR
        box = None
        if m.any():
            bx_, by_, bw_, bh_, sc = bx[m], by[m], bw[m], bh[m], conf[m]
            boxes = np.stack([bx_ - bw_ / 2, by_ - bh_ / 2, bx_ + bw_ / 2, by_ + bh_ / 2], axis=1)
            keep = E.nms_np(boxes, sc, E.IOU_THR); boxes, sc = boxes[keep], sc[keep]
            b = boxes[sc.argmax()]; sx, sy = W0 / E.IMG_SIZE, H0 / E.IMG_SIZE
            box = (max(0.0, b[0] * sx), max(0.0, b[1] * sy), min(float(W0), b[2] * sx), min(float(H0), b[3] * sy))
        t.append(time.perf_counter())
        if box is None:
            continue
        # --- crop ---
        crop = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        t.append(time.perf_counter())
        # --- ocr preprocess ---
        imo = crop.convert('L').resize((E.W_OCR, E.H_OCR), Image.BILINEAR)
        xo = np.asarray(imo, dtype=np.float32) / 255.0
        xo = (xo - 0.5) / 0.5
        xoi = np.clip(np.round(xo * osc), -128, 127).astype(np.int8)
        t.append(time.perf_counter())
        # --- ocr DPU ---
        oo = ocr.run(xoi.reshape(1, E.H_OCR, E.W_OCR, 1))
        t.append(time.perf_counter())
        # --- CTC greedy decode ---
        seq = oo.reshape(54, 53).argmax(axis=1)
        prev = -1; chars = []
        for c in seq:
            if c != 0 and c != prev:
                chars.append(E.CHARS[int(c)])
            prev = c
        _ = ''.join(chars)
        t.append(time.perf_counter())
        if i >= 0:
            d = np.diff(np.asarray(t)) * 1000.0
            for k, v in zip(keys[:-1], d):
                acc[k][i] = v
            acc["total"][i] = (t[-1] - t[0]) * 1000.0
    res = {}
    for k in keys:
        a = acc[k][~np.isnan(acc[k])]
        if a.size:
            res[k] = dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                          p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
                          max_ms=float(a.max()), n=int(a.size))
    return res


if __name__ == "__main__":
    main()
