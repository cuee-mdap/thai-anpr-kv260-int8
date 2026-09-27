"""
bench_platform_power.py — [E9a / ตอบ R3-5] วัด throughput + กำลังไฟ "ที่วัดจริง" ของ GPU และ CPU
  เดิมเปเปอร์ใช้ TDP (CPU ~65 W, GPU ~170 W) ซึ่ง reviewer 3 ข้อ 5 ติงว่าไม่ใช่ฐานเดียวกับ KV260 (INA260 วัดจริง)
  สคริปต์นี้วัดของจริงเท่าที่ฮาร์ดแวร์อนุญาต:
    · GPU : nvidia-smi power.draw (กำลังไฟทั้งการ์ด วัดจริงโดยเซนเซอร์บนบอร์ด) สุ่มตัวอย่างระหว่างรัน
    · CPU : เครื่องนี้ไม่มี RAPL (kernel 5.4 ไม่รองรับ Raptor Lake) → วัดได้เฉพาะ throughput
            กำลังไฟ CPU จึงยังต้องอ้าง spec (i7-13700: PL1 65 W / PL2 219 W) และต้องระบุให้ชัดในเปเปอร์
  โหลดงานที่ใช้ = detector ตัวจริงของเรา batch 1 + recognizer ตัวจริง (โปรโตคอลเดียวกับ Table V)

*** ต้องรันตอน GPU ว่าง เท่านั้น ***  รัน: python gpu/eval/bench_platform_power.py --iters 500
ผลลัพธ์: outputs/e9_power/platform_power.json
"""
import os, sys, json, time, subprocess, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train'))
from y_a_train import YOLOv7TinySmallPlus  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'quantize'))

DET_CKPT = 'outputs/trained_yolov7_tiny_small_plus/yolov7_tiny_small_best.pth'


def gpu_power():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().split('\n')[0]
        return float(out)
    except Exception:
        return None


def gpu_util():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().split('\n')[0]
        return float(out)
    except Exception:
        return None


def bench_throughput(model, device, iters, warmup, sample_power):
    """โหมด throughput: ไม่ sync ทุก iteration (โปรโตคอลเดียวกับตัวเลข FPS ที่รายงานใน Table V)
       → ได้ FPS กับกำลังไฟจากการรันเดียวกัน จึงหารกันได้อย่างถูกต้อง"""
    x = torch.randn(1, 3, 320, 320, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        pw, ut = [], []
        step = max(1, iters // 40)
        t0 = time.perf_counter()
        for i in range(iters):
            model(x)
            if sample_power and i % step == 0:
                p_, u_ = gpu_power(), gpu_util()
                if p_: pw.append(p_)
                if u_ is not None: ut.append(u_)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    d = dict(mode='throughput', iters=iters, fps=float(iters / wall), ms_per_frame=float(wall / iters * 1000))
    if pw:
        d.update(power_W_mean=round(float(np.mean(pw)), 2), power_W_max=round(float(np.max(pw)), 2),
                 gpu_util_mean=round(float(np.mean(ut)), 1),
                 fps_per_W=round(d['fps'] / float(np.mean(pw)), 3))
    return d


def bench(model, device, iters, warmup, sample_power):
    x = torch.randn(1, 3, 320, 320, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat = np.empty(iters)
        pw, ut = [], []
        step = max(1, iters // 40)
        t0 = time.perf_counter()
        for i in range(iters):
            t = time.perf_counter()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            lat[i] = (time.perf_counter() - t) * 1000
            if sample_power and i % step == 0:
                p, u = gpu_power(), gpu_util()
                if p: pw.append(p)
                if u is not None: ut.append(u)
        wall = time.perf_counter() - t0
    d = dict(iters=iters, mean_ms=float(lat.mean()), p50_ms=float(np.percentile(lat, 50)),
             p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)),
             fps=float(iters / wall))
    if pw:
        d.update(power_W_mean=round(float(np.mean(pw)), 2), power_W_max=round(float(np.max(pw)), 2),
                 power_W_idle_before=IDLE, gpu_util_mean=round(float(np.mean(ut)), 1),
                 fps_per_W=round(d['fps'] / float(np.mean(pw)), 3))
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--out', default='outputs/e9_power/platform_power.json')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    util0 = gpu_util()
    if util0 is not None and util0 > 15:
        print(f'⚠️ GPU กำลังถูกใช้งานอยู่ ({util0}%) — ค่าที่วัดจะไม่ถูกต้อง หยุดก่อน', flush=True)
        sys.exit(2)
    IDLE = gpu_power()
    print(f'[E9a] GPU idle power = {IDLE} W (util {util0}%)', flush=True)

    sd = torch.load(DET_CKPT, map_location='cpu')
    res = {'note': 'detector ตัวเดียวกับ Table V, batch 1, input 320x320',
           'gpu_idle_W': IDLE, 'date': time.strftime('%Y-%m-%d %H:%M')}

    m = YOLOv7TinySmallPlus(nc=1, num_predictions=3)
    m.load_state_dict(sd.get('model_state_dict', sd))
    mc = m.cuda()
    res['gpu_fp32_throughput'] = bench_throughput(mc, torch.device('cuda'), args.iters, args.warmup, True)
    print('[E9] GPU throughput:', json.dumps(res['gpu_fp32_throughput']), flush=True)
    res['gpu_fp32_latency'] = bench(mc, torch.device('cuda'), args.iters, args.warmup, True)
    print('[E9] GPU latency   :', json.dumps(res['gpu_fp32_latency']), flush=True)

    m2 = YOLOv7TinySmallPlus(nc=1, num_predictions=3)
    m2.load_state_dict(sd.get('model_state_dict', sd))
    torch.set_num_threads(os.cpu_count())
    res['cpu_fp32'] = bench_throughput(m2.cpu(), torch.device('cpu'), max(100, args.iters // 4), 20, False)
    res['cpu_fp32']['power_note'] = ('RAPL ใช้ไม่ได้บนเครื่องนี้ (kernel 5.4 ไม่รองรับ Raptor Lake) '
                                     'i7-13700 spec: PL1 65 W / PL2 219 W → 65 W เป็นขอบล่าง')
    res['cpu_fp32']['threads'] = os.cpu_count()
    print('[E9a] CPU :', json.dumps(res['cpu_fp32']), flush=True)

    json.dump(res, open(args.out, 'w'), indent=1, ensure_ascii=False)
    print('[E9a] เขียนผล ->', args.out, flush=True)
