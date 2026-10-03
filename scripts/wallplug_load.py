"""
wallplug_load.py — [S3 / R3-5] สร้างโหลดคงที่ให้อ่านมิเตอร์ที่ปลั๊ก (wall-plug) ได้ทีละช่วง
โหลด = detector ตัวเดียวกับ Table V, batch 1, input 320×320 (เหมือน bench_platform_power.py)

    cd testocr
    python s1_260930/wallplug_load.py --phase idle --seconds 180
    python s1_260930/wallplug_load.py --phase gpu  --seconds 180
    python s1_260930/wallplug_load.py --phase cpu  --seconds 180
แต่ละช่วงพิมพ์เวลาเริ่ม/จบ + FPS · อ่านมิเตอร์ระหว่างช่วง (ข้าม 30 วินาทีแรก) แล้วกรอก S3_wallplug_record.csv
ผลแต่ละช่วงต่อท้าย outputs/e9_power/wallplug_phases.jsonl
"""
import os, sys, json, time, argparse
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'gpu', 'train'))
DET_CKPT = os.path.join(ROOT, 'outputs/trained_yolov7_tiny_small_plus/yolov7_tiny_small_best.pth')


def gpu_power():
    try:
        import subprocess
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
        return float(out.decode().split()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['idle', 'gpu', 'cpu'])
    ap.add_argument('--seconds', type=int, default=180)
    ap.add_argument('--repeat', type=int, default=1, help='ครั้งที่ (ใส่ลง log)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs/e9_power/wallplug_phases.jsonl'))
    a = ap.parse_args()

    rec = dict(phase=a.phase, repeat=a.repeat, seconds=a.seconds)
    if a.phase == 'idle':
        t0 = datetime.now()
        print(f'[idle] เริ่ม {t0:%H:%M:%S} — ไม่ต้องทำอะไร อ่านมิเตอร์ช่วงนี้ ({a.seconds} s)', flush=True)
        time.sleep(a.seconds)
        rec.update(start=f'{t0:%H:%M:%S}', end=f'{datetime.now():%H:%M:%S}', fps=0.0)
    else:
        from y_a_train import YOLOv7TinySmallPlus
        dev = torch.device('cuda' if a.phase == 'gpu' else 'cpu')
        if a.phase == 'gpu' and not torch.cuda.is_available():
            sys.exit('ไม่มี CUDA')
        m = YOLOv7TinySmallPlus(nc=1, num_predictions=3, use_csp4=True)
        sd = torch.load(DET_CKPT, map_location='cpu')
        m.load_state_dict(sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd)
        m = m.to(dev).eval()
        if dev.type == 'cpu':
            torch.set_num_threads(os.cpu_count())   # เหมือน bench_platform_power.py
        x =torch.randn(1, 3, 320, 320, device=dev)
        with torch.no_grad():
            for _ in range(50):
                m(x)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = datetime.now(); ts = time.time(); n = 0; pw = []
            print(f'[{a.phase}] เริ่ม {t0:%H:%M:%S} — อ่านมิเตอร์หลัง 30 s แรก ({a.seconds} s)', flush=True)
            while time.time() - ts < a.seconds:
                m(x); n += 1
                if dev.type == 'cuda' and n % 200 == 0:
                    torch.cuda.synchronize()
                    p = gpu_power()
                    if p is not None:
                        pw.append(p)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            el = time.time() - ts
        rec.update(start=f'{t0:%H:%M:%S}', end=f'{datetime.now():%H:%M:%S}', frames=n,
                   fps=round(n / el, 2), threads=torch.get_num_threads(),
                   nvidia_smi_W_mean=round(sum(pw) / len(pw), 2) if pw else None)
    print(f'[{a.phase}] จบ {rec["end"]} · FPS = {rec["fps"]}'
          + (f' · nvidia-smi {rec.get("nvidia_smi_W_mean")} W' if rec.get('nvidia_smi_W_mean') else ''), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
