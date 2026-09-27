"""
act_stats.py — [E1 / ตอบ R1-1 + R3-1] สถิติการกระจายของ activation ต่อเลเยอร์ + error จำลองของ INT8

ทำไม: reviewer 1 ข้อ 1 บอกว่า claim เรื่อง csp4 มีแต่ข้อสันนิษฐาน ไม่มีสถิติ activation
สคริปต์นี้วัดจริงจาก checkpoint ที่เทรนไว้แล้ว (inference อย่างเดียว ไม่ต้องเทรนใหม่):

  pass 1 : เก็บ moment ต่อเลเยอร์ (n, Σx, Σx², Σx³, Σx⁴), min/max รวม, min/max ต่อ channel
           -> mean, std, dynamic range, excess kurtosis, per-channel range spread,
              และ fix_point แบบ DPU (สเกล power-of-two ต่อ tensor เหมือน Vitis-AI)
  pass 2 : ใช้สเกลจาก pass 1 -> จำลอง quantize INT8 (round/clamp) วัด SQNR + cosine similarity
           + histogram (หา p99.99) + สัดส่วน outlier |x-μ| > 3σ

รัน:
  conda run -n lpr_pt python gpu/eval/act_stats.py                 # ทุกโมเดลที่มี checkpoint
  conda run -n lpr_pt python gpu/eval/act_stats.py --max_images 200 --only plus_seed0,base_seed1
ผลลัพธ์: outputs/act_stats/{per_layer.csv, prehead_summary.csv, SUMMARY.json}
"""
import os, sys, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from y_a_train import YOLOv7TinySmallPlus   # noqa: E402

# ---- รายชื่อโมเดล: name -> (checkpoint, use_csp4, width_mult) -------------------
MODELS = {}
for s in (0, 1, 2):
    MODELS[f'plus_seed{s}'] = (f'runs/ablation/plus_seed{s}/yolov7_tiny_small_best.pth', True, 1.0)
    MODELS[f'base_seed{s}'] = (f'runs/ablation/base_seed{s}/yolov7_tiny_small_best.pth', False, 1.0)
for s in (3, 4, 5):
    MODELS[f'base_seed{s}'] = (f'runs/ablation2/base_seed{s}/yolov7_tiny_small_best.pth', False, 1.0)
    MODELS[f'basewide_seed{s-3}'] = (f'runs/ablation2/basewide_seed{s-3}/yolov7_tiny_small_best.pth', False, 1.5)
# [R2 รอบตรวจสุดท้าย] ซีดชุดขยายของ ablation3 — ให้สถิติ activation ครอบคลุม "ทุก" checkpoint ของ Table III
for s in (3, 4, 5):
    MODELS[f'plus_seed{s}'] = (f'runs/ablation3/plus_seed{s}/yolov7_tiny_small_best.pth', True, 1.0)
    MODELS[f'basewide_seed{s}'] = (f'runs/ablation3/basewide_seed{s}/yolov7_tiny_small_best.pth', False, 1.5)
for s in (6, 7, 8, 9):
    MODELS[f'base_seed{s}'] = (f'runs/ablation3/base_seed{s}/yolov7_tiny_small_best.pth', False, 1.0)
MODELS['plus_deployed'] = ('outputs/trained_yolov7_tiny_small_plus/yolov7_tiny_small_best.pth', True, 1.0)


class CalibSet(Dataset):
    """ชุด calibration เดียวกับ y_b_qt.py (cv2 BGR -> resize 320 -> ToTensor -> Normalize(0.5,0.5))"""
    def __init__(self, images_dir, img_size=320, max_images=500):
        self.dir = images_dir
        self.files = sorted(f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png')))
        if max_images > 0:
            self.files = self.files[:max_images]
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.dir, self.files[i]))
        return self.tf(img)


def dpu_fix_point(absmax: float) -> int:
    """fix_point แบบ DPU: q = clamp(round(x * 2^p), -128, 127) — เลือก p ให้ครอบ absmax พอดี"""
    if absmax <= 0:
        return 0
    return int(math.floor(math.log2(127.0 / absmax)))


def hook_targets(model):
    """เลือกจุดที่มีความหมายต่อ quantization: output ของทุก conv/bn/relu + บล็อก csp + input ของ head"""
    names = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
            names.append((n, m))
    return names


@torch.no_grad()
def analyse(model, loader, device):
    acc = {}          # layer -> dict ของ moment
    prehead = {'name': 'PRE-HEAD (head input)'}

    def mk_hook(name):
        def hook(_m, _inp, out):
            t = out.detach().float()
            a = acc.setdefault(name, dict(n=0, s1=0.0, s2=0.0, s3=0.0, s4=0.0,
                                          mn=float('inf'), mx=float('-inf'),
                                          ch_mn=None, ch_mx=None, shape=tuple(t.shape[1:])))
            a['n'] += t.numel()
            a['s1'] += t.sum().item()
            a['s2'] += (t * t).sum().item()
            a['s3'] += (t ** 3).sum().item()
            a['s4'] += (t ** 4).sum().item()
            a['mn'] = min(a['mn'], t.min().item())
            a['mx'] = max(a['mx'], t.max().item())
            if t.dim() == 4:
                cmn = t.amin(dim=(0, 2, 3)); cmx = t.amax(dim=(0, 2, 3))
                a['ch_mn'] = cmn if a['ch_mn'] is None else torch.minimum(a['ch_mn'], cmn)
                a['ch_mx'] = cmx if a['ch_mx'] is None else torch.maximum(a['ch_mx'], cmx)
        return hook

    def prehead_hook(_m, inp):
        t = inp[0].detach().float()
        a = acc.setdefault('PRE-HEAD', dict(n=0, s1=0.0, s2=0.0, s3=0.0, s4=0.0,
                                            mn=float('inf'), mx=float('-inf'),
                                            ch_mn=None, ch_mx=None, shape=tuple(t.shape[1:])))
        a['n'] += t.numel(); a['s1'] += t.sum().item(); a['s2'] += (t * t).sum().item()
        a['s3'] += (t ** 3).sum().item(); a['s4'] += (t ** 4).sum().item()
        a['mn'] = min(a['mn'], t.min().item()); a['mx'] = max(a['mx'], t.max().item())
        cmn = t.amin(dim=(0, 2, 3)); cmx = t.amax(dim=(0, 2, 3))
        a['ch_mn'] = cmn if a['ch_mn'] is None else torch.minimum(a['ch_mn'], cmn)
        a['ch_mx'] = cmx if a['ch_mx'] is None else torch.maximum(a['ch_mx'], cmx)

    handles = [m.register_forward_hook(mk_hook(n)) for n, m in hook_targets(model)]
    handles.append(model.head.register_forward_pre_hook(prehead_hook))

    # ---------- pass 1 : moments ----------
    for xb in loader:
        model(xb.to(device))

    stats = {}
    for name, a in acc.items():
        n = a['n']; mu = a['s1'] / n
        var = max(a['s2'] / n - mu * mu, 1e-20); sd = math.sqrt(var)
        m4 = a['s4'] / n - 4 * mu * a['s3'] / n + 6 * mu * mu * a['s2'] / n - 3 * mu ** 4
        kurt = m4 / (var * var) - 3.0
        absmax = max(abs(a['mn']), abs(a['mx']))
        ch_rng = (a['ch_mx'] - a['ch_mn']).cpu().numpy() if a['ch_mn'] is not None else np.array([0.0])
        nz = ch_rng[ch_rng > 0]
        med = float(np.median(nz)) if nz.size else 0.0   # ใช้ median ของ channel ที่ไม่ตาย (บาง channel range=0)
        dead_pct = 100.0 * float((ch_rng <= 0).mean()) if ch_rng.size else 0.0
        stats[name] = dict(shape=a['shape'], mean=mu, std=sd, min=a['mn'], max=a['mx'],
                           absmax=absmax, dyn_range=a['mx'] - a['mn'], kurtosis_excess=kurt,
                           ch_range_spread=(float(ch_rng.max()) / med if med > 0 else float('nan')),
                           dead_channel_pct=dead_pct,
                           fix_point=dpu_fix_point(absmax))

    # ---------- pass 2 : quant error + histogram + outliers ----------
    NB = 4096
    q = {k: dict(se=0.0, sig=0.0, dot=0.0, nq=0.0, nx=0.0, out3=0, n=0,
                 hist=np.zeros(NB), lo=stats[k]['min'], hi=stats[k]['max']) for k in stats}
    acc.clear()

    def mk_hook2(name):
        def hook(_m, _inp, out):
            _collect(name, out.detach().float())
        return hook

    def _collect(name, t):
        st = stats[name]; qq = q[name]
        p = st['fix_point']; scale = 2.0 ** p
        tq = torch.clamp(torch.round(t * scale), -128, 127) / scale
        err = t - tq
        qq['se'] += (err * err).sum().item()
        qq['sig'] += (t * t).sum().item()
        qq['dot'] += (t * tq).sum().item()
        qq['nq'] += (tq * tq).sum().item()
        qq['n'] += t.numel()
        qq['out3'] += int(((t - st['mean']).abs() > 3 * st['std']).sum().item())
        h = torch.histc(t, bins=NB, min=qq['lo'], max=qq['hi']).cpu().numpy()
        qq['hist'] += h

    def prehead_hook2(_m, inp):
        _collect('PRE-HEAD', inp[0].detach().float())

    for h in handles:
        h.remove()
    handles = [m.register_forward_hook(mk_hook2(n)) for n, m in hook_targets(model)]
    handles.append(model.head.register_forward_pre_hook(prehead_hook2))
    for xb in loader:
        model(xb.to(device))
    for h in handles:
        h.remove()

    for name, st in stats.items():
        qq = q[name]
        st['sqnr_db'] = 10 * math.log10(qq['sig'] / qq['se']) if qq['se'] > 0 else float('inf')
        st['cos_sim'] = qq['dot'] / math.sqrt(qq['sig'] * qq['nq']) if qq['sig'] * qq['nq'] > 0 else float('nan')
        st['outlier_3sigma_pct'] = 100.0 * qq['out3'] / qq['n']
        # p99.99 ของ |x| จาก histogram
        h = qq['hist']; edges = np.linspace(qq['lo'], qq['hi'], NB + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        order = np.argsort(np.abs(centers))
        cum = np.cumsum(h[order]) / max(h.sum(), 1)
        idx = np.searchsorted(cum, 0.9999)
        st['p99_99_abs'] = float(abs(centers[order][min(idx, NB - 1)]))
        st['p99_99_over_absmax'] = st['p99_99_abs'] / st['absmax'] if st['absmax'] > 0 else float('nan')
        # สัดส่วนระดับ INT8 ที่ถูกใช้จริง (bulk ใช้กี่ระดับจาก 256)
        step = 2.0 ** (-st['fix_point'])
        st['levels_used_p9999'] = min(255.0, 2 * st['p99_99_abs'] / step)
        st['levels_used_pct'] = 100.0 * st['levels_used_p9999'] / 255.0
    return stats


def main():
    ap = argparse.ArgumentParser(description='[E1] per-layer activation statistics + simulated INT8 error')
    ap.add_argument('--calib_dir', default='data/thai_license_plate_dataset_for_yolov7/images/train')
    ap.add_argument('--max_images', type=int, default=500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', default='outputs/act_stats')
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = CalibSet(args.calib_dir, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    print(f'[E1] calib = {len(ds)} images from {args.calib_dir} | device={device}')

    want = set(x.strip() for x in args.only.split(',') if x.strip())
    rows, prehead_rows, summary = [], [], {}

    for name, (ckpt, csp4, wm) in MODELS.items():
        if want and name not in want:
            continue
        if not os.path.exists(ckpt):
            print(f'  SKIP {name} (ไม่พบ {ckpt})'); continue
        model = YOLOv7TinySmallPlus(nc=1, num_predictions=3, use_csp4=csp4, width_mult=wm).to(device)
        sd = torch.load(ckpt, map_location=device)
        sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd); model.eval()
        nparam = sum(p.numel() for p in model.parameters())
        print(f'  >> {name}: csp4={csp4} width_mult={wm} params={nparam:,}')
        st = analyse(model, loader, device)
        variant = 'Plus(csp4)' if csp4 else ('Base-Wide' if wm != 1.0 else 'base')
        for layer, s in st.items():
            rows.append(dict(model=name, variant=variant, params=nparam, layer=layer,
                             shape='x'.join(map(str, s['shape'])), **{k: v for k, v in s.items() if k != 'shape'}))
        ph = st['PRE-HEAD']
        prehead_rows.append(dict(model=name, variant=variant, params=nparam,
                                 **{k: v for k, v in ph.items() if k != 'shape'}))
        summary[name] = dict(variant=variant, params=nparam, prehead=ph['shape'] and list(ph['shape']),
                             prehead_absmax=ph['absmax'], prehead_min=ph['min'], prehead_max=ph['max'],
                             prehead_kurtosis=ph['kurtosis_excess'], prehead_sqnr_db=ph['sqnr_db'],
                             prehead_levels_used_pct=ph['levels_used_pct'],
                             prehead_ch_range_spread=ph['ch_range_spread'],
                             worst_layer_sqnr=min((v['sqnr_db'], k) for k, v in st.items())[::-1])
        print(f'     PRE-HEAD: absmax={ph["absmax"]:.3f} kurt={ph["kurtosis_excess"]:.1f} '
              f'SQNR={ph["sqnr_db"]:.2f} dB levels={ph["levels_used_pct"]:.1f}%')
        del model
        torch.cuda.empty_cache()

    import csv as _csv
    if rows:
        with open(os.path.join(args.out, 'per_layer.csv'), 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        with open(os.path.join(args.out, 'prehead_summary.csv'), 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(prehead_rows[0].keys())); w.writeheader(); w.writerows(prehead_rows)
        with open(os.path.join(args.out, 'SUMMARY.json'), 'w') as f:
            json.dump(dict(calib_images=len(ds), calib_dir=args.calib_dir, models=summary), f, indent=1)
        print(f'[E1] เขียนผลแล้ว -> {args.out}/ (per_layer.csv {len(rows)} แถว)')


if __name__ == '__main__':
    main()
