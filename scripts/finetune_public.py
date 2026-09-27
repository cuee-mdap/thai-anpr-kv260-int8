#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# finetune_public.py — [1.4] fine-tune the deployed Thai-plate detector on a public
# LP dataset subset (OpenALPR us) to test whether the architecture + training recipe
# TRANSFERS with target-domain fine-tuning (zero-shot transfer already measured ~0).
# Reuses the real training components verbatim from y_a_train.py:
#   YOLOv7TinySmallPlus, LicensePlateDataset, ComputeLoss, collate_fn, same transform,
#   same Adam/lr and loss; initialization = the deployed Thai checkpoint.
import os, sys, argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def _anchors_from_env(default=((10.0, 13.0), (16.0, 30.0), (33.0, 23.0))):
    """[E6] อ่าน anchor จาก env ANCHORS เช่น ANCHORS="11.7,2.4;16.1,3.1;22.0,3.9"
       ไม่ตั้งค่า = ใช้ของเดิมเป๊ะ (anchor เป็น hyperparameter ของโดเมน ไม่ใช่สถาปัตยกรรม)"""
    import os as _o
    v = _o.environ.get('ANCHORS')
    if not v:
        return [list(a) for a in default]
    return [[float(x) for x in p.split(',')] for p in v.split(';')]



HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from y_a_train import YOLOv7TinySmallPlus, LicensePlateDataset, ComputeLoss, collate_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--init_ckpt',
        default=os.path.join(HERE, '../../outputs/trained_yolov7_tiny_small_plus/yolov7_tiny_small_best.pth'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    # [E6] ทำ ablation ซ้ำบนโดเมนอื่น → ต้อง fine-tune ได้ทั้ง Plus และ base
    ap.add_argument('--no_csp4', action='store_true', help='fine-tune รุ่นที่ไม่มี csp4 (สำหรับ E6)')
    ap.add_argument('--width_mult', type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = 320
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    tr = LicensePlateDataset(os.path.join(args.dataset_path, 'images/train'),
                             os.path.join(args.dataset_path, 'labels/train'), tf, img_size)
    va = LicensePlateDataset(os.path.join(args.dataset_path, 'images/val'),
                             os.path.join(args.dataset_path, 'labels/val'), tf, img_size)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val = DataLoader(va, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"train {len(tr)}  val {len(va)}  device {device}")

    use_csp4 = not args.no_csp4
    model = YOLOv7TinySmallPlus(nc=1, use_csp4=use_csp4, width_mult=args.width_mult).to(device)
    print(f"[E6] variant={'Plus(csp4)' if use_csp4 else 'base(no csp4)'} width_mult={args.width_mult} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    sd = torch.load(args.init_ckpt, map_location=device)
    model.load_state_dict(sd.get('model_state_dict', sd))
    print("init from", os.path.basename(args.init_ckpt))

    anchors = torch.tensor(_anchors_from_env(), device=device)
    loss_fn = ComputeLoss(model, anchors, num_classes=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for images, targets in trl:
            images = torch.stack(images).to(device).float()
            targets = [t.to(device) for t in targets]
            opt.zero_grad()
            preds = model(images)
            loss, _ = loss_fn(preds, targets, img_size=img_size, batch_idx=0, log_batches=set())
            loss.backward(); opt.step()
            tot += loss.item()
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for images, targets in val:
                images = torch.stack(images).to(device).float()
                targets = [t.to(device) for t in targets]
                loss, _ = loss_fn(model(images), targets, img_size=img_size, batch_idx=0, log_batches=set())
                vtot += loss.item()
        vavg = vtot / max(1, len(val))
        star = ''
        if vavg < best_val:
            best_val = vavg
            torch.save({'model_state_dict': model.state_dict()}, args.out)
            star = '  *saved'
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d}  train {tot/max(1,len(trl)):.4f}  val {vavg:.4f}{star}", flush=True)
    print("done; best val loss", round(best_val, 4), "->", args.out)

if __name__ == '__main__':
    main()
