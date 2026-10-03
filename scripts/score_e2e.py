"""
score_e2e.py — ให้คะแนนผลรันบนบอร์ด (json จาก run_e2e_int8.py) ด้วยวิธีเดียวกับ iou_coupling.py
    python score_e2e.py --board outputs/e2e_board_int8/board_ciou_seed0.json \
                        --gt outputs/e4_653/per_image_4cfg.json --tag ciou_seed0 --out runs/ciou/score_ciou_seed0.json
ผลลัพธ์ (ไม่มีเลขทะเบียน ไม่มีชื่อไฟล์ภาพ → เอากลับเครื่องอื่นได้):
    ความถูกต้องทั้งป้าย (exact match) ต่อทุกฉาก และต่อฉากที่ตรวจเจอ, อัตราตรวจเจอ,
    IoU เฉลี่ย/มัธยฐาน, สัดส่วน IoU ≥ 0.5/0.75/0.9, ความถูกต้องแยกตาม IoU bin
"""
import os, json, argparse
import numpy as np
from PIL import Image

IMG_DIR = 'data/thai_license_plate_dataset_for_yolov7/images/test'
LBL_DIR = 'data/thai_license_plate_dataset_for_yolov7/labels/test'
BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def gt_boxes(stem, W, H):
    p = os.path.join(LBL_DIR, stem + '.txt')
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        v = line.split()
        if len(v) < 5:
            continue
        xc, yc, w, h = (float(t) for t in v[1:5])
        out.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
    return out


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    board = json.load(open(a.board, encoding='utf-8'))
    gtmap = {r['file']: r['gt'] for r in json.load(open(a.gt, encoding='utf-8'))}
    rows = []
    for r in board['results']:
        stem = os.path.splitext(r['file'])[0]
        gt = gtmap.get(stem)
        ip = os.path.join(IMG_DIR, r['file'])
        if gt is None or not os.path.exists(ip):
            continue
        with Image.open(ip) as im:
            W, H = im.size
        gts = gt_boxes(stem, W, H)
        box = r.get('box')
        best = max((iou(box, g) for g in gts), default=0.0) if (box and gts) else 0.0
        rows.append((bool(r.get('det_ok')), best, int(r.get('pred', '') == gt)))

    n = len(rows)
    det = [x for x in rows if x[0]]
    ious = np.array([x[1] for x in det]) if det else np.zeros(1)
    per_bin = []
    for lo, hi in BINS:
        sel = [x for x in det if lo <= x[1] < hi]
        if sel:
            per_bin.append(dict(iou_bin=f'[{lo:.1f},{hi:.1f})', n=len(sel),
                                acc_pct=round(100.0 * sum(x[2] for x in sel) / len(sel), 2)))
    s = dict(
        tag=a.tag, n_scenes=n, n_detected=len(det),
        n_correct=sum(x[2] for x in rows),
        acc_all_pct=round(100.0 * sum(x[2] for x in rows) / max(1, n), 2),
        acc_detected_pct=round(100.0 * sum(x[2] for x in det) / max(1, len(det)), 2),
        det_rate_pct=round(100.0 * len(det) / max(1, n), 2),
        mean_iou=round(float(ious.mean()), 4), median_iou=round(float(np.median(ious)), 4),
        frac_iou_ge_050=round(100.0 * float((ious >= 0.5).mean()), 2),
        frac_iou_ge_075=round(100.0 * float((ious >= 0.75).mean()), 2),
        frac_iou_ge_090=round(100.0 * float((ious >= 0.9).mean()), 2),
        ms_per_scene=board.get('ms_per_scene'),
        per_bin=per_bin)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(s, open(a.out, 'w'), indent=1)
    print(f"{a.tag}: ถูก {s['n_correct']}/{n} = {s['acc_all_pct']}% · ตรวจเจอ {s['det_rate_pct']}% · "
          f"IoU เฉลี่ย {s['mean_iou']} · IoU≥0.75 {s['frac_iou_ge_075']}%  → {a.out}")


if __name__ == '__main__':
    main()
