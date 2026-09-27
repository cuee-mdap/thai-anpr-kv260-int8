"""
iou_coupling.py — [E10d / ตอบ R3-3] คุณภาพกล่อง (IoU) ↔ ความถูกต้องของการอ่านป้าย

reviewer 3 ข้อ 3: mAP@0.5:0.95 ตก 5.3 จุดจาก INT8 กระทบ crop → กระทบ input ของ recognizer
สคริปต์นี้วัดจาก "ผลรันจริงบนบอร์ด INT8 ครบ 648 ฉาก" ที่มีอยู่แล้ว:
  - IoU ของกล่อง INT8 (บนบอร์ด) เทียบ ground-truth box
  - แบ่ง bin ตาม IoU แล้วดูอัตราการอ่านถูก (exact-match ทั้งป้าย) ต่อ bin
รัน: python gpu/eval/iou_coupling.py
ผลลัพธ์: outputs/iou_coupling/{per_scene.csv, SUMMARY.json}
"""
import os, json, csv
import numpy as np
from PIL import Image

BOARD = 'outputs/e2e_board_int8/board_int8_e2e_n648.json'
GTTXT = 'outputs/e4_653/per_image_4cfg.json'
IMG_DIR = 'data/thai_license_plate_dataset_for_yolov7/images/test'
LBL_DIR = 'data/thai_license_plate_dataset_for_yolov7/labels/test'
OUT = 'outputs/iou_coupling'


def gt_boxes(stem, W, H):
    """YOLO normalized -> pixel xyxy"""
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
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    board = json.load(open(BOARD))['results']
    gtmap = {r['file']: r['gt'] for r in json.load(open(GTTXT))}
    rows = []
    for r in board:
        stem = os.path.splitext(r['file'])[0]
        gt_txt = gtmap.get(stem)
        if gt_txt is None:
            continue
        ip = os.path.join(IMG_DIR, r['file'])
        if not os.path.exists(ip):
            continue
        with Image.open(ip) as im:
            W, H = im.size
        gts = gt_boxes(stem, W, H)
        box = r.get('box')
        best = max((iou(box, g) for g in gts), default=0.0) if (box and gts) else 0.0
        rows.append(dict(file=r['file'], det_ok=bool(r.get('det_ok')), iou=round(best, 4),
                         pred=r.get('pred', ''), gt=gt_txt,
                         correct=int(r.get('pred', '') == gt_txt)))

    with open(os.path.join(OUT, 'per_scene.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    det = [r for r in rows if r['det_ok']]
    ious = np.array([r['iou'] for r in det])
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    per_bin = []
    for lo, hi in bins:
        sel = [r for r in det if lo <= r['iou'] < hi]
        if sel:
            acc = 100.0 * sum(r['correct'] for r in sel) / len(sel)
            per_bin.append(dict(iou_bin=f'[{lo:.1f},{hi:.1f})', n=len(sel), acc_pct=round(acc, 2),
                                mean_iou=round(float(np.mean([r['iou'] for r in sel])), 3)))
    summary = dict(
        n_scenes=len(rows), n_detected=len(det),
        mean_iou=round(float(ious.mean()), 4), median_iou=round(float(np.median(ious)), 4),
        p10_iou=round(float(np.percentile(ious, 10)), 4), p05_iou=round(float(np.percentile(ious, 5)), 4),
        frac_iou_ge_050=round(100.0 * float((ious >= 0.5).mean()), 2),
        frac_iou_ge_075=round(100.0 * float((ious >= 0.75).mean()), 2),
        frac_iou_ge_090=round(100.0 * float((ious >= 0.9).mean()), 2),
        acc_overall_detected=round(100.0 * sum(r['correct'] for r in det) / len(det), 2),
        per_bin=per_bin,
        note='INT8 บนบอร์ด KV260 (n=648, protocol เดียวกับ Table VI) เทียบ GT box ของชุด test')
    json.dump(summary, open(os.path.join(OUT, 'SUMMARY.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
