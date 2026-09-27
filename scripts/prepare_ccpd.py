"""
prepare_ccpd.py — [E6 / ตอบ R1-4 + R3-1] เตรียม CCPD2020 (ป้ายทะเบียนจีน "ป้ายเขียว") เป็นรูปแบบ YOLO
  CCPD เข้ารหัสกล่องไว้ในชื่อไฟล์: ...-<x1&y1_x2&y2>-<4 มุม>-<เลขป้าย>-...
  ฟิลด์ที่ 3 (นับจาก 1) = มุมซ้ายบน & มุมขวาล่าง เป็นพิกเซล → แปลงเป็น YOLO (xc yc w h normalized)
รัน: python gpu/eval/prepare_ccpd.py
ผลลัพธ์: data/ccpd_plate_yolov7/{images,labels}/{train,val,test}   (คลาสเดียว = 0 เหมือนชุดไทย)
"""
import os, shutil, argparse
from PIL import Image

SRC = 'data/_ccpd_tmp/CCPD2020/ccpd_green'
OUT = 'data/ccpd_plate_yolov7'


def parse_box(name):
    """คืน (x1, y1, x2, y2) พิกเซล จากชื่อไฟล์ CCPD"""
    parts = name.split('-')
    if len(parts) < 4:
        return None
    try:
        tl, br = parts[2].split('_')
        x1, y1 = (int(v) for v in tl.split('&'))
        x2, y2 = (int(v) for v in br.split('&'))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_per_split', type=int, default=0, help='0 = ใช้ทั้งหมด')
    args = ap.parse_args()

    total = {}
    for split in ('train', 'val', 'test'):
        src = os.path.join(SRC, split)
        if not os.path.isdir(src):
            print(f'  ข้าม {split} (ไม่พบ {src})'); continue
        os.makedirs(f'{OUT}/images/{split}', exist_ok=True)
        os.makedirs(f'{OUT}/labels/{split}', exist_ok=True)
        files = sorted(f for f in os.listdir(src) if f.endswith('.jpg'))
        if args.max_per_split:
            files = files[:args.max_per_split]
        ok = 0
        for i, fn in enumerate(files):
            box = parse_box(fn)
            if box is None:
                continue
            sp = os.path.join(src, fn)
            try:
                with Image.open(sp) as im:
                    W, H = im.size
            except Exception:
                continue
            x1, y1, x2, y2 = box
            xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            if not (0 < xc < 1 and 0 < yc < 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            stem = f'{split}_{i:06d}'
            shutil.copy(sp, f'{OUT}/images/{split}/{stem}.jpg')
            with open(f'{OUT}/labels/{split}/{stem}.txt', 'w') as f:
                f.write(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
            ok += 1
            if ok % 1000 == 0:
                print(f'  {split}: {ok}/{len(files)}', flush=True)
        total[split] = ok
        print(f'[E6/CCPD] {split}: {ok} ภาพ')
    print('[E6/CCPD] เสร็จ', total, '->', OUT)


if __name__ == '__main__':
    main()
