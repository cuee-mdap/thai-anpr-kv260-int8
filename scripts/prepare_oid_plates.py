"""
prepare_oid_plates.py — [E6 / ตอบ R1-4 + R3-1] เตรียมชุด out-of-domain จาก Open Images V7
  คลาส "Vehicle registration plate" (/m/01jfm_) — สาธารณะ ดาวน์โหลดตรงจาก S3 ไม่ต้องขอสิทธิ์
  ครอบคลุมหลายประเทศ/หลายรูปแบบป้าย → ใช้ทดสอบว่าข้อสรุปเรื่อง csp4 ยังจริงนอกโดเมนไทยไหม

ทำ 2 ขั้น:
  1) filter annotation (val+test ของ OID) → รายชื่อภาพ + กล่อง
  2) ดาวน์โหลดภาพจาก S3 + แปลงเป็นรูปแบบ YOLO เหมือน data/thai_license_plate_dataset_for_yolov7/
     (กล่องของ OID เป็น normalized XMin/XMax/YMin/YMax อยู่แล้ว → แปลงเป็น xc yc w h ได้ตรง ๆ)

รัน: python gpu/eval/prepare_oid_plates.py --workers 16
ผลลัพธ์: data/oid_plate_yolov7/{images,labels}/{train,val,test}
"""
import os, csv, random, argparse, collections
from concurrent.futures import ThreadPoolExecutor
import urllib.request

CLS = '/m/01jfm_'
TMP = 'data/_oid_tmp'
OUT = 'data/oid_plate_yolov7'
S3 = 'https://open-images-dataset.s3.amazonaws.com/{split}/{img}.jpg'


def load_boxes():
    per_img = collections.defaultdict(list)     # (split, image_id) -> [box,...]
    for split in ('validation', 'test'):
        src = os.path.join(TMP, 'val.csv' if split == 'validation' else 'test.csv')
        with open(src) as f:
            for x in csv.DictReader(f):
                if x['LabelName'] != CLS:
                    continue
                xmin, xmax = float(x['XMin']), float(x['XMax'])
                ymin, ymax = float(x['YMin']), float(x['YMax'])
                if xmax - xmin < 1e-4 or ymax - ymin < 1e-4:
                    continue
                per_img[(split, x['ImageID'])].append(
                    ((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin))
    return per_img


def fetch(job):
    split, img, dst = job
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return True
    try:
        with urllib.request.urlopen(S3.format(split=split, img=img), timeout=60) as r, open(dst, 'wb') as f:
            f.write(r.read())
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split_ratio', default='0.70,0.10,0.20', help='train,val,test')
    args = ap.parse_args()

    per_img = load_boxes()
    keys = sorted(per_img)
    random.Random(args.seed).shuffle(keys)
    r_tr, r_va, _ = (float(v) for v in args.split_ratio.split(','))
    n = len(keys); n_tr = int(n * r_tr); n_va = int(n * r_va)
    parts = {'train': keys[:n_tr], 'val': keys[n_tr:n_tr + n_va], 'test': keys[n_tr + n_va:]}
    print(f'[E6] ภาพที่มีป้ายทั้งหมด {n} | train {len(parts["train"])} val {len(parts["val"])} test {len(parts["test"])}')

    jobs = []
    for part, ks in parts.items():
        os.makedirs(f'{OUT}/images/{part}', exist_ok=True)
        os.makedirs(f'{OUT}/labels/{part}', exist_ok=True)
        for split, img in ks:
            jobs.append((split, img, f'{OUT}/images/{part}/{img}.jpg'))

    ok = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, good in enumerate(ex.map(fetch, jobs), 1):
            ok += bool(good)
            if i % 250 == 0:
                print(f'  ดาวน์โหลด {i}/{len(jobs)} (สำเร็จ {ok})', flush=True)
    print(f'[E6] ดาวน์โหลดเสร็จ {ok}/{len(jobs)}')

    # เขียน label เฉพาะภาพที่โหลดสำเร็จ (คลาสเดียว = 0 เหมือนชุดไทย)
    written = collections.Counter()
    for part, ks in parts.items():
        for split, img in ks:
            ip = f'{OUT}/images/{part}/{img}.jpg'
            if not (os.path.exists(ip) and os.path.getsize(ip) > 1000):
                if os.path.exists(ip):
                    os.remove(ip)
                continue
            with open(f'{OUT}/labels/{part}/{img}.txt', 'w') as f:
                for xc, yc, w, h in per_img[(split, img)]:
                    f.write(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
            written[part] += 1
    print('[E6] พร้อมใช้:', dict(written), '->', OUT)


if __name__ == '__main__':
    main()
