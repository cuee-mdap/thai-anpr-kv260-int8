# -*- coding: utf-8 -*-
"""parse_b2_logs.py — อ่านค่า COCO mAP ทั้ง 3 ระดับจาก log ที่ run_b2_full.sh สร้างไว้
   (สคริปต์ bash เดิม grep ผิดเพราะ COCO เว้นวรรคหลัง IoU=0.50 หลายช่อง — ไม่ต้องรันโมเดลใหม่)
   รัน: python s1_261001/parse_b2_logs.py
"""
import csv, glob, os, re

ROOT = os.environ.get('TESTOCR_ROOT', '.')   # รันจาก repo root หรือกำหนด TESTOCR_ROOT
E6 = os.path.join(ROOT, 'runs/e6_ood')
OUT = os.path.join(E6, 'B2_CHECK/b2_parsed.csv')

PAT = re.compile(
    r'Average Precision\s+\(AP\) @\[\s*IoU=([\d.:]+)\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([\d.\-]+)')


def maps(path):
    """คืน dict {'0.50':x, '0.75':y, '0.50:0.95':z} หน่วยเปอร์เซ็นต์ (เอาบล็อกสุดท้ายในไฟล์)"""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for iou, val in PAT.findall(open(path, errors='ignore').read()):
        out[iou] = float(val) * 100          # บล็อกหลังทับบล็อกหน้า = ผลล่าสุด
    return out


def row(group, anchors, ds, var, seed, prec, m):
    f = lambda k: ('%.2f' % m[k]) if k in m else ''
    return [group, anchors, ds, var, seed, prec, f('0.50'), f('0.75'), f('0.50:0.95')]


def main():
    rows = []
    # ---- ชุดที่เปเปอร์ใช้: anchor ไทยเดิม ----
    for d in sorted(glob.glob(f'{E6}/ccpd_*_seed*/') + glob.glob(f'{E6}/oid_*_seed*/')):
        b = os.path.basename(d.rstrip('/'))
        ds, rest = b.split('_', 1)
        var = rest.split('_')[0]
        seed = b.split('seed')[-1]
        fp = maps(os.path.join(d, 'float_full.log')) or maps(os.path.join(d, 'float.log'))
        i8 = maps(os.path.join(d, 'int8.log'))
        if fp:
            rows.append(row('paper', 'thai-default', ds, var, seed, 'FP32', fp))
        if i8:
            rows.append(row('paper', 'thai-default', ds, var, seed, 'INT8', i8))
    # ---- ชุด anchor k-means (E6b · OID เท่านั้น) ----
    for d in sorted(glob.glob(f'{E6}/b_oid_*_seed*/')):
        b = os.path.basename(d.rstrip('/'))
        var = b.replace('b_oid_', '').split('_')[0]
        seed = b.split('seed')[-1]
        fp = maps(os.path.join(d, 'float_full.log'))
        i8 = maps(os.path.join(d, 'int8_full.log'))
        if fp:
            rows.append(row('refit', 'oid-kmeans', 'oid', var, seed, 'FP32', fp))
        if i8:
            rows.append(row('refit', 'oid-kmeans', 'oid', var, seed, 'INT8', i8))

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'anchors', 'dataset', 'variant', 'seed', 'precision',
                    'mAP50', 'mAP75', 'mAP5095'])
        w.writerows(rows)

    print(f'เขียน {OUT}  ({len(rows)} แถว)\n')
    hdr = f"{'group':6s}{'anchors':13s}{'ds':6s}{'var':6s}{'seed':5s}{'prec':6s}{'@0.5':>8s}{'@0.75':>8s}{'@.5:.95':>9s}"
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f'{r[0]:6s}{r[1]:13s}{r[2]:6s}{r[3]:6s}{r[4]:5s}{r[5]:6s}{r[6]:>8s}{r[7]:>8s}{r[8]:>9s}')


if __name__ == '__main__':
    main()
