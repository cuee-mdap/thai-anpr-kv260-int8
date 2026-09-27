"""collect_e6.py — ดึงค่า mAP จริงจาก log ของ E6 ทุกโฟลเดอร์ (แก้บั๊ก parser ที่ช่องว่างไม่ตรง)"""
import re, glob, os, csv

MARK = {'mAP50': 'IoU=0.50 ', 'mAP75': 'IoU=0.75 ', 'mAP5095': 'IoU=0.50:0.95 '}


def grab(path):
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, errors='ignore'):
        if 'Average Precision' not in line or 'area=   all' not in line:
            continue
        for k, mark in MARK.items():
            if mark in line:
                try:
                    out[k] = round(float(line.rsplit('=', 1)[1].strip()) * 100, 2)
                except ValueError:
                    pass
    return out if out.get('mAP50') is not None else None


rows = []
for d in sorted(glob.glob('runs/e6_ood/*_seed*/')):
    b = os.path.basename(d.rstrip('/'))
    tag = b[2:] if b.startswith('b_') else b          # b_oid_plus_seed0 -> oid_plus_seed0
    anch = 'fitted' if b.startswith('b_') else 'thai-default'
    parts = tag.split('_')
    ds, var, seed = parts[0], parts[1], parts[2].replace('seed', '')
    for prec, f in (('FP32', d + 'float.log'), ('INT8', d + 'int8.log')):
        r = grab(f)
        if r:
            rows.append(dict(dataset=ds, anchors=anch, variant=var, seed=seed, precision=prec, **r))

os.makedirs('outputs/e6_ood', exist_ok=True)
with open('outputs/e6_ood/results_all.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f'{"dataset":<6}{"anchors":<14}{"variant":<9}{"seed":<5}{"prec":<6}{"mAP@0.5":>9}{"@0.75":>8}{"@0.5:0.95":>11}')
for r in rows:
    print(f'{r["dataset"]:<6}{r["anchors"]:<14}{r["variant"]:<9}{r["seed"]:<5}{r["precision"]:<6}'
          f'{r["mAP50"]:>9}{r["mAP75"] if r["mAP75"] is not None else "-":>8}{r["mAP5095"]:>11}')
print('\n-> outputs/e6_ood/results_all.csv')
