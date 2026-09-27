"""
aggregate_ablation.py — [E5] รวมผล ablation ทุกซีด (เฟสเก่า + ซีดใหม่) เป็นตัวเลขสำหรับ Table III
  แหล่งข้อมูล (ผลรันจริงทั้งหมด):
    runs/ablation/results_{fp32,int8}.csv        seeds 0-2 ของ plus/base
    runs/ablation2/results_new_{fp32,int8}.csv   basewide 0-2 · base 3-5
    runs/ablation3/results_e5.csv + float.log    plus 3-5 · base 6-9 · basewide 3-5  (ซีดใหม่ของรอบแก้ไข)
รัน: python gpu/eval/aggregate_ablation.py
ผลลัพธ์: outputs/ablation_all/{per_seed.csv, SUMMARY.md}
"""
import os, csv, glob, re, json
import statistics as st

OUT = 'outputs/ablation_all'
COLLAPSE = 20.0   # เกณฑ์ "พัง" = mAP@0.5 ต่ำกว่า 20%


def add(rows, variant, seed, prec, m50, m75, m5095, src):
    rows.append(dict(variant=variant, seed=int(seed), precision=prec,
                     mAP50=float(m50), mAP75=float(m75), mAP5095=float(m5095), source=src))


def main():
    rows = []
    # ---- เฟส 1: runs/ablation (plus/base seed 0-2) ----
    for prec, f in (('FP32', 'runs/ablation/results_fp32.csv'), ('INT8', 'runs/ablation/results_int8.csv')):
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                add(rows, r['variant'], r['seed'], prec, r['mAP50'], r['mAP75'], r['mAP5095'], 'ablation')
    # ---- เฟส 2: runs/ablation2 (basewide 0-2, base 3-5) ----
    for prec, f in (('FP32', 'runs/ablation2/results_new_fp32.csv'), ('INT8', 'runs/ablation2/results_new_int8.csv')):
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                add(rows, r['variant'], r['seed'], prec, r['mAP50'], r['mAP75'], r['mAP5095'], 'ablation2')
    # ---- เฟส 3: runs/ablation3 (ซีดใหม่ของรอบแก้ไข R2) ----
    f = 'runs/ablation3/results_e5.csv'
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            if r['precision'] == 'INT8' and r['mAP50']:
                add(rows, r['variant'], r['seed'], 'INT8', r['mAP50'], r['mAP75'], r['mAP5095'], 'ablation3')
    for d in sorted(glob.glob('runs/ablation3/*/')):
        b = os.path.basename(d.rstrip('/'))
        m = re.match(r'(plus|base|basewide)_seed(\d+)', b)
        if not m or not os.path.exists(d + 'float.log'):
            continue
        txt = open(d + 'float.log', errors='ignore').read()
        def grab(pat):
            g = re.findall(pat, txt)
            return float(g[-1]) * 100 if g else None
        v50 = grab(r'Average Precision.*IoU=0\.50\s+\|\s+area=\s+all.*?=\s+([\d.]+)')
        v75 = grab(r'Average Precision.*IoU=0\.75\s+\|\s+area=\s+all.*?=\s+([\d.]+)')
        v5095 = grab(r'Average Precision.*IoU=0\.50:0\.95\s+\|\s+area=\s+all.*?=\s+([\d.]+)')
        if v50 is not None:
            add(rows, m.group(1), m.group(2), 'FP32', v50, v75 or 0, v5095 or 0, 'ablation3')

    os.makedirs(OUT, exist_ok=True)
    rows.sort(key=lambda r: (r['variant'], r['precision'], r['seed']))
    with open(f'{OUT}/per_seed.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- สรุปต่อกลุ่ม ----
    lines = ['# สรุป ablation ทุกซีด (ผลรันจริง)', '',
             f'เกณฑ์ "พัง" = mAP@0.5 < {COLLAPSE:.0f}%', '',
             '| variant | precision | n | mean ± std | min–max | พัง | ค่าต่อซีด |',
             '|---|---|---|---|---|---|---|']
    summary = {}
    for v in ('plus', 'base', 'basewide'):
        for p in ('FP32', 'INT8'):
            vals = [(r['seed'], r['mAP50']) for r in rows if r['variant'] == v and r['precision'] == p]
            if not vals:
                continue
            xs = [x for _, x in vals]
            coll = [f's{s}={x:.1f}' for s, x in vals if x < COLLAPSE]
            sd = st.pstdev(xs) if len(xs) > 1 else 0.0
            summary[f'{v}_{p}'] = dict(n=len(xs), mean=round(st.mean(xs), 2), std=round(sd, 2),
                                       min=round(min(xs), 2), max=round(max(xs), 2),
                                       collapsed=len(coll), per_seed={f's{s}': round(x, 2) for s, x in vals})
            lines.append(f'| {v} | {p} | {len(xs)} | {st.mean(xs):.2f} ± {sd:.2f} | '
                         f'{min(xs):.1f}–{max(xs):.1f} | {len(coll)}/{len(xs)} | '
                         f'{", ".join(f"{x:.1f}" for _, x in vals)} |')
    open(f'{OUT}/SUMMARY.md', 'w').write('\n'.join(lines) + '\n')
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
    print('\n'.join(lines))
    print(f'\nเขียน -> {OUT}/per_seed.csv, SUMMARY.md, summary.json')


if __name__ == '__main__':
    main()
