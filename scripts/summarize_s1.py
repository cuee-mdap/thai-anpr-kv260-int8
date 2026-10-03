"""
summarize_s1.py — รวมผล S1 (CIoU) เทียบกับ MSE เดิม (Plus ซีด 0–2 จาก results/ablation_per_seed.csv)
    python summarize_s1.py --runs runs/ciou --seeds 0 1 2 --baseline ablation_per_seed.csv
อ่าน:  runs/ciou/plus_seedN/{float.log,int8.log}  (บรรทัด [COCO] ของ y_b_qt.py)
       runs/ciou/score_ciou_seedN.json, runs/ciou/score_deployed.json  (จาก score_e2e.py)
เขียน: runs/ciou/S1_SUMMARY.md + S1_per_seed.csv   (ไม่มีข้อมูลส่วนบุคคล)
"""
import os, re, csv, json, argparse, statistics as st

PAT = re.compile(r'\[COCO\] mAP@0\.5:0\.95=([\d.]+)%\s+mAP@0\.5=([\d.]+)%\s+mAP@0\.75=([\d.]+)%')


def last_coco(path):
    if not os.path.exists(path):
        return None
    hit = None
    for line in open(path, errors='ignore'):
        m = PAT.search(line)
        if m:
            hit = dict(mAP5095=float(m.group(1)), mAP50=float(m.group(2)), mAP75=float(m.group(3)))
    return hit


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    return f'{st.mean(v):.2f} ± {st.pstdev(v):.2f}' if len(v) > 1 else f'{v[0]:.2f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs/ciou')
    ap.add_argument('--seeds', nargs='+', default=['0', '1', '2'])
    ap.add_argument('--baseline', default='ablation_per_seed.csv')
    a = ap.parse_args()

    rows = []
    for s in a.seeds:
        d = os.path.join(a.runs, f'plus_seed{s}')
        fp = last_coco(os.path.join(d, 'float.log'))
        # int8.log ของ --mode full อาจมีทั้งบรรทัด float และ INT8 → บรรทัด [COCO] สุดท้ายคือ INT8
        q = last_coco(os.path.join(d, 'int8.log'))
        sc_p = os.path.join(a.runs, f'score_ciou_seed{s}.json')
        sc = json.load(open(sc_p)) if os.path.exists(sc_p) else {}
        rows.append(dict(loss='CIoU', seed=s,
                         fp32_50=fp and fp['mAP50'], fp32_75=fp and fp['mAP75'], fp32_5095=fp and fp['mAP5095'],
                         int8_50=q and q['mAP50'], int8_75=q and q['mAP75'], int8_5095=q and q['mAP5095'],
                         e2e_acc=sc.get('acc_all_pct'), e2e_acc_det=sc.get('acc_detected_pct'), mean_iou=sc.get('mean_iou'),
                         iou_ge_075=sc.get('frac_iou_ge_075')))

    base = {}
    if os.path.exists(a.baseline):
        for r in csv.DictReader(open(a.baseline)):
            if r['variant'] == 'plus' and r['seed'] in a.seeds:
                base[(r['seed'], r['precision'])] = r

    with open(os.path.join(a.runs, 'S1_per_seed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    col = lambda k: [r[k] for r in rows]
    bcol = lambda prec, k: [float(base[(s, prec)][k]) for s in a.seeds if (s, prec) in base]
    dep = os.path.join(a.runs, 'score_deployed.json')
    dep = json.load(open(dep)) if os.path.exists(dep) else {}

    L = ['# S1 — CIoU vs MSE (Plus, ซีด ' + ', '.join(a.seeds) + ', ± = population sd)', '',
         '| box loss | FP32 mAP@0.5 | FP32 mAP@0.5:0.95 | INT8 mAP@0.5 | INT8 mAP@0.75 | INT8 mAP@0.5:0.95 | ΔmAP@0.5:0.95 (INT8−FP32) |',
         '|---|---|---|---|---|---|---|']
    if base:
        d_b = [q - f for q, f in zip(bcol('INT8', 'mAP5095'), bcol('FP32', 'mAP5095'))]
        L.append(f"| MSE (เดิม) | {ms(bcol('FP32','mAP50'))} | {ms(bcol('FP32','mAP5095'))} | {ms(bcol('INT8','mAP50'))} | "
                 f"{ms(bcol('INT8','mAP75'))} | {ms(bcol('INT8','mAP5095'))} | {ms(d_b)} |")
    d_c = [q - f for q, f in zip(col('int8_5095'), col('fp32_5095')) if q is not None and f is not None]
    L.append(f"| CIoU | {ms(col('fp32_50'))} | {ms(col('fp32_5095'))} | {ms(col('int8_50'))} | "
             f"{ms(col('int8_75'))} | {ms(col('int8_5095'))} | {ms(d_c)} |")
    L += ['', '## บนบอร์ด KV260 (INT8, 648 ฉาก, exact match ทั้งป้าย)', '',
          '| detector | ถูกทั้งชุด (%) | ถูกเฉพาะที่ตรวจเจอ (%) | IoU เฉลี่ย | IoU ≥ 0.75 (%) |', '|---|---|---|---|---|']
    if dep:
        L.append(f"| deployed (MSE, 1 ตัว) | {dep['acc_all_pct']} | {dep['acc_detected_pct']} | {dep['mean_iou']} | {dep['frac_iou_ge_075']} |")
    L.append(f"| CIoU ({len(a.seeds)} ซีด) | {ms(col('e2e_acc'))} | {ms(col('e2e_acc_det'))} | {ms(col('mean_iou'))} | {ms(col('iou_ge_075'))} |")
    L += ['', '## รายซีด', '', '| seed | FP32 .5/.5:.95 | INT8 .5/.75/.5:.95 | e2e % | IoU เฉลี่ย |', '|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['seed']} | {r['fp32_50']}/{r['fp32_5095']} | {r['int8_50']}/{r['int8_75']}/{r['int8_5095']} | "
                 f"{r['e2e_acc']} | {r['mean_iou']} |")
    L += ['', 'หมายเหตุ: ตรวจว่า deployed (MSE) ได้เลขเดียวกับในเปเปอร์ก่อนใช้แถว CIoU',
          '(ถ้าไม่ตรง = ตัวให้คะแนน/ไฟล์ GT ไม่ใช่ชุดเดียวกัน)']
    open(os.path.join(a.runs, 'S1_SUMMARY.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
