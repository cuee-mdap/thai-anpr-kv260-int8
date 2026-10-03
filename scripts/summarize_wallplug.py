# -*- coding: utf-8 -*-
"""summarize_wallplug.py — [S3] สรุปผลจาก shelly_raw.csv ตามสเปกข้อ 6

  · ตัด 30 s แรกของแต่ละช่วง (label)
  · mean / sd / min / max ของ W ต่อช่องต่อ label
  · W เฉลี่ยจาก ΔWh ด้วย: W = ΔWh × 3600 / Δt(s)  (ตรวจไขว้กับ mean ของ apower)
  · รวมรอบ r1..r3 ของ label เดียวกันเป็นค่าเดียว (mean ± sd ระหว่างรอบ)
  · append ลง S3_wallplug_record.csv (ไม่เขียนทับ)

ใช้: python s1_261001/summarize_wallplug.py
"""
import csv, math, os, re, statistics as st
from collections import defaultdict

RAW = 'outputs/e9_power/shelly_raw.csv'
REC = 'journal/Submit261001/2_WORK_งานที่ทำ/ผลการทดลอง/S3_wallplug/S3_wallplug_record.csv'
SKIP_S = 30.0          # ตัด 30 วินาทีแรกของแต่ละช่วง

HDR = ['label_base', 'channel', 'name', 'n_reps', 'n_samples',
       'W_mean', 'W_sd_within', 'W_sd_between_reps', 'W_min', 'W_max',
       'W_from_dWh', 'V_mean', 'pf_mean']


def f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def main():
    if not os.path.exists(RAW):
        raise SystemExit(f'ไม่พบ {RAW}')
    rows = list(csv.DictReader(open(RAW, encoding='utf-8')))
    # group: (label, channel) -> samples
    g = defaultdict(list)
    for r in rows:
        t = f(r['t_s'])
        if t is None or t < SKIP_S:
            continue
        g[(r['label'], int(r['channel']))].append(r)

    per_rep = defaultdict(list)          # (label_base, ch) -> [ (mean, sd, mn, mx, n, dwh_w, v, pf) ]
    for (label, ch), ss in sorted(g.items()):
        W = [f(s['W']) for s in ss]
        W = [w for w in W if w is not None]
        if not W:
            continue
        base = re.sub(r'_r\d+$', '', label)
        V = [f(s['V']) for s in ss if f(s['V']) is not None]
        PF = [f(s['pf']) for s in ss if f(s['pf']) is not None]
        wh = [f(s['Wh_total']) for s in ss if f(s['Wh_total']) is not None]
        ts = [f(s['t_s']) for s in ss if f(s['t_s']) is not None]
        dwh_w = None
        if len(wh) >= 2 and len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
            dwh_w = (wh[-1] - wh[0]) * 3600.0 / (ts[-1] - ts[0])
        per_rep[(base, ch)].append(dict(
            mean=st.mean(W), sd=st.pstdev(W) if len(W) > 1 else 0.0,
            mn=min(W), mx=max(W), n=len(W), dwh=dwh_w,
            v=st.mean(V) if V else None, pf=st.mean(PF) if PF else None))

    name_of = {int(r['channel']): r['name'] for r in rows}
    out = []
    for (base, ch), reps in sorted(per_rep.items()):
        means = [x['mean'] for x in reps]
        dwh = [x['dwh'] for x in reps if x['dwh'] is not None]
        out.append([base, ch, name_of.get(ch, ''), len(reps), sum(x['n'] for x in reps),
                    round(st.mean(means), 2),
                    round(st.mean([x['sd'] for x in reps]), 2),
                    round(st.pstdev(means), 2) if len(means) > 1 else 0.0,
                    round(min(x['mn'] for x in reps), 2),
                    round(max(x['mx'] for x in reps), 2),
                    round(st.mean(dwh), 2) if dwh else '',
                    round(st.mean([x['v'] for x in reps if x['v'] is not None]), 1) if any(x['v'] for x in reps) else '',
                    round(st.mean([x['pf'] for x in reps if x['pf'] is not None]), 3) if any(x['pf'] for x in reps) else ''])

    os.makedirs(os.path.dirname(REC), exist_ok=True)
    new = not os.path.exists(REC)
    with open(REC, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HDR)
        w.writerows(out)

    print(f'ตัด {SKIP_S:.0f} s แรกของแต่ละช่วง · append {len(out)} แถว -> {REC}\n')
    hdr = f"{'label':18s}{'ch':>3s} {'name':8s}{'reps':>5s}{'W_mean':>9s}{'±sd_in':>8s}{'±sd_rep':>9s}{'W(ΔWh)':>9s}{'V':>7s}{'pf':>6s}"
    print(hdr); print('-' * len(hdr))
    for r in out:
        if r[5] < 0.5:            # ข้ามช่องว่าง
            continue
        print(f'{r[0]:18s}{r[1]:>3d} {r[2]:8s}{r[3]:>5d}{r[5]:>9.2f}{r[6]:>8.2f}{r[7]:>9.2f}'
              f'{(r[10] if r[10]!="" else float("nan")):>9.2f}{(r[11] if r[11]!="" else float("nan")):>7.1f}'
              f'{(r[12] if r[12]!="" else float("nan")):>6.2f}')


if __name__ == '__main__':
    main()
