"""fit_anchors.py — [E6] หา anchor 3 ตัวด้วย k-means จากกล่องจริงของแต่ละชุดข้อมูล
   anchor เป็น hyperparameter ของ detector ไม่ใช่สถาปัตยกรรมที่กำลังทดสอบ (csp4)
   การใช้ anchor ที่พอดีกับแต่ละโดเมนคือการควบคุมตัวแปรที่ถูกต้อง: เปลี่ยนเฉพาะ anchor ส่วนโครงข่ายเหมือนเดิม
รัน: python gpu/eval/fit_anchors.py data/ccpd_plate_yolov7
"""
import sys, glob, numpy as np

IMG, STRIDE = 320, 8   # ของเรา anchor ถูกคูณด้วย stride ตอน decode


def main(root, k=3):
    wh = []
    for f in glob.glob(f'{root}/labels/train/*.txt'):
        for l in open(f):
            v = l.split()
            if len(v) >= 5:
                wh.append([float(v[3]) * IMG / STRIDE, float(v[4]) * IMG / STRIDE])
    X = np.array(wh)
    rng = np.random.default_rng(0)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(80):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    C = C[C[:, 0].argsort()]
    print(f'{root}: n={len(X)} | anchors (หน่วยเดียวกับของเดิม) =',
          '[' + ', '.join(f'[{a:.1f}, {b:.1f}]' for a, b in C) + ']')
    print('   ขนาดจริงที่ 320px:', ', '.join(f'{a*STRIDE:.0f}x{b*STRIDE:.0f}' for a, b in C))
    return C


if __name__ == '__main__':
    for r in (sys.argv[1:] or ['data/thai_license_plate_dataset_for_yolov7',
                               'data/oid_plate_yolov7', 'data/ccpd_plate_yolov7']):
        main(r)
