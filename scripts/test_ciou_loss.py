"""
test_ciou_loss.py — ตรวจ y_a_train_ciou.py ก่อนเทรนจริง (รันบน CPU ได้ ~ไม่กี่วินาที)
    cp test_ciou_loss.py y_a_train_ciou.py gpu/train/
    python gpu/train/test_ciou_loss.py
ตรวจ 4 อย่าง:
  1) bbox_ciou ตรงกับ torchvision.ops.complete_box_iou_loss (ถ้ามี)
  2) ถ้า prediction decode แล้วตรงกล่องจริงพอดี → box loss ≈ 0
  3) gradient ไหลกลับถึง pred และ loss ลดลงเมื่อทำ gradient descent
  4) objectness loss เท่ากับ ComputeLoss เดิม (ส่วนนี้ต้องไม่เปลี่ยน)
"""
import os, sys, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import y_a_train_ciou as C  # noqa: E402

T = C.T
torch.manual_seed(0)


class Dummy:
    num_predictions = 3


def xyxy(b):
    return torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                        b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


# 1) เทียบกับ torchvision
b1 = torch.rand(64, 4) * torch.tensor([40, 40, 12, 4]) + torch.tensor([0, 0, 0.5, 0.5])
b2 = torch.rand(64, 4) * torch.tensor([40, 40, 12, 4]) + torch.tensor([0, 0, 0.5, 0.5])
mine = 1 - C.bbox_ciou(b1, b2)
try:
    from torchvision.ops import complete_box_iou_loss
    ref = complete_box_iou_loss(xyxy(b1), xyxy(b2), reduction='none')
    err = (mine - ref).abs().max().item()
    print(f'[1] CIoU vs torchvision: max |diff| = {err:.2e}')
    assert err < 1e-4
except ImportError:
    print('[1] ข้าม (torchvision ไม่มี complete_box_iou_loss)')

# ชุดทดสอบ: grid 40, 3 anchors (หน่วย grid), ป้าย 2 ป้ายในภาพ 320
anchors = torch.tensor([[10., 13.], [16., 30.], [33., 23.]])
img = 320
W = 40
targets = [torch.tensor([[100., 150., 180., 175.], [22., 31., 61., 46.]])]
lf_new = C.ComputeLossCIoU(Dummy(), anchors)
lf_old = T.ComputeLoss(Dummy(), anchors)

# 2) สร้าง pred ที่ decode แล้วตรงกล่องจริง
pred = torch.full((3 * W * W, 5), -6.0)
p5 = pred.view(3, W, W, 5)
for t in targets[0]:
    xc, yc = (t[0] + t[2]) / 2 / img * W, (t[1] + t[3]) / 2 / img * W
    w, h = (t[2] - t[0]) / img * W, (t[3] - t[1]) / img * W
    gx, gy = int(xc), int(yc)
    a = int(torch.argmax(torch.stack([lf_new.iou((w, h), (aw, ah)) for aw, ah in anchors])))
    fx, fy = xc - gx, yc - gy
    p5[a, gy, gx, 0] = math.log(fx / (1 - fx))
    p5[a, gy, gx, 1] = math.log(fy / (1 - fy))
    p5[a, gy, gx, 2] = math.log(w / anchors[a, 0])
    p5[a, gy, gx, 3] = math.log(h / anchors[a, 1])
    p5[a, gy, gx, 4] = 6.0

# แยก box/obj ออกจากกันด้วยการตั้ง box_weight
C.ComputeLossCIoU.box_weight = 0.0
obj_only_new = lf_new([pred], targets, img, 0, [])[1]
C.ComputeLossCIoU.box_weight = 1.0
total_new = lf_new([pred], targets, img, 0, [])[1]
box_new = total_new - obj_only_new
print(f'[2] box loss เมื่อ pred ตรงกล่องจริง = {box_new:.2e}')
assert abs(box_new) < 1e-4

# 4) objectness ต้องเท่ากับ BCE แบบเดิม (target=1 เฉพาะ cell ที่จับคู่ป้าย)
bt = torch.zeros(3, W, W, 4)
for t in targets[0]:
    xc, yc = (t[0] + t[2]) / 2 / img * W, (t[1] + t[3]) / 2 / img * W
    w, h = (t[2] - t[0]) / img * W, (t[3] - t[1]) / img * W
    gx, gy = int(xc), int(yc)
    a = int(torch.argmax(torch.stack([lf_old.iou((w, h), (aw, ah)) for aw, ah in anchors])))
    bt[a, gy, gx] = torch.tensor([xc - gx, yc - gy, math.log(w / anchors[a, 0]), math.log(h / anchors[a, 1])])
ot = (bt.abs().sum(-1) > 0).float()
obj_old = torch.nn.functional.binary_cross_entropy_with_logits(p5[..., 4], ot, reduction='sum').item()
print(f'[4] objectness: ใหม่ {obj_only_new:.4f} · BCE แบบเดิม {obj_old:.4f}')
assert abs(obj_only_new - obj_old) < 1e-3

# 3) gradient ไหลและ loss ลดลง
q = (pred.clone() + 0.5 * torch.randn_like(pred)).requires_grad_(True)
opt = torch.optim.SGD([q], lr=0.05)
first = None
for it in range(200):
    opt.zero_grad()
    loss, val = lf_new([q], targets, img, 0, [])
    loss.backward()
    opt.step()
    first = val if first is None else first
print(f'[3] loss รวม: เริ่ม {first:.3f} → หลัง 200 step {val:.3f}')
assert val < first

print('ผ่านทุกข้อ — ใช้ y_a_train_ciou.py เทรนได้')
