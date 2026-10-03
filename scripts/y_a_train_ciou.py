"""
y_a_train_ciou.py — [S1 / ตอบ R3-3] เทรน detector ตัวเดิมแต่เปลี่ยน box loss จาก MSE เป็น CIoU

ไม่แก้ไฟล์ gpu/train/y_a_train.py เลย: import โมดูลเดิม แล้วสลับคลาส ComputeLoss เป็นเวอร์ชัน CIoU
ทุกอย่างอื่นเหมือนเดิม (โมเดล, anchors, img 320, Adam 1e-3, batch 2, early stop, seed, validate)

วางไฟล์นี้ไว้ที่ gpu/train/ (ข้าง y_a_train.py) แล้วรันจาก root testocr:
    python gpu/train/y_a_train_ciou.py --seed 0 --epochs 300 --output_dir runs/ciou/plus_seed0
อาร์กิวเมนต์เพิ่ม:
    --box_weight W   น้ำหนักของ box loss (ค่าเริ่มต้น 1.0)
อาร์กิวเมนต์อื่นส่งต่อให้ main() ของ y_a_train.py ทั้งหมด (--seed --epochs --output_dir --no_csp4 ...)

ความต่างจาก MSE เดิม:
  - MSE เดิมคิดบนค่า raw (tx, ty, log w, log h) ของทุก cell (cell ที่ไม่มีป้ายถูกดึงเข้าหา 0)
  - CIoU คิดเฉพาะ cell/anchor ที่จับคู่กับป้าย โดย decode แบบเดียวกับตอน validate/บนบอร์ด:
        cx = grid_x + sigmoid(tx),  w = anchor_w * exp(tw)   (หน่วย grid)
  - objectness (BCE ทุก cell) เหมือนเดิมทุกประการ
"""
import os
import sys
import math

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import y_a_train as T  # noqa: E402  (โมดูลเดิม — ไม่ถูกแก้)


def bbox_ciou(b1, b2, eps=1e-7):
    """CIoU ของกล่องรูปแบบ (cx, cy, w, h) — tensor [N,4] ทั้งคู่ คืนค่า [N]"""
    b1x1, b1x2 = b1[:, 0] - b1[:, 2] / 2, b1[:, 0] + b1[:, 2] / 2
    b1y1, b1y2 = b1[:, 1] - b1[:, 3] / 2, b1[:, 1] + b1[:, 3] / 2
    b2x1, b2x2 = b2[:, 0] - b2[:, 2] / 2, b2[:, 0] + b2[:, 2] / 2
    b2y1, b2y2 = b2[:, 1] - b2[:, 3] / 2, b2[:, 1] + b2[:, 3] / 2

    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(min=0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(min=0)
    w1, h1 = b1[:, 2], b1[:, 3] + eps
    w2, h2 = b2[:, 2], b2[:, 3] + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (b1[:, 0] - b2[:, 0]) ** 2 + (b1[:, 1] - b2[:, 1]) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class ComputeLossCIoU(T.ComputeLoss):
    box_weight = 1.0

    def __call__(self, predictions, targets, img_size, batch_idx, log_batches):
        total_box_loss = 0.0
        total_obj_loss = 0.0

        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            num_preds = self.model.num_predictions
            grid_size = int(math.sqrt(pred.size(0) // num_preds))
            H, W = grid_size, grid_size
            pred = pred.view(num_preds, H, W, 5)
            obj_target = torch.zeros_like(pred[..., 4])

            # การจับคู่ป้าย→(anchor, cell) เหมือน ComputeLoss เดิมทุกบรรทัด
            # ถ้าสองป้ายตกช่องเดียวกัน ตัวหลังทับตัวแรก (เหมือนเดิม)
            matched = {}
            for t in target:
                x_min, y_min, x_max, y_max = t
                x_center = (x_min + x_max) / 2 / img_size * W
                y_center = (y_min + y_max) / 2 / img_size * H
                width = (x_max - x_min) / img_size * W
                height = (y_max - y_min) / img_size * H

                grid_x = min(int(x_center), W - 1)
                grid_y = min(int(y_center), H - 1)

                anchor_ws = self.anchors[:, 0]
                anchor_hs = self.anchors[:, 1]
                iou_scores = torch.stack([
                    self.iou((width, height), (aw, ah))
                    for aw, ah in zip(anchor_ws, anchor_hs)
                ])
                anchor_idx = torch.argmax(iou_scores).item()
                if anchor_idx >= num_preds:
                    continue

                obj_target[anchor_idx, grid_y, grid_x] = 1.0
                matched[(anchor_idx, grid_y, grid_x)] = [float(x_center), float(y_center),
                                                         float(width), float(height)]

            if matched:
                keys = list(matched.keys())
                a = torch.tensor([k[0] for k in keys], device=pred.device)
                gy = torch.tensor([k[1] for k in keys], device=pred.device)
                gx = torch.tensor([k[2] for k in keys], device=pred.device)
                tbox = torch.tensor([matched[k] for k in keys], device=pred.device, dtype=torch.float32)
                p = pred[a, gy, gx, :4].float()
                anc = self.anchors.to(pred.device).float()[a]
                pbox = torch.stack([
                    gx.float() + torch.sigmoid(p[:, 0]),
                    gy.float() + torch.sigmoid(p[:, 1]),
                    anc[:, 0] * torch.exp(p[:, 2].clamp(max=8.0)),
                    anc[:, 1] * torch.exp(p[:, 3].clamp(max=8.0)),
                ], dim=1)
                box_loss = self.box_weight * (1.0 - bbox_ciou(pbox, tbox)).sum()
            else:
                box_loss = pred[..., :4].sum() * 0.0

            obj_loss = self.bce_loss(pred[..., 4], obj_target).sum()
            total_box_loss += box_loss
            total_obj_loss += obj_loss

        total_loss = total_box_loss + total_obj_loss
        return total_loss / len(predictions), total_loss.item()


def main():
    argv = sys.argv[1:]
    if '--box_weight' in argv:
        i = argv.index('--box_weight')
        ComputeLossCIoU.box_weight = float(argv[i + 1])
        del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv
    T.ComputeLoss = ComputeLossCIoU   # train_yolov7_plus() เรียก ComputeLoss จาก globals ของโมดูล
    print(f"[S1] box loss = CIoU (box_weight={ComputeLossCIoU.box_weight}) · อย่างอื่นเหมือน y_a_train.py", flush=True)
    T.main()


if __name__ == '__main__':
    main()
