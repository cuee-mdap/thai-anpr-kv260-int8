"""
models_baseline.py — [E3 / ตอบ R1-2 + R3-2] baseline ที่ "deploy บน DPU ได้จริง" ขนาดใกล้เคียงของเรา

reviewer 1 ข้อ 2 และ reviewer 3 ข้อ 2: YOLOv8n/YOLOv5nu เป็น FP32 upper bound ที่ quantize/deploy ไม่ได้
(C2f / SPPF / DFL head ไม่รองรับบน DPUCZDX8G) → ต้องมี peer ที่ผ่าน PTQ INT8 + compile + รันบนบอร์ดได้

YOLOv4TinySmall = แบ็กโบน CSPDarknet53-tiny (โครงของ YOLOv4-tiny: CSP split/concat + LeakyReLU + maxpool)
ตัดที่ stride 8 เพื่อให้ "หัว/แอนคอร์/loss/โปรโตคอลประเมิน เหมือนของเราทุกอย่าง" — ต่างกันแค่แบ็กโบน
  · ใช้เฉพาะ op ที่ DPUCZDX8G รองรับ: conv, BN, LeakyReLU, maxpool, concat, slice
  · output เหมือน YOLOv7TinySmallPlus เป๊ะ: (B, 40*40*3, 5) → ใช้ ComputeLoss/eval/quantize เดิมได้ทันที
  · width: ปรับให้พารามิเตอร์ใกล้ 1.35 M ของรุ่นที่เสนอ (--arch yolov4tiny --width_mult ...)
"""
import torch
import torch.nn as nn


def _cbl(cin, cout, k=3, s=1):
    """Conv-BN-LeakyReLU (LeakyReLU 0.1 = op มาตรฐานของ YOLOv4-tiny และ DPU รองรับ)"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class CSPTinyBlock(nn.Module):
    """บล็อก CSP ของ YOLOv4-tiny: split ครึ่งช่อง -> 2 conv -> concat -> transition -> concat กับ route"""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.conv1 = _cbl(c, c, 3)
        self.conv2 = _cbl(c // 2, c // 2, 3)
        self.conv3 = _cbl(c // 2, c // 2, 3)
        self.conv4 = _cbl(c, c, 1)

    def forward(self, x):
        x = self.conv1(x)
        route = x
        x = x[:, self.c // 2:, :, :]          # slice ครึ่งหลัง (DPU: strided_slice)
        x = self.conv2(x)
        r2 = x
        x = self.conv3(x)
        x = torch.cat([x, r2], dim=1)          # -> c
        x = self.conv4(x)
        return torch.cat([route, x], dim=1)    # -> 2c


class YOLOv4TinySmall(nn.Module):
    """YOLOv4-tiny backbone ที่หัวเหมือน YOLOv7TinySmallPlus (stride 8, 3 anchors, 1 class)"""

    def __init__(self, nc=1, num_predictions=3, width_mult=1.0):
        super().__init__()
        self.num_predictions = num_predictions
        w = lambda c: max(8, int(round(c * width_mult / 8)) * 8)   # noqa: E731
        c1, c2, c3, c4 = w(24), w(48), w(96), w(192)

        self.stem = nn.Sequential(_cbl(3, c1, 3, 1), _cbl(c1, c2, 3, 2))   # 320 -> 160
        self.csp1 = CSPTinyBlock(c2)                                        # -> 2*c2
        self.pool1 = nn.MaxPool2d(2, 2)                                     # 160 -> 80
        self.csp2 = CSPTinyBlock(2 * c2)                                    # -> 4*c2
        self.pool2 = nn.MaxPool2d(2, 2)                                     # 80 -> 40
        self.csp3 = CSPTinyBlock(4 * c2)                                    # -> 8*c2
        self.neck = nn.Sequential(_cbl(8 * c2, c4, 3), _cbl(c4, c3, 1))
        self.head = nn.Conv2d(c3, nc * 5 * num_predictions, 1, 1, 0)
        self.bn_head = nn.BatchNorm2d(nc * 5 * num_predictions)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.csp1(x))
        x = self.pool2(self.csp2(x))
        x = self.csp3(x)
        x = self.neck(x)
        out = self.head(x)
        out = self.bn_head(out)
        out = out.view(x.size(0), self.num_predictions, 5, x.size(2), x.size(3))
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out.view(x.size(0), -1, 5)


def build_baseline(arch, nc=1, num_predictions=3, width_mult=1.0):
    if arch == 'yolov4tiny':
        return YOLOv4TinySmall(nc, num_predictions, width_mult)
    raise ValueError(f'unknown baseline arch: {arch}')


if __name__ == '__main__':
    for wm in (0.75, 1.0, 1.25, 1.5):
        m = YOLOv4TinySmall(width_mult=wm)
        n = sum(p.numel() for p in m.parameters())
        y = m(torch.zeros(1, 3, 320, 320))
        print(f'width_mult={wm:<5} params={n:,}  out={tuple(y.shape)}')
