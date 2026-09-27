"""
models_ocr_baseline.py — [E4 / ตอบ R3-2 ฝั่ง recognizer] baseline ที่ deploy INT8 บน DPU ได้

reviewer 3 ข้อ 2: EasyOCR ที่ fine-tune แล้วได้ 98.15% สูงกว่า CRNN ของเรา (97.01%)
แต่ EasyOCR = 44.3 M พารามิเตอร์ + BiLSTM → **ลง DPUCZDX8G ไม่ได้** (ต้อง CPU fallback)
จึงต้องมี peer ที่ "ทั้ง deploy ได้และ quantize INT8 ได้" มาเทียบ → LPRNet (Zherzdev & Gruzdev)
เป็นตัวเลือกมาตรฐาน: CNN ล้วน ไม่มี recurrent ใช้ conv แยกแกน (1x3 / 3x1) ขนาดเล็กมาก

LPRNetSmall: ใช้เฉพาะ conv/BN/ReLU/maxpool/dropout → DPU รองรับครบ
output convention เหมือน CRNN เดิมเป๊ะ: (T, B, num_classes) เพื่อใช้ CTC loss/decoder/eval ชุดเดียวกัน
"""
import torch
import torch.nn as nn

HEIGHT, WIDTH = 70, 220   # เท่ากับ pipeline เดิม


class SmallBasicBlock(nn.Module):
    """บล็อกของ LPRNet: 1x1 -> 3x1 -> 1x3 -> 1x1 (แยกแกนเพื่อลดพารามิเตอร์)"""

    def __init__(self, cin, cout):
        super().__init__()
        c = cout // 4
        self.block = nn.Sequential(
            nn.Conv2d(cin, c, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(c, cout, 1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNetSmall(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),      # 70x220
            nn.MaxPool2d((2, 2)),                                                       # 35x110
            SmallBasicBlock(64, 128), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),                                           # 17x110
            SmallBasicBlock(128, 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            SmallBasicBlock(256, 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),                                           # 8x55
            nn.Dropout(dropout),
            nn.Conv2d(256, 256, kernel_size=(4, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 5x55
            nn.Dropout(dropout),
        )
        self._h = self._probe()
        self.fc = nn.Conv2d(256, num_classes, kernel_size=(self._h, 1))   # ยุบความสูงให้เหลือ 1

    def _probe(self):
        with torch.no_grad():
            y = self.backbone(torch.zeros(1, 1, HEIGHT, WIDTH))
        return y.shape[2]

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)          # [B, C, 1, W]
        x = x.squeeze(2)        # [B, C, W]
        return x.permute(2, 0, 1)   # [W(T), B, C]  — เหมือน CRNN เดิม


def build_ocr(arch, num_classes):
    if arch == 'lprnet':
        return LPRNetSmall(num_classes)
    raise ValueError(f'unknown ocr baseline arch: {arch}')


if __name__ == '__main__':
    m = LPRNetSmall(53)
    y = m(torch.zeros(2, 1, HEIGHT, WIDTH))
    print(f'LPRNetSmall params={sum(p.numel() for p in m.parameters()):,}  out={tuple(y.shape)} (T,B,C)')
