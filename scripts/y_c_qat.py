"""
y_c_qat.py — [E2 / ตอบ R1-8 + R3-1] Quantization-Aware Training ของ detector

คำถามของ reviewer: "ถ้าไม่มี csp4 แล้วใช้ QAT แทน PTQ จะรอด quantization collapse ไหม?"
วิธี: ใช้ QatProcessor ของ vai_q_pytorch (fake-quant ตรงกับ DPU) fine-tune จาก checkpoint FP32 เดิม
      → ได้ "deployable weights" → ส่งเข้า flow เดิมของเปเปอร์ (y_b_qt.py PTQ/compile ของ Vitis-AI 2.5)
      → วัด mAP ด้วยโปรโตคอลเดียวกับ Table III ทุกประการ

*** รันในคอนเทนเนอร์ที่มี GPU + pytorch_nndct ***
  docker run --gpus all -v $(pwd):/workspace -w /workspace xilinx/vitis-ai-pytorch-gpu:3.0.0.001 bash -lc \
    "source /opt/vitis_ai/conda/etc/profile.d/conda.sh; conda activate vitis-ai-pytorch; \
     python3 gpu/quantize/y_c_qat.py --ckpt runs/ablation/base_seed1/yolov7_tiny_small_best.pth \
             --no_csp4 --epochs 60 --out_dir runs/qat/base_seed1"
"""
import os, sys, types, argparse, time
import torch
import torch.nn as nn

# ---- stub โมดูลที่คอนเทนเนอร์อาจไม่มี (ใช้แค่ตอน import y_a_train) -------------
for name, attrs in (('matplotlib', {'use': lambda *a, **k: None}),
                    ('matplotlib.pyplot', {}), ('matplotlib.patches', {})):
    try:
        __import__(name)
    except ImportError:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
try:
    import torchmetrics  # noqa: F401
except ImportError:
    for n in ('torchmetrics', 'torchmetrics.detection', 'torchmetrics.detection.mean_ap'):
        sys.modules[n] = types.ModuleType(n)

    class _MAP:  # stub — QAT ไม่ได้ใช้ validation mAP (ประเมินด้วย flow 2.5 ทีหลัง)
        def __init__(self, *a, **k):
            pass
    sys.modules['torchmetrics.detection.mean_ap'].MeanAveragePrecision = _MAP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train'))
from y_a_train import YOLOv7TinySmallPlus, LicensePlateDataset, ComputeLoss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402
from pytorch_nndct import QatProcessor  # noqa: E402
from pytorch_nndct.nn import QuantStub, DeQuantStub  # noqa: E402


class QATWrapper(nn.Module):
    """ห่อโมเดลเดิมด้วย QuantStub/DeQuantStub ตามที่ vai_q_pytorch ต้องการ (โครงข้างในไม่เปลี่ยน)"""

    def __init__(self, model):
        super().__init__()
        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()
        self.model = model

    def forward(self, x):
        x = self.quant_stub(x)
        y = self.model(x)
        return self.dequant_stub(y)

ANCHORS = [[10, 13], [16, 30], [33, 23]]
IMG_SIZE = 320


def collate(batch):   # เหมือน collate_fn ของ y_a_train (คืน list แล้ว stack ในลูป)
    return [b[0] for b in batch], [b[1] for b in batch]


def rebuild_from_deployable(dep_model, plain_model):
    """
    โมเดลที่ QatProcessor คืนมา "พับ BatchNorm เข้า conv แล้ว" และตั้งชื่อใหม่เป็น module_N
    จึงจับคู่ตรง ๆ กับโครงเดิมไม่ได้ → ประกอบกลับด้วยวิธี:
      · คัดน้ำหนัก conv (+bias) ตามลำดับการประกาศ ซึ่งตรงกับลำดับ conv ของโมเดลเดิม
      · ตั้ง BatchNorm ทุกตัวให้เป็น identity (weight=1, bias=0, mean=0, var=1, eps=0)
    ผลลัพธ์คือกราฟที่ให้ค่าเท่ากัน และเป็นรูปแบบที่ flow 2.5 (y_b_qt.py) รับได้ตามปกติ
    """
    dep_sd = dep_model.state_dict()
    conv_keys = [k for k, v in dep_sd.items() if k.endswith('.weight') and v.dim() == 4]
    plain_convs = [m for m in plain_model.modules() if isinstance(m, nn.Conv2d)]
    if len(conv_keys) != len(plain_convs):
        raise RuntimeError(f'จำนวน conv ไม่ตรงกัน: deployable {len(conv_keys)} vs original {len(plain_convs)}')
    for k, conv in zip(conv_keys, plain_convs):
        base = k[: -len('.weight')]
        w = dep_sd[k]
        if w.shape != conv.weight.shape:
            raise RuntimeError(f'shape ไม่ตรง: {k} {tuple(w.shape)} vs {tuple(conv.weight.shape)}')
        conv.weight.data.copy_(w.detach().cpu())
        b = dep_sd.get(base + '.bias')
        if conv.bias is not None:
            conv.bias.data.copy_(b.detach().cpu() if b is not None else torch.zeros_like(conv.bias))
    n_bn = 0
    for m in plain_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0); m.bias.data.zero_()
            m.running_mean.zero_(); m.running_var.fill_(1.0); m.eps = 0.0
            n_bn += 1
    return len(conv_keys), n_bn


def main():
    ap = argparse.ArgumentParser(description='[E2] QAT ของ detector ด้วย vai_q_pytorch')
    ap.add_argument('--ckpt', required=True, help='checkpoint FP32 ตั้งต้น')
    ap.add_argument('--no_csp4', action='store_true')
    ap.add_argument('--width_mult', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--dataset', default='data/thai_license_plate_dataset_for_yolov7')
    ap.add_argument('--out_dir', required=True)
    # [E2e] ตรึงสถิติ BatchNorm ระหว่าง QAT — จำเป็น!
    #   พบว่าถ้าไม่ตรึง: ตอนเทรน BN ใช้ batch statistics (batch=2) แต่ตอน to_deployable() จะ fold ด้วย
    #   running statistics ที่เพี้ยนไปมาก → โมเดลที่ export ออกมาพังทั้งที่ loss ตอนเทรนดูดี
    #   (ตรวจแล้ว: plus_seed0 loss ตอนเทรน 0.174 แต่ checkpoint ที่ export มี loss 133)
    ap.add_argument('--no_freeze_bn', action='store_true', help='ไม่ตรึง BN (พฤติกรรมเดิมที่มีปัญหา)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_csp4 = not args.no_csp4

    model = YOLOv7TinySmallPlus(nc=1, num_predictions=3, use_csp4=use_csp4, width_mult=args.width_mult)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model = model.to(dev).train()
    n_par = sum(p.numel() for p in model.parameters())
    print(f'[E2] ckpt={args.ckpt} csp4={use_csp4} width_mult={args.width_mult} params={n_par:,} dev={dev}', flush=True)

    # ---- QAT wrapper (fake-quant ตรงกับ DPU) ----
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev)
    wrapped = QATWrapper(model).to(dev).train()
    qat = QatProcessor(wrapped, inputs=dummy, bitwidth=8, device=dev)
    qmodel = qat.trainable_model().to(dev)

    tf = transforms.Compose([   # เหมือน y_a_train ทุกอย่าง
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    train_ds = LicensePlateDataset(os.path.join(args.dataset, 'images/train'),
                                   os.path.join(args.dataset, 'labels/train'), tf, img_size=IMG_SIZE)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate)
    print(f'[E2] train images = {len(train_ds)} | epochs = {args.epochs}', flush=True)

    loss_fn = ComputeLoss(model, torch.tensor(ANCHORS, dtype=torch.float32, device=dev), num_classes=1)   # ใช้ attribute num_predictions จากโมเดลเดิม
    opt = torch.optim.Adam(qmodel.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if not args.no_freeze_bn:
        n_bn = 0
        for mod in qmodel.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()                 # ใช้ running stats ตอน forward
                mod.momentum = 0.0         # ไม่อัปเดต running stats
                n_bn += 1
        print(f'[E2] ตรึง BatchNorm {n_bn} ตัว (train ใช้ running stats เดียวกับตอน deploy)', flush=True)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):   # --epochs 0 = ข้ามการเทรน (ใช้ตรวจ key mapping)
        qmodel.train()
        if not args.no_freeze_bn:          # .train() รีเซ็ต BN กลับ → ตรึงซ้ำทุก epoch
            for mod in qmodel.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        tot = 0.0
        for i, (imgs, targets) in enumerate(loader):
            imgs = torch.stack(imgs).to(dev).float()
            targets = [t.to(dev) for t in targets]
            out = qmodel(imgs)
            loss, _ = loss_fn(out, targets, IMG_SIZE, i, set())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        sched.step()
        print(f'[E2] epoch {ep}/{args.epochs} loss={tot/max(1,len(loader)):.4f} '
              f'({(time.time()-t0)/60:.1f} min)', flush=True)

    # ---- แปลงเป็นน้ำหนักที่ deploy ได้ แล้วเซฟในรูปแบบโมเดลเดิม ----
    dep_dir = os.path.join(args.out_dir, 'deployable')
    os.makedirs(dep_dir, exist_ok=True)
    # [E2f] เซฟสถานะของโมเดล QAT เองไว้ด้วย — ใช้ประเมิน mAP ของโมเดล fake-quant โดยตรง
    #   (พบว่า to_deployable() ทำให้ความแม่นยำหายไปทั้งสองสถาปัตยกรรม จึงต้องมีตัวเทียบที่ไม่ผ่านขั้นนี้)
    torch.save(qmodel.state_dict(), os.path.join(args.out_dir, 'qat_trainable_state.pth'))
    deployable = qat.to_deployable(qmodel, dep_dir)
    _ks = list(deployable.state_dict().keys())
    print(f'[E2] deployable state_dict: {len(_ks)} tensors | ตัวอย่าง: {_ks[:6]}', flush=True)
    plain = YOLOv7TinySmallPlus(nc=1, num_predictions=3, use_csp4=use_csp4, width_mult=args.width_mult)
    n_conv, n_bn = rebuild_from_deployable(deployable, plain)
    out_ckpt = os.path.join(args.out_dir, 'yolov7_tiny_small_best.pth')   # ชื่อเดียวกับ flow เดิม
    torch.save(plain.state_dict(), out_ckpt)
    print(f'[E2] saved {out_ckpt} | conv {n_conv} ตัว, BN identity {n_bn} ตัว', flush=True)

    # ---- ตรวจความถูกต้อง: โมเดลที่ประกอบกลับต้องให้ค่าใกล้เคียงของเดิม ----
    plain_gpu = plain.to(dev).eval(); deployable.eval()
    orig = YOLOv7TinySmallPlus(nc=1, num_predictions=3, use_csp4=use_csp4, width_mult=args.width_mult)
    orig.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    orig = orig.to(dev).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev)
        y_ref = deployable(x)
        y_new = plain_gpu(x)
        y_org = orig(x)
    print(f'[E2] verify-3way: |y_orig|mean={y_org.abs().mean():.4e} | '
          f'plain-vs-orig max|Δ|={(y_new-y_org).abs().max():.4e} | '
          f'deployable-vs-orig max|Δ|={(y_ref-y_org).abs().max():.4e}', flush=True)
    plain = plain_gpu.cpu()
    d = (y_ref - y_new).abs()
    print(f'[E2] verify: max|Δ|={d.max():.4e} mean|Δ|={d.mean():.4e} '
          f'(|y|mean={y_ref.abs().mean():.4e})', flush=True)
    print(f'[E2] DONE in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
