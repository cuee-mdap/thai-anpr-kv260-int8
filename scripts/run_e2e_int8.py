#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_e2e_int8.py — on-board (KV260) INT8 end-to-end ANPR evaluation
# Replicates the host FP32 protocol of gpu/eval/eval_end2end.py exactly:
#   detect: PIL RGB -> Resize(320,320) BILINEAR -> /255 -> (x-0.5)/0.5 -> DPU INT8
#           decode (sigmoid xy, exp wh, anchors, stride) -> conf>=0.10 -> NMS IoU 0.30
#           -> best box (max conf) -> scale to original -> clamp -> reject w/h<2
#   crop:   expand=0.0, img.crop(int(x1),int(y1),int(x2),int(y2))
#   ocr:    crop -> convert('L') -> Resize(220,70) BILINEAR -> /255 -> (x-0.5)/0.5
#           -> DPU INT8 -> argmax over 53 classes x 54 steps -> CTC greedy collapse
# INT8 quantization of inputs uses the runtime fix_point (2^6=64) and NHWC layout.
import os, sys, json, time, math
import faulthandler; faulthandler.enable()
import numpy as np
from PIL import Image
import xir, vart

MODEL_DIR = "/home/root/kv260_yolov7_ocr"
IMG_DIR   = "/home/root/e2e_eval/images"
OUT_JSON  = "/home/root/e2e_eval/board_int8_e2e.json"
LIST_FILE = "/home/root/e2e_eval/filelist.txt"

IMG_SIZE = 320
CONF_THR = 0.10
IOU_THR  = 0.30
ANCHORS  = np.array([[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]], dtype=np.float32)
W_OCR, H_OCR = 220, 70
LETTERS = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
DIGITS  = '0123456789'
CHARS   = ['<blank>'] + list(LETTERS + DIGITS)   # 53 classes

class DPU:
    """Runner with tensors queried once and buffers pre-allocated once."""
    def __init__(self, xmodel):
        # keep graph+subgraph referenced for the runner's lifetime (else segfault)
        self.g = xir.Graph.deserialize(xmodel)
        self.subs = [s for s in self.g.get_root_subgraph().toposort_child_subgraph()
                     if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        self.r = vart.Runner.create_runner(self.subs[0], "run")
        it = self.r.get_input_tensors()[0]
        ot = self.r.get_output_tensors()[0]
        self.in_dims = tuple(it.dims)
        self.out_dims = tuple(ot.dims)
        self.in_scale = 2.0 ** it.get_attr("fix_point")
        self.out_scale = 2.0 ** (-ot.get_attr("fix_point"))
        self.inp = [np.empty(self.in_dims, dtype=np.int8, order='C')]
        self.out = [np.empty(self.out_dims, dtype=np.int8, order='C')]

    def run(self, arr_int8):
        self.inp[0][...] = arr_int8.reshape(self.in_dims)
        jid = self.r.execute_async(self.inp, self.out)
        self.r.wait(jid)
        return self.out[0].astype(np.float32) * self.out_scale

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def nms_np(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        order = order[1:][iou <= thr]
    return keep

def detect_best_box(runner, img_pil, in_scale):
    W0, H0 = img_pil.size
    im = img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    xi = np.clip(np.round(x * in_scale), -128, 127).astype(np.int8)   # NHWC (320,320,3)
    out = runner.run(xi)                           # (1,40,40,15) float
    p = out[0].transpose(2, 0, 1).reshape(3, 5, 40, 40)   # (A,5,H,W)
    gh, gw = 40, 40
    stride = IMG_SIZE / gh
    gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
    bx = (sigmoid_np(p[:, 0]) + gx[None]) * stride
    by = (sigmoid_np(p[:, 1]) + gy[None]) * stride
    bw = ANCHORS[:, 0][:, None, None] * np.exp(p[:, 2]) * stride
    bh = ANCHORS[:, 1][:, None, None] * np.exp(p[:, 3]) * stride
    conf = sigmoid_np(p[:, 4])
    m = conf >= CONF_THR
    if not m.any():
        return None, None
    bx, by, bw, bh, sc = bx[m], by[m], bw[m], bh[m], conf[m]
    boxes = np.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], axis=1)
    keep = nms_np(boxes, sc, IOU_THR)
    boxes, sc = boxes[keep], sc[keep]
    best = boxes[sc.argmax()]; bconf = float(sc.max())
    sx, sy = W0 / IMG_SIZE, H0 / IMG_SIZE
    x1, y1 = max(0.0, best[0] * sx), max(0.0, best[1] * sy)
    x2, y2 = min(float(W0), best[2] * sx), min(float(H0), best[3] * sy)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, None
    return (x1, y1, x2, y2), bconf

def ocr_read(runner, crop_pil, in_scale):
    im = crop_pil.convert('L').resize((W_OCR, H_OCR), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    xi = np.clip(np.round(x * in_scale), -128, 127).astype(np.int8)   # (70,220)
    out = runner.run(xi.reshape(1, H_OCR, W_OCR, 1))                   # (1,1,54,53)
    seq = out.reshape(54, 53).argmax(axis=1)
    prev = -1; chars = []
    for c in seq:
        if c != 0 and c != prev:
            chars.append(CHARS[int(c)])
        prev = c
    return ''.join(chars)

def main():
    # [ตรวจความเสี่ยง R1-2] รับ xmodel ของ detector และไฟล์ผลลัพธ์จาก argv เพื่อเทียบ detector คนละตัวบนฉากเดียวกัน
    #   ใช้: python3 run_e2e_int8.py [limit] [detector.xmodel] [out.json]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    det_xmodel = sys.argv[2] if len(sys.argv) > 2 else os.path.join(MODEL_DIR, "dt_model.xmodel")
    out_json = sys.argv[3] if len(sys.argv) > 3 else OUT_JSON
    files = [l.strip() for l in open(LIST_FILE, encoding='utf-8') if l.strip()]
    if limit:
        files = files[:limit]
    det = DPU(det_xmodel)
    ocr = DPU(os.path.join(MODEL_DIR, "lpr_model.xmodel"))
    dsc, osc = det.in_scale, ocr.in_scale
    print(f"input scales: det x{dsc}  ocr x{osc}; images: {len(files)}", flush=True)
    res = []; t0 = time.time()
    for k, fn in enumerate(files):
        img = Image.open(os.path.join(IMG_DIR, fn))
        box, conf = detect_best_box(det, img, dsc)
        if box is None:
            res.append({"file": fn, "det_ok": False, "pred": ""})
        else:
            x1, y1, x2, y2 = box
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            pred = ocr_read(ocr, crop, osc)
            res.append({"file": fn, "det_ok": True, "conf": round(conf, 4),
                        "box": [round(v, 1) for v in box], "pred": pred})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(files)}  ({time.time()-t0:.1f}s)", flush=True)
    dt = time.time() - t0
    json.dump({"n": len(res), "elapsed_s": round(dt, 1),
               "ms_per_scene": round(1000 * dt / max(1, len(res)), 1),
               "results": res},
              open(out_json, "w", encoding='utf-8'), ensure_ascii=False)
    print(f"DONE {len(res)} scenes in {dt:.1f}s -> {out_json}", flush=True)

if __name__ == "__main__":
    main()
