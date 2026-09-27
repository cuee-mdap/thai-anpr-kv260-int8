import os
import sys
import time
import json
import argparse
import random
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_nndct.apis import torch_quantizer
import torch.nn.functional as F

# ตั้งค่า UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DIVIDER = '-----------------------------------------'

# กำหนดค่าคงที่ (ต้องตรงกับที่ใช้ในการฝึก)
WIDTH, HEIGHT = 220, 70
LETTERS = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
DIGITS = '0123456789'

# การแมปตัวอักษร
CHARACTERS = LETTERS + DIGITS
char_list = ['<blank>'] + list(CHARACTERS)
char_to_idx = {char: idx for idx, char in enumerate(char_list)}
idx_to_char = {idx: char for idx, char in enumerate(char_list)}

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        ann_path = os.path.join(root_dir, split, f"{split}_annotations.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotations not found at {ann_path}")

        with open(ann_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.split, 'images',
                                self.annotations[idx]['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        text = self.annotations[idx]['annotations'][0]['text']
        text = text.split('\n')[0]  # ดึงเฉพาะบรรทัดแรก
        text = ''.join(c for c in text if c != ' ')
        target = [char_to_idx.get(char, 0) for char in text]
        target_length = len(target)
        target = torch.tensor(target, dtype=torch.long)

        return image, target, target_length, text

def collate_fn(batch):
    images, targets, target_lengths, texts = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, targets, target_lengths, texts

def decode_predictions(preds):
    """Convert model output to text"""
    decoded_preds = []
    for pred in preds:
        pred = pred.cpu().numpy()
        pred_indices = [p for i, p in enumerate(pred) if (p != 0 and (i == 0 or p != pred[i - 1]))]
        pred_chars = [idx_to_char.get(idx, '') for idx in pred_indices]
        decoded_preds.append(''.join(pred_chars))
    return decoded_preds

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # CNN layers (ต้องตรงกับที่ใช้ในการฝึก)
        self.cnn = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Convolutional Block 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Convolutional Block 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Convolutional Block 4
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Convolutional Block 5
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Convolutional Block 6
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Convolutional Block 7
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self._initialize_cnn_output_size()

        # ปรับปรุง Temporal Convolutional Layers
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),  # dilation=1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),  # dilation=1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),  # dilation=1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(0.5)  # เพิ่ม Dropout เพื่อป้องกัน Overfitting

        # เปลี่ยนเลเยอร์ Linear เป็นเลเยอร์ Convolutional
        self.fc = nn.Conv2d(512, num_classes, kernel_size=(self.cnn_output_height, 1), stride=(1, 1))

    def _initialize_cnn_output_size(self):
        dummy_input = torch.zeros(1, 1, HEIGHT, WIDTH)
        dummy_output = self.cnn(dummy_input)
        _, c, h, w = dummy_output.size()
        self.cnn_output_size = c
        self.cnn_output_height = h
        self.cnn_output_width = w

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        x = self.temporal_conv(x)  # [B, C, H, W]
        x = self.dropout(x)
        x = self.fc(x)  # [B, num_classes, 1, W]
        x = x.squeeze(2)  # [B, num_classes, W]
        x = x.permute(2, 0, 1)  # [W, B, num_classes]
        return x

@torch.no_grad()
def test_random_samples(model, device, test_loader, num_samples=100):
    """Test model on random samples and show predictions"""
    print(f"\nTesting {num_samples} random samples...")
    model.eval()

    # Get random indices
    total_samples = len(test_loader.dataset)
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    correct_words = 0
    correct_chars = 0
    total_chars = 0
    total_time = 0

    print("\nPrediction Results:")
    print("-" * 80)
    print(f"{'Image':25} {'Predicted':20} {'Ground Truth':20} {'Time (ms)':10} {'Correct':8}")
    print("-" * 80)

    with torch.no_grad():
        for idx in random_indices:
            # Get single sample
            image, target, target_length, text = test_loader.dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Time prediction
            start_time = time.time()
            output = model(image)
            output = nn.LogSoftmax(dim=2)(output)
            process_time = (time.time() - start_time) * 1000
            total_time += process_time

            # Decode prediction
            preds = output.argmax(dim=2).transpose(1, 0)
            pred_texts = decode_predictions(preds)
            prediction = pred_texts[0]

            # Calculate accuracy
            char_matches = sum(1 for a, b in zip(prediction, text) if a == b)
            correct_chars += char_matches
            total_chars += len(text)
            if prediction == text:
                correct_words += 1

            # Get image name
            image_name = test_loader.dataset.annotations[idx]['file_name']

            # Print result
            print(f"{image_name[:25]:25} {prediction:20} {text:20} {process_time:10.2f} {'✓' if prediction == text else '✗':8}")

    # Calculate statistics
    word_accuracy = 100.0 * correct_words / num_samples
    char_accuracy = 100.0 * correct_chars / total_chars
    avg_time = total_time / num_samples

    print("\nSummary:")
    print("-" * 60)
    print(f"Word Accuracy: {correct_words}/{num_samples} ({word_accuracy:.2f}%)")
    print(f"Character Accuracy: {correct_chars}/{total_chars} ({char_accuracy:.2f}%)")
    print(f"Average Processing Time: {avg_time:.2f}ms")

    return {
        'word_accuracy': word_accuracy,
        'char_accuracy': char_accuracy,
        'avg_process_time': avg_time,
        'total_samples': num_samples
    }

def levenshtein(a, b):
    """ระยะแก้ไขระดับอักขระ (edit distance) ใช้คำนวณ CER — ไม่พึ่ง lib ภายนอก"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + cost))  # substitution
        prev = cur
    return prev[lb]

@torch.no_grad()
def evaluate_ocr(model, device, test_loader, label='model', show_samples=20):
    """
    ประเมิน OCR แบบ deterministic บนชุด test 'ทั้งหมด' (ไม่สุ่ม, ไม่ augmentation) [E2]
    คืน Word Accuracy (ตรงเป๊ะ), Character Accuracy (= 100 - CER), CER, เวลาเฉลี่ย/ภาพ
    ใช้ฟังก์ชันเดียวกันทั้ง float และ INT8 เพื่อให้เทียบกันได้
    """
    model.eval()
    correct_words = 0
    total_words = 0
    total_edit = 0          # ผลรวม edit distance ทุกภาพ
    total_ref_chars = 0     # ผลรวมจำนวนอักขระ ground-truth
    total_time = 0.0
    shown = 0

    print(f"\nEvaluating OCR ({label}) on FULL test set (deterministic)...")
    print("-" * 80)
    print(f"{'Predicted':22} {'Ground Truth':22} {'CER':>6} {'OK':>3}")
    print("-" * 80)

    for data, targets, target_lengths, texts in test_loader:
        data = data.to(device)
        start = time.time()
        output = model(data)
        output = nn.LogSoftmax(dim=2)(output)
        total_time += (time.time() - start) * 1000.0
        preds = output.argmax(dim=2).transpose(1, 0)  # [B, W]
        pred_texts = decode_predictions(preds)
        for pred, gt in zip(pred_texts, texts):
            ed = levenshtein(pred, gt)
            total_edit += ed
            total_ref_chars += len(gt)
            total_words += 1
            ok = (pred == gt)
            if ok:
                correct_words += 1
            if shown < show_samples:
                print(f"{pred:22} {gt:22} {ed / max(1, len(gt)):6.2f} {'✓' if ok else '✗':>3}")
                shown += 1

    word_acc = 100.0 * correct_words / max(1, total_words)
    cer = 100.0 * total_edit / max(1, total_ref_chars)
    char_acc = 100.0 - cer
    avg_time = total_time / max(1, total_words)

    print(f"\nSummary ({label}):")
    print("-" * 60)
    print(f"Samples            : {total_words}")
    print(f"Word Accuracy      : {correct_words}/{total_words} ({word_acc:.2f}%)")
    print(f"Character Accuracy : {char_acc:.2f}%  (= 100 - CER)")
    print(f"CER                : {cer:.2f}%  (edit={total_edit}, ref_chars={total_ref_chars})")
    print(f"Avg Time / image   : {avg_time:.2f} ms")

    return {
        'word_accuracy': word_acc,
        'char_accuracy': char_acc,
        'cer': cer,
        'avg_process_time': avg_time,
        'total_samples': total_words,
    }

def save_ocr_results(out_dir, metrics, model_label):
    """บันทึกผล OCR ลงไฟล์ (รวม CER)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(out_dir, f'eval_{model_label}_{timestamp}.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"OCR Evaluation Results ({model_label})\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: CRNN ({model_label})\n")
        f.write(f"Input Size: {HEIGHT}x{WIDTH}\n")
        f.write(f"Eval: FULL test set, deterministic, no augmentation\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Samples            : {metrics['total_samples']}\n")
        f.write(f"  Word Accuracy      : {metrics['word_accuracy']:.2f}%\n")
        f.write(f"  Character Accuracy : {metrics['char_accuracy']:.2f}% (= 100 - CER)\n")
        f.write(f"  CER                : {metrics['cer']:.2f}%\n")
        f.write(f"  Avg Time / image   : {metrics['avg_process_time']:.2f} ms\n")
    print(f"\nDetailed results saved to {results_file}")
    return results_file

def export_model(quantizer, model_dir, metrics):
    """Export quantized model and metadata"""
    try:
        # Export quantization config
        quantizer.export_quant_config()

        # Export xmodel with fixed input shape
        quantizer.export_xmodel(
            deploy_check=True,
            output_dir=model_dir
        )

        # Export metadata
        meta = {
            "model_info": {
                "input_shape": [1, 1, HEIGHT, WIDTH],
                "character_set": char_list,
                "model_version": "1.0",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            "quantization_info": {
                "quantizer": "Vitis AI",
                "precision": "INT8",
                "calibration_size": 100
            },
            "performance_metrics": metrics
        }

        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print("\nModel exported successfully with metadata")

    except Exception as e:
        print(f"\nError exporting model: {str(e)}")
        raise

def _ocr_model(num_classes):
    """[E4] เลือกสถาปัตยกรรม recognizer — default crnn = เดิมเป๊ะ"""
    import os as _o, sys as _s
    arch = _o.environ.get('OCR_ARCH', 'crnn')
    if arch == 'crnn':
        return CRNN(num_classes=num_classes), arch
    _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'train'))
    from models_ocr_baseline import build_ocr
    return build_ocr(arch, num_classes), arch


def quantize(build_dir, quant_mode, batchsize, ckpt='best_model.pth',
             calib_split='test', calib_size=-1, quant_config=None):
    print(f"\nQuantizing model in {quant_mode} mode...")

    float_model_dir = os.path.join(build_dir, 'float_model')
    quant_model_dir = os.path.join(build_dir, 'quant_model')
    os.makedirs(float_model_dir, exist_ok=True)
    os.makedirs(quant_model_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model (ใช้โมเดล CRNN ที่ใช้ในการฝึก)
    model, _arch = _ocr_model(len(char_list))
    model = model.to(device)
    print(f'[E4] recognizer arch={_arch} params={sum(p.numel() for p in model.parameters()):,}')
    model_path = os.path.join(float_model_dir, ckpt)  # E2 finding: last_model.pth ดีกว่า best_model.pth

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print("Loading model weights...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode and freeze BN stats
    model.eval()

    def freeze_bn_stats(m):
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
    if quant_mode != 'float_test':   # float baseline ใช้ running stats ปกติ (ไม่ freeze BN)
        model.apply(freeze_bn_stats)

    print("Model loaded successfully")

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    print("Loading test dataset...")
    test_dataset = LicensePlateDataset('data/thai_license_plate_dataset_from_real_data', 'test', transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    print(f"Loaded {len(test_dataset)} test images")

    # [E8] แหล่ง/ขนาดของชุด calibration — ค่าเริ่มต้น = พฤติกรรมเดิมเป๊ะ (ใช้ test loader 100 batch แรก)
    #   หมายเหตุสำคัญ: ของเดิม calibrate บน "test split" ซึ่งไม่ถูกหลัก → --calib_split train คือโปรโตคอลที่ถูกต้อง
    if calib_split == 'test' and calib_size < 0:
        calib_loader, calib_batches = test_loader, 100
        calib_desc = 'test split (legacy behaviour), first 100 batches'
    else:
        calib_dataset = LicensePlateDataset('data/thai_license_plate_dataset_from_real_data', calib_split, transform)
        if calib_size > 0:
            calib_dataset = torch.utils.data.Subset(calib_dataset, list(range(min(calib_size, len(calib_dataset)))))
        calib_loader = DataLoader(calib_dataset, batch_size=batchsize, shuffle=False,
                                  collate_fn=collate_fn, num_workers=4)
        calib_batches = len(calib_loader)
        calib_desc = f'{calib_split} split, {len(calib_dataset)} images, {calib_batches} batches'
    print(f"[E8] calibration set: {calib_desc} | quant_config={quant_config}")

    try:
        # Calibration mode
        if quant_mode == 'calib':
            print('\nStarting calibration...')
            input_data = next(iter(calib_loader))[0].to(device)

            # Initialize quantizer for calibration
            quantizer = torch_quantizer(
                quant_mode='calib',
                module=model,
                input_args=(input_data,),
                output_dir=quant_model_dir,
                quant_config_file=quant_config   # [E8] None = DPU default (min–max, power-of-two) เหมือนเดิม
            )
            quant_model = quantizer.quant_model
            quant_model.eval()

            # Calibration process
            print("\nRunning calibration...")
            with torch.no_grad():
                for i, (data, _, _, _) in enumerate(calib_loader):
                    if i >= calib_batches:
                        break
                    print(f'Calibrating batch {i+1}/{calib_batches}')
                    data = data.to(device)
                    _ = quant_model(data)

            # Export quantization config and model
            print("\nExporting quantized model...")
            export_model(quantizer, quant_model_dir, {})

        # Float baseline testing — เทียบกับ INT8 บนชุด/โค้ดเดียวกัน [E2]
        elif quant_mode == 'float_test':
            print('\nEvaluating FLOAT model (baseline)...')
            metrics = evaluate_ocr(model, device, test_loader, label='float')
            save_ocr_results(quant_model_dir, metrics, 'float')

        # Testing mode (INT8)
        elif quant_mode == 'test':
            print('\nTesting quantized model...')
            dummy_input = next(iter(test_loader))[0].to(device)

            # Initialize quantizer for testing
            quantizer = torch_quantizer(
                quant_mode='test',
                module=model,
                input_args=(dummy_input,),
                output_dir=quant_model_dir,
                quant_config_file=quant_config   # [E8]
            )
            quant_model = quantizer.quant_model

            # Evaluate on FULL test set (deterministic) + CER [E2]
            metrics = evaluate_ocr(quant_model, device, test_loader, label='INT8')

            # Export model
            print("\nExporting quantized model...")
            export_model(quantizer, quant_model_dir, metrics)

            # Save detailed results (รวม CER)
            save_ocr_results(quant_model_dir, metrics, 'INT8')

    except Exception as e:
        print(f"\nQuantization failed: {str(e)}")
        raise

    print(f"\nResults saved in {quant_model_dir}")
    print(DIVIDER)

def run_main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--build_dir', type=str, default='build',
                        help='Path to build folder. Default is build')
    parser.add_argument('-q', '--quant_mode', type=str, default='calib',
                        choices=['calib', 'test', 'float_test'],
                        help='Mode: calib | test (INT8) | float_test (float baseline). Default calib')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='Testing batch size. Default is 32')
    parser.add_argument('--ckpt', type=str, default='best_model.pth',
                        help='checkpoint ใน float_model/ (แนะนำ last_model.pth — ดีกว่า best_model.pth)')
    # [E8] ตอบ R3-3(iii): ความไวของ recognizer ต่อชุด/วิธี calibration
    parser.add_argument('--calib_split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='ชุดที่ใช้ calibrate (default test = พฤติกรรมเดิม; train = โปรโตคอลที่ถูกต้อง)')
    parser.add_argument('--calib_size', type=int, default=-1,
                        help='จำนวนภาพ calibration (-1 = เดิม 100 batch)')
    parser.add_argument('--quant_config', type=str, default=None,
                        help='ไฟล์ json quant config (เช่น entropy) — None = DPU default min–max')
    args = parser.parse_args()

    # determinism — ให้ผลซ้ำได้ (E2)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print('\n' + DIVIDER)
    print('PyTorch Quantization for License Plate Recognition')
    print('PyTorch version: ', torch.__version__)
    print(DIVIDER)
    print('Command line options:')
    print('--build_dir   :', args.build_dir)
    print('--quant_mode  :', args.quant_mode)
    print('--batchsize   :', args.batchsize)
    print(DIVIDER)

    print('--ckpt        :', args.ckpt)
    print('--calib_split :', args.calib_split, '| --calib_size:', args.calib_size,
          '| --quant_config:', args.quant_config)
    quantize(args.build_dir, args.quant_mode, args.batchsize, args.ckpt,
             args.calib_split, args.calib_size, args.quant_config)

if __name__ == '__main__':
    run_main()
