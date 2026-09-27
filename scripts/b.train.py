import os
import random
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# กำหนดค่าคงที่
WIDTH, HEIGHT = 220, 70
LETTERS = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
DIGITS = '0123456789'
MAX_LENGTH = 7  # ความยาวสูงสุดของป้ายทะเบียน

# การแมปตัวอักษร
CHARACTERS = LETTERS + DIGITS
char_list = ['<blank>'] + list(CHARACTERS)
char_to_idx = {char: idx for idx, char in enumerate(char_list)}
idx_to_char = {idx: char for idx, char in enumerate(char_list)}

# ฟังก์ชันสำหรับสร้างไดเรกทอรี
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# คลาส Dataset
class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        with open(os.path.join(root_dir, split, f"{split}_annotations.json"), 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.split, 'images', self.annotations[idx]['file_name'])
        image = Image.open(img_name).convert('L')
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

from models_ocr_baseline import build_ocr   # [E4] recognizer baseline ที่ deploy DPU ได้ (LPRNet)


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # CNN layers
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

        self.dropout = nn.Dropout(0.5)
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

def decode_predictions(preds):
    pred_texts = []
    for pred in preds:
        pred_indices = pred.detach().cpu().numpy()
        pred_indices = [p for i, p in enumerate(pred_indices) if (p != 0 and (i == 0 or p != pred_indices[i - 1]))]
        pred_text = ''.join([idx_to_char.get(idx, '') for idx in pred_indices])
        pred_texts.append(pred_text)
    return pred_texts

# ฟังก์ชันสำหรับทดสอบโมเดล
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    with torch.no_grad():
        for data, targets, target_lengths, texts in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            output = model(data)
            output = nn.LogSoftmax(dim=2)(output)
            T, batch_size, _ = output.size()
            input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(device)
            loss = ctc_loss(output, targets, input_lengths, target_lengths)
            test_loss += loss.item()
            preds = output.argmax(dim=2).transpose(1, 0)
            pred_texts = decode_predictions(preds)
            for pred_text, label_text in zip(pred_texts, texts):
                correct_chars += sum(1 for a, b in zip(pred_text, label_text) if a == b)
                total_chars += len(label_text)
                if pred_text == label_text:
                    correct_words += 1
                total_words += 1
    test_loss /= len(test_loader.dataset)
    word_accuracy = 100. * correct_words / total_words
    char_accuracy = 100. * correct_chars / total_chars
    print(f'Test set: Average loss: {test_loss:.4f}')
    print(f'Word-level Accuracy: {correct_words}/{total_words} ({word_accuracy:.2f}%)')
    print(f'Character-level Accuracy: {correct_chars}/{total_chars} ({char_accuracy:.2f}%)')
    return word_accuracy, char_accuracy

# ฟังก์ชันสำหรับฝึกโมเดล
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    for batch_idx, (data, targets, target_lengths, texts) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = nn.LogSoftmax(dim=2)(output)
        T, batch_size, _ = output.size()
        input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(device)
        loss = ctc_loss(output, targets, input_lengths, target_lengths)
        loss.backward()
        # ทำการตัดกราเดียนต์
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]    Loss: {loss.item():.6f}')
            preds = output.argmax(dim=2).transpose(1, 0)
            pred_texts = decode_predictions(preds)
            print(f'Predict: {pred_texts[0]}, Label: {texts[0]}')

# ฟังก์ชันหลัก
def main():
    SEED=int(os.environ.get('OCR_SEED','42'))
    os.environ['PYTHONHASHSEED']=str(SEED); random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    OUTDIR=os.environ.get('OCR_OUTDIR','outputs/build_ocr/float_model')
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'[ocr] seed={SEED} outdir={OUTDIR}')
    output_directory = "data/thai_license_plate_dataset_from_real_data"
    batch_size = 64     # เพิ่มขนาดแบทช์
    num_epochs = 200    # จำนวน epoch สูงสุด
    learning_rate = 0.0005  # ปรับค่า learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # การเพิ่มข้อมูลและการทำ normalization
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomRotation(degrees=2),  # ลดองศาการหมุน
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_dataset = LicensePlateDataset(output_directory, 'train', transform)
    val_dataset = LicensePlateDataset(output_directory, 'val', transform)
    test_dataset = LicensePlateDataset(output_directory, 'test', transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    ARCH = os.environ.get('OCR_ARCH', 'crnn')   # [E4] crnn = เดิมเป๊ะ | lprnet = baseline
    model = (CRNN(num_classes=len(char_list)) if ARCH == 'crnn'
             else build_ocr(ARCH, len(char_list))).to(device)
    print(f'[ocr] arch={ARCH} params={sum(p.numel() for p in model.parameters()):,}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # ปรับค่า step_size และ gamma

    best_accuracy = 0
    epochs_over_95 = 0  # ตัวแปรสำหรับ Early Stopping
    create_directory(OUTDIR)  # สร้างไดเรกทอรีสำหรับบันทึกโมเดล

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        word_accuracy, char_accuracy = test(model, device, val_loader)
        scheduler.step()

        # บันทึกโมเดลที่ดีที่สุด
        if word_accuracy > best_accuracy:
            best_accuracy = word_accuracy
            torch.save(model.state_dict(), os.path.join(OUTDIR,'best_model.pth'))

        # ตรวจสอบ Early Stopping
        if word_accuracy >= float(os.environ.get('OCR_ESTOP','95.0')):
            epochs_over_95 += 1
            if epochs_over_95 >= 5:
                print(f"Early stopping at epoch {epoch} as word accuracy >= 95% for 5 consecutive epochs.")
                break
        else:
            epochs_over_95 = 0

    # บันทึกโมเดลหลังจากการฝึกเสร็จสิ้น
    torch.save(model.state_dict(), os.path.join(OUTDIR,'last_model.pth'))

    # โหลดโมเดลที่ดีที่สุดและทำการทดสอบขั้นสุดท้าย
    best_model_path = os.path.join(OUTDIR,'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"No best model found at {best_model_path}. Using the last model.")
        model.load_state_dict(torch.load(os.path.join(OUTDIR,'last_model.pth')))

    final_word_accuracy, final_char_accuracy = test(model, device, test_loader)
    print(f"Final Test Word Accuracy: {final_word_accuracy:.2f}%")
    print(f"Final Test Character Accuracy: {final_char_accuracy:.2f}%")

    # บันทึกผลลัพธ์ลงไฟล์
    with open(os.path.join(OUTDIR,'test_results.txt'), 'w') as f:
        f.write(f"Final Test Word Accuracy: {final_word_accuracy:.2f}%\n")
        f.write(f"Final Test Character Accuracy: {final_char_accuracy:.2f}%\n")

    # บันทึกโมเดลในรูปแบบ TorchScript สำหรับ Vitis AI Quantizer
    example_input = torch.randn(1, 1, HEIGHT, WIDTH).to(device)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, os.path.join(OUTDIR,'float_model.pth'))

if __name__ == '__main__':
    main()
# (lpr_pt) $ python b.train.py 
# <python-env>:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
#   return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
# ---- training log of the deployed recognizer (loss/accuracy only) ----
# The per-sample "Predict / Label" lines of the original log are removed from this public copy:
# they contain real Thai plate numbers, which are personal data under Thailand's PDPA.
# Train Epoch: 1 [0/3221 (0%)]    Loss: 30.696222
# Test set: Average loss: 0.0523
# Word-level Accuracy: 0/687 (0.00%)
# Character-level Accuracy: 0/4300 (0.00%)
# Train Epoch: 2 [0/3221 (0%)]    Loss: 3.188700
# Test set: Average loss: 0.0253
# Word-level Accuracy: 3/687 (0.44%)
# Character-level Accuracy: 983/4300 (22.86%)
# Train Epoch: 3 [0/3221 (0%)]    Loss: 1.219002
# Test set: Average loss: 0.0066
# Word-level Accuracy: 325/687 (47.31%)
# Character-level Accuracy: 3535/4300 (82.21%)
# Train Epoch: 4 [0/3221 (0%)]    Loss: 0.436191
# Test set: Average loss: 0.0058
# Word-level Accuracy: 364/687 (52.98%)
# Character-level Accuracy: 3602/4300 (83.77%)
# Train Epoch: 5 [0/3221 (0%)]    Loss: 0.307682
# Test set: Average loss: 0.0034
# Word-level Accuracy: 493/687 (71.76%)
# Character-level Accuracy: 3753/4300 (87.28%)
# Train Epoch: 6 [0/3221 (0%)]    Loss: 0.159652
# Test set: Average loss: 0.0020
# Word-level Accuracy: 577/687 (83.99%)
# Character-level Accuracy: 4045/4300 (94.07%)
# Train Epoch: 7 [0/3221 (0%)]    Loss: 0.083498
# Test set: Average loss: 0.0018
# Word-level Accuracy: 600/687 (87.34%)
# Character-level Accuracy: 4077/4300 (94.81%)
# Train Epoch: 8 [0/3221 (0%)]    Loss: 0.047541
# Test set: Average loss: 0.0017
# Word-level Accuracy: 584/687 (85.01%)
# Character-level Accuracy: 4091/4300 (95.14%)
# Train Epoch: 9 [0/3221 (0%)]    Loss: 0.042429
# Test set: Average loss: 0.0027
# Word-level Accuracy: 548/687 (79.77%)
# Character-level Accuracy: 3936/4300 (91.53%)
# Train Epoch: 10 [0/3221 (0%)]    Loss: 0.134636
# Test set: Average loss: 0.0020
# Word-level Accuracy: 584/687 (85.01%)
# Character-level Accuracy: 4054/4300 (94.28%)
# Train Epoch: 11 [0/3221 (0%)]    Loss: 0.092630
# Test set: Average loss: 0.0013
# Word-level Accuracy: 610/687 (88.79%)
# Character-level Accuracy: 4126/4300 (95.95%)
# Train Epoch: 12 [0/3221 (0%)]    Loss: 0.067411
# Test set: Average loss: 0.0016
# Word-level Accuracy: 602/687 (87.63%)
# Character-level Accuracy: 4130/4300 (96.05%)
# Train Epoch: 13 [0/3221 (0%)]    Loss: 0.037743
# Test set: Average loss: 0.0013
# Word-level Accuracy: 616/687 (89.67%)
# Character-level Accuracy: 4103/4300 (95.42%)
# Train Epoch: 14 [0/3221 (0%)]    Loss: 0.133481
# Test set: Average loss: 0.0014
# Word-level Accuracy: 596/687 (86.75%)
# Character-level Accuracy: 4086/4300 (95.02%)
# Train Epoch: 15 [0/3221 (0%)]    Loss: 0.029997
# Test set: Average loss: 0.0017
# Word-level Accuracy: 604/687 (87.92%)
# Character-level Accuracy: 4042/4300 (94.00%)
# Train Epoch: 16 [0/3221 (0%)]    Loss: 0.078353
# Test set: Average loss: 0.0011
# Word-level Accuracy: 633/687 (92.14%)
# Character-level Accuracy: 4150/4300 (96.51%)
# Train Epoch: 17 [0/3221 (0%)]    Loss: 0.013628
# Test set: Average loss: 0.0011
# Word-level Accuracy: 627/687 (91.27%)
# Character-level Accuracy: 4124/4300 (95.91%)
# Train Epoch: 18 [0/3221 (0%)]    Loss: 0.051174
# Test set: Average loss: 0.0010
# Word-level Accuracy: 623/687 (90.68%)
# Character-level Accuracy: 4135/4300 (96.16%)
# Train Epoch: 19 [0/3221 (0%)]    Loss: 0.091541
# Test set: Average loss: 0.0013
# Word-level Accuracy: 617/687 (89.81%)
# Character-level Accuracy: 4136/4300 (96.19%)
# Train Epoch: 20 [0/3221 (0%)]    Loss: 0.037793
# Test set: Average loss: 0.0014
# Word-level Accuracy: 619/687 (90.10%)
# Character-level Accuracy: 4128/4300 (96.00%)
# Train Epoch: 21 [0/3221 (0%)]    Loss: 0.011521
# Test set: Average loss: 0.0012
# Word-level Accuracy: 617/687 (89.81%)
# Character-level Accuracy: 4073/4300 (94.72%)
# Train Epoch: 22 [0/3221 (0%)]    Loss: 0.038152
# Test set: Average loss: 0.0011
# Word-level Accuracy: 628/687 (91.41%)
# Character-level Accuracy: 4145/4300 (96.40%)
# Train Epoch: 23 [0/3221 (0%)]    Loss: 0.058001
# Test set: Average loss: 0.0011
# Word-level Accuracy: 640/687 (93.16%)
# Character-level Accuracy: 4155/4300 (96.63%)
# Train Epoch: 24 [0/3221 (0%)]    Loss: 0.012604
# Test set: Average loss: 0.0010
# Word-level Accuracy: 629/687 (91.56%)
# Character-level Accuracy: 4170/4300 (96.98%)
# Train Epoch: 25 [0/3221 (0%)]    Loss: 0.010391
# Test set: Average loss: 0.0009
# Word-level Accuracy: 635/687 (92.43%)
# Character-level Accuracy: 4171/4300 (97.00%)
# Train Epoch: 26 [0/3221 (0%)]    Loss: 0.067855
# Test set: Average loss: 0.0011
# Word-level Accuracy: 629/687 (91.56%)
# Character-level Accuracy: 4145/4300 (96.40%)
# Train Epoch: 27 [0/3221 (0%)]    Loss: 0.004197
# Test set: Average loss: 0.0010
# Word-level Accuracy: 631/687 (91.85%)
# Character-level Accuracy: 4154/4300 (96.60%)
# Train Epoch: 28 [0/3221 (0%)]    Loss: 0.024667
# Test set: Average loss: 0.0008
# Word-level Accuracy: 646/687 (94.03%)
# Character-level Accuracy: 4210/4300 (97.91%)
# Train Epoch: 29 [0/3221 (0%)]    Loss: 0.011704
# Test set: Average loss: 0.0018
# Word-level Accuracy: 596/687 (86.75%)
# Character-level Accuracy: 4102/4300 (95.40%)
# Train Epoch: 30 [0/3221 (0%)]    Loss: 0.069443
# Test set: Average loss: 0.0012
# Word-level Accuracy: 628/687 (91.41%)
# Character-level Accuracy: 4148/4300 (96.47%)
# Train Epoch: 31 [0/3221 (0%)]    Loss: 0.027078
# Test set: Average loss: 0.0010
# Word-level Accuracy: 640/687 (93.16%)
# Character-level Accuracy: 4181/4300 (97.23%)
# Train Epoch: 32 [0/3221 (0%)]    Loss: 0.086525
# Test set: Average loss: 0.0013
# Word-level Accuracy: 626/687 (91.12%)
# Character-level Accuracy: 4170/4300 (96.98%)
# Train Epoch: 33 [0/3221 (0%)]    Loss: 0.025764
# Test set: Average loss: 0.0012
# Word-level Accuracy: 616/687 (89.67%)
# Character-level Accuracy: 4114/4300 (95.67%)
# Train Epoch: 34 [0/3221 (0%)]    Loss: 0.012689
# Test set: Average loss: 0.0021
# Word-level Accuracy: 601/687 (87.48%)
# Character-level Accuracy: 4080/4300 (94.88%)
# Train Epoch: 35 [0/3221 (0%)]    Loss: 0.033544
# Test set: Average loss: 0.0007
# Word-level Accuracy: 648/687 (94.32%)
# Character-level Accuracy: 4201/4300 (97.70%)
# Train Epoch: 36 [0/3221 (0%)]    Loss: 0.013132
# Test set: Average loss: 0.0011
# Word-level Accuracy: 619/687 (90.10%)
# Character-level Accuracy: 4180/4300 (97.21%)
# Train Epoch: 37 [0/3221 (0%)]    Loss: 0.013129
# Test set: Average loss: 0.0010
# Word-level Accuracy: 631/687 (91.85%)
# Character-level Accuracy: 4164/4300 (96.84%)
# Train Epoch: 38 [0/3221 (0%)]    Loss: 0.018332
# Test set: Average loss: 0.0012
# Word-level Accuracy: 625/687 (90.98%)
# Character-level Accuracy: 4141/4300 (96.30%)
# Train Epoch: 39 [0/3221 (0%)]    Loss: 0.041492
# Test set: Average loss: 0.0009
# Word-level Accuracy: 640/687 (93.16%)
# Character-level Accuracy: 4181/4300 (97.23%)
# Train Epoch: 40 [0/3221 (0%)]    Loss: 0.048646
# Test set: Average loss: 0.0009
# Word-level Accuracy: 645/687 (93.89%)
# Character-level Accuracy: 4197/4300 (97.60%)
# Train Epoch: 41 [0/3221 (0%)]    Loss: 0.015273
# Test set: Average loss: 0.0024
# Word-level Accuracy: 576/687 (83.84%)
# Character-level Accuracy: 4117/4300 (95.74%)
# Train Epoch: 42 [0/3221 (0%)]    Loss: 0.021365
# Test set: Average loss: 0.0011
# Word-level Accuracy: 641/687 (93.30%)
# Character-level Accuracy: 4182/4300 (97.26%)
# Train Epoch: 43 [0/3221 (0%)]    Loss: 0.008310
# Test set: Average loss: 0.0010
# Word-level Accuracy: 635/687 (92.43%)
# Character-level Accuracy: 4178/4300 (97.16%)
# Train Epoch: 44 [0/3221 (0%)]    Loss: 0.005631
# Test set: Average loss: 0.0008
# Word-level Accuracy: 648/687 (94.32%)
# Character-level Accuracy: 4188/4300 (97.40%)
# Train Epoch: 45 [0/3221 (0%)]    Loss: 0.016346
# Test set: Average loss: 0.0008
# Word-level Accuracy: 647/687 (94.18%)
# Character-level Accuracy: 4209/4300 (97.88%)
# Train Epoch: 46 [0/3221 (0%)]    Loss: 0.030591
# Test set: Average loss: 0.0008
# Word-level Accuracy: 648/687 (94.32%)
# Character-level Accuracy: 4205/4300 (97.79%)
# Train Epoch: 47 [0/3221 (0%)]    Loss: 0.025442
# Test set: Average loss: 0.0012
# Word-level Accuracy: 623/687 (90.68%)
# Character-level Accuracy: 4126/4300 (95.95%)
# Train Epoch: 48 [0/3221 (0%)]    Loss: 0.006541
# Test set: Average loss: 0.0038
# Word-level Accuracy: 564/687 (82.10%)
# Character-level Accuracy: 4055/4300 (94.30%)
# Train Epoch: 49 [0/3221 (0%)]    Loss: 0.098050
# Test set: Average loss: 0.0009
# Word-level Accuracy: 639/687 (93.01%)
# Character-level Accuracy: 4157/4300 (96.67%)
# Train Epoch: 50 [0/3221 (0%)]    Loss: 0.008124
# Test set: Average loss: 0.0010
# Word-level Accuracy: 636/687 (92.58%)
# Character-level Accuracy: 4171/4300 (97.00%)
# Train Epoch: 51 [0/3221 (0%)]    Loss: 0.019239
# Test set: Average loss: 0.0007
# Word-level Accuracy: 651/687 (94.76%)
# Character-level Accuracy: 4221/4300 (98.16%)
# Train Epoch: 52 [0/3221 (0%)]    Loss: 0.009463
# Test set: Average loss: 0.0007
# Word-level Accuracy: 648/687 (94.32%)
# Character-level Accuracy: 4220/4300 (98.14%)
# Train Epoch: 53 [0/3221 (0%)]    Loss: 0.003872
# Test set: Average loss: 0.0006
# Word-level Accuracy: 657/687 (95.63%)
# Character-level Accuracy: 4227/4300 (98.30%)
# Train Epoch: 54 [0/3221 (0%)]    Loss: 0.000382
# Test set: Average loss: 0.0007
# Word-level Accuracy: 656/687 (95.49%)
# Character-level Accuracy: 4214/4300 (98.00%)
# Train Epoch: 55 [0/3221 (0%)]    Loss: 0.002281
# Test set: Average loss: 0.0006
# Word-level Accuracy: 655/687 (95.34%)
# Character-level Accuracy: 4221/4300 (98.16%)
# Train Epoch: 56 [0/3221 (0%)]    Loss: 0.002636
# Test set: Average loss: 0.0006
# Word-level Accuracy: 656/687 (95.49%)
# Character-level Accuracy: 4240/4300 (98.60%)
# Train Epoch: 57 [0/3221 (0%)]    Loss: 0.000355
# Test set: Average loss: 0.0006
# Word-level Accuracy: 654/687 (95.20%)
# Character-level Accuracy: 4221/4300 (98.16%)
# Early stopping at epoch 57 as word accuracy >= 95% for 5 consecutive epochs.
# Test set: Average loss: 0.0007
# Word-level Accuracy: 675/703 (96.02%)
# Character-level Accuracy: 4340/4427 (98.03%)
# Final Test Word Accuracy: 96.02%
# Final Test Character Accuracy: 98.03%