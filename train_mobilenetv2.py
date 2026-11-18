
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

DATA_DIR = Path("split_dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("분류 클래스:", train_dataset.classes)
print("학습 이미지:", len(train_dataset))
print("검증 이미지:", len(val_dataset))

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(train_dataset.classes)
)

model = model.to(device)

print(
    "MobileNetV2 파라미터 수:",
    sum(p.numel() for p in model.parameters())
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_acc = -1.0
history = []

for epoch in range(EPOCHS):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    avg_val_loss = val_loss / val_total

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    history.append([
        epoch + 1,
        avg_train_loss,
        train_acc,
        avg_val_loss,
        val_acc
    ])

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Train Acc: {train_acc * 100:.2f}% "
        f"Val Loss: {avg_val_loss:.4f} "
        f"Val Acc: {val_acc * 100:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,
            "image_size": IMAGE_SIZE,
            "epoch": epoch + 1,
            "val_accuracy": val_acc
        }, MODEL_DIR / "mobilenetv2.pth")

        print("최고 성능 MobileNetV2 모델 저장 완료!")

with open(
    MODEL_DIR / "mobilenetv2_history.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ])
    writer.writerows(history)

print("\nMobileNetV2 학습 완료!")
print(f"최고 검증 정확도: {best_val_acc * 100:.2f}%")
print("모델 저장 위치: models/mobilenetv2.pth")
