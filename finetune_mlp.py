
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
IMAGE_SIZE = 64
PATIENCE = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "split_dataset"
MODEL_DIR = BASE_DIR / "models"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

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
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("분류 클래스:", train_dataset.classes)


class MLP(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = MLP(
    num_classes=len(train_dataset.classes)
)

checkpoint = torch.load(
    MODEL_DIR / "mlp.pth",
    map_location="cpu",
    weights_only=True
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    saved_classes = checkpoint.get("classes")

    if (
        saved_classes is not None
        and saved_classes != train_dataset.classes
    ):
        raise ValueError("클래스 순서가 일치하지 않습니다.")
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model = model.to(device)

print("기존 MLP 가중치 불러오기 완료")

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)

best_val_acc = -1.0
no_improvement = 0
history = []

for epoch in range(EPOCHS):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += labels.size(0)

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    avg_train_loss = train_loss / train_total
    avg_val_loss = val_loss / val_total

    scheduler.step(val_acc)

    history.append([
        epoch + 1,
        avg_train_loss,
        train_acc,
        avg_val_loss,
        val_acc
    ])

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Acc: {train_acc * 100:.2f}% "
        f"Val Acc: {val_acc * 100:.2f}% "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,
            "image_size": IMAGE_SIZE,
            "epoch": epoch + 1,
            "val_accuracy": val_acc
        }, MODEL_DIR / "mlp_finetuned.pth")

        print("최고 성능 MLP 모델 저장 완료!")

    else:
        no_improvement += 1

    if no_improvement >= PATIENCE:
        print("조기 종료: 검증 정확도 개선 없음")
        break

with open(
    MODEL_DIR / "mlp_finetuned_history.csv",
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

print("\nMLP 파인튜닝 완료!")
print(f"최고 검증 정확도: {best_val_acc * 100:.2f}%")
print("저장 위치: models/mlp_finetuned.pth")
