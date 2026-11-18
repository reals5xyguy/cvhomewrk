
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = Path("split_dataset/test")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

CLASSES = ["glass", "metal", "paper", "plastic"]


class MLP(nn.Module):
    def __init__(self, num_classes=4, image_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, 512),
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


def get_model(name):
    if name == "MLP":
        model = MLP()
        image_size = 64
        filename = "mlp.pth"

    elif name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features, 4
        )
        image_size = 224
        filename = "resnet18.pth"

    elif name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.last_channel, 4
        )
        image_size = 224
        filename = "mobilenetv2.pth"

    checkpoint = torch.load(
        MODEL_DIR / filename,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_classes = checkpoint.get("classes")
        if saved_classes is not None and saved_classes != CLASSES:
            raise ValueError(
                f"{name} 클래스 순서 불일치: {saved_classes}"
            )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, image_size


def evaluate(name):
    print(f"\n===== {name} 평가 시작 =====")

    model, image_size = get_model(name)

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=test_transform
    )

    if test_dataset.classes != CLASSES:
        raise ValueError(
            f"테스트 데이터 클래스 순서 불일치: {test_dataset.classes}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    print("\n분류 결과:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0
        )
    )

    with open(
        RESULT_DIR / f"{name.lower()}_confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + CLASSES)

        for class_name, row in zip(CLASSES, cm):
            writer.writerow([class_name] + row.tolist())

    return [name, accuracy, precision, recall, f1]


def main():
    print("사용 장치:", DEVICE)

    results = []

    for name in ["MLP", "ResNet18", "MobileNetV2"]:
        results.append(evaluate(name))

    with open(
        RESULT_DIR / "model_comparison.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score"
        ])
        writer.writerows(results)

    print("\n===== 최종 테스트 결과 =====")

    for name, accuracy, precision, recall, f1 in results:
        print(
            f"{name}: "
            f"Accuracy {accuracy * 100:.2f}% | "
            f"F1-score {f1 * 100:.2f}%"
        )

    print("\n결과 저장 완료: results/model_comparison.csv")


if __name__ == "__main__":
    main()