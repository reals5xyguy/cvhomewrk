
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "split_dataset" / "test"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["glass", "metal", "paper", "plastic"]

MODEL_FILES = {
    "ResNet18_Original": "resnet18.pth",
    "ResNet18_Finetuned": "resnet18_finetuned.pth",
    "MobileNetV2_Original": "mobilenetv2.pth",
    "MobileNetV2_Finetuned": "mobilenetv2_finetuned.pth"
}


def load_model(name, filename):
    if name.startswith("ResNet18"):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)

    elif name.startswith("MobileNetV2"):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.last_channel, 4
        )

    else:
        raise ValueError("지원하지 않는 모델입니다.")

    checkpoint = torch.load(
        MODEL_DIR / filename,
        map_location="cpu",
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        saved_classes = checkpoint.get("classes")

        if (
            saved_classes is not None
            and saved_classes != CLASSES
        ):
            raise ValueError(
                f"{name}: 클래스 순서가 일치하지 않습니다."
            )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


def evaluate(name, filename, test_loader):
    print(f"\n===== {name} 평가 시작 =====")

    model = load_model(name, filename)

    y_true = []
    y_pred = []

    with torch.inference_mode():
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
            labels=list(range(4)),
            average="macro",
            zero_division=0
        )
    )

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    print("\n클래스별 평가:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(4)),
            target_names=CLASSES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(4))
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

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    if test_dataset.classes != CLASSES:
        raise ValueError(
            "테스트 데이터의 클래스 순서가 일치하지 않습니다."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("테스트 이미지 수:", len(test_dataset))

    results = []

    for name, filename in MODEL_FILES.items():
        results.append(
            evaluate(name, filename, test_loader)
        )

    with open(
        RESULT_DIR / "finetuning_comparison.csv",
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

    print("\n===== 파인튜닝 전후 성능 비교 =====")

    for name, accuracy, precision, recall, f1 in results:
        print(
            f"{name}: "
            f"Accuracy {accuracy * 100:.2f}% | "
            f"F1-score {f1 * 100:.2f}%"
        )

    print(
        "\n결과 저장 완료: "
        "results/finetuning_comparison.csv"
    )


if __name__ == "__main__":
    main()
