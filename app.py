
from pathlib import Path
from io import BytesIO
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="Recycling AI")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_v2"
HTML_PATH = BASE_DIR / "templates" / "index.html"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["glass", "metal", "paper", "plastic"]

CLASS_NAMES = {
    "glass": "유리",
    "metal": "금속",
    "paper": "종이",
    "plastic": "플라스틱"
}

MAX_FILE_SIZE = 10 * 1024 * 1024


class MLP(nn.Module):
    def __init__(self):
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
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.network(x)


def create_model(name):
    if name == "MLP":
        model = MLP()
        filename = "mlp.pth"
        image_size = 64

    elif name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features, 4
        )
        filename = "resnet18.pth"
        image_size = 224

    elif name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.last_channel, 4
        )
        filename = "mobilenetv2.pth"
        image_size = 224

    else:
        raise ValueError("지원하지 않는 모델입니다.")

    checkpoint = torch.load(
        MODEL_DIR / filename,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

        saved_classes = checkpoint.get("classes")

        if saved_classes is not None:
            if saved_classes != CLASSES:
                raise ValueError(
                    f"{name} 클래스 순서가 일치하지 않습니다."
                )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, image_size


print("사용 장치:", DEVICE)

LOADED_MODELS = {}

for model_name in ["MLP", "ResNet18", "MobileNetV2"]:
    model, image_size = create_model(model_name)

    LOADED_MODELS[model_name] = {
        "model": model,
        "image_size": image_size
    }

    print(f"{model_name} 모델 로딩 완료")


def predict_image(image):
    results = {}

    for model_name, model_info in LOADED_MODELS.items():
        model = model_info["model"]
        image_size = model_info["image_size"]

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        with torch.inference_mode():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

            confidence, predicted_index = torch.max(
                probabilities, dim=0
            )

        class_name = CLASSES[predicted_index.item()]

        results[model_name] = {
            "prediction": CLASS_NAMES[class_name],
            "confidence": round(
                confidence.item() * 100, 2
            )
        }

    
    votes = [
        result["prediction"]
        for result in results.values()
    ]

    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)[0]

    if most_common[1] >= 2:
        final_prediction = most_common[0]
        decision_method = "다수결"
    else:
        final_prediction = results["MobileNetV2"]["prediction"]
        decision_method = "MobileNetV2 기준"

    return {
        "final_prediction": final_prediction,
        "vote_count": vote_counts[final_prediction],
        "decision_method": decision_method,
        "models": results
    }



@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(
        content=HTML_PATH.read_text(encoding="utf-8")
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="JPG, PNG, WEBP 이미지만 업로드할 수 있습니다."
        )

    contents = await image.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="이미지는 10MB 이하만 업로드할 수 있습니다."
        )

    try:
        uploaded_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="올바른 이미지 파일이 아닙니다."
        )

    results = await run_in_threadpool(
        predict_image,
        uploaded_image
    )

    return results
